#!/usr/bin/env python3
"""
Multi-domain validation system for measuring domain-specific perplexity.
This helps tune training data ratios by showing which domains the model struggles with.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm


class DomainValidationDataset(Dataset):
    """Single domain validation dataset"""
    
    def __init__(self, domain: str, tokenizer, max_length: int = 2048, max_samples: int = 1000):
        self.domain = domain
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load domain-specific data from validation directory
        validation_dir = Path("validation")
        
        if domain == "english":
            # C4 validation for general English
            parquet_file = validation_dir / "c4_validation_2k.parquet"
            if not parquet_file.exists():
                # Fallback to old location
                parquet_file = Path("c4-validation.parquet")
            
            df = pd.read_parquet(parquet_file)
            # Sample subset for efficiency
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
            self.samples = df['text'].tolist()
            
        elif domain == "math":
            # GSM8K for math reasoning
            parquet_file = validation_dir / "gsm8k_validation.parquet"
            if not parquet_file.exists():
                # Fallback to old location
                parquet_file = Path("test-00000-of-00001.parquet")
            
            df = pd.read_parquet(parquet_file)
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
            # Combine question and answer for full context
            self.samples = [
                f"Question: {row['question']}\nAnswer: {row['answer']}"
                for _, row in df.iterrows()
            ]
            
        elif domain == "code":
            # HumanEval for code generation
            parquet_file = validation_dir / "humaneval_validation.parquet"
            if not parquet_file.exists():
                # Fallback to old location
                parquet_file = Path("test-00000-of-00001 (1).parquet")
            
            df = pd.read_parquet(parquet_file)
            # Use all HumanEval samples (only 164)
            self.samples = [
                row['prompt'] + row['canonical_solution']
                for _, row in df.iterrows()
            ]
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        print(f"Loaded {len(self.samples)} samples for {domain} validation")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Create input_ids and labels (same for language modeling)
        input_ids = encoded['input_ids']
        
        # Pad to max_length for batch processing
        if len(input_ids) < self.max_length:
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(encoded['input_ids']) + [0] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            attention_mask = [1] * self.max_length
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(input_ids, dtype=torch.long),  # Same as input for LM
            'domain': self.domain
        }


class MultiDomainValidator:
    """Runs validation on multiple domains and reports domain-specific metrics"""
    
    def __init__(self, tokenizer, batch_size: int = 32, max_samples_per_domain: int = 1000):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_samples = max_samples_per_domain
        
        # Create domain-specific datasets
        self.domains = ["english", "math", "code"]
        self.datasets = {}
        self.dataloaders = {}
        
        for domain in self.domains:
            try:
                dataset = DomainValidationDataset(
                    domain, 
                    tokenizer, 
                    max_samples=max_samples_per_domain
                )
                self.datasets[domain] = dataset
                
                # Create dataloader with proper batch size
                # Ensure batch size is multiple of 8 for FP8
                batch_size = (batch_size // 8) * 8
                if batch_size == 0:
                    batch_size = 8
                    
                self.dataloaders[domain] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    drop_last=False
                )
                
            except Exception as e:
                print(f"Warning: Failed to load {domain} validation: {e}")
                continue
    
    def validate_domain(self, model, domain: str, max_batches: int = 50, device: str = 'cuda') -> Dict:
        """Validate on a single domain"""
        
        if domain not in self.dataloaders:
            return {'error': f'No dataloader for domain {domain}'}
        
        model.eval()
        dataloader = self.dataloaders[domain]
        
        total_loss = 0
        total_tokens = 0
        all_losses = []
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader, desc=f"Validating {domain}", leave=False)):
                if i >= max_batches:
                    break
                
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                # Get loss (handle both return formats)
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                else:
                    loss = outputs.loss
                
                # Track metrics
                batch_tokens = attention_mask.sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens
                all_losses.append(loss.item())
        
        # Calculate metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            'domain': domain,
            'loss': avg_loss,
            'perplexity': perplexity,
            'total_tokens': total_tokens,
            'num_batches': min(i + 1, max_batches),
            'loss_std': np.std(all_losses) if all_losses else 0
        }
    
    def validate_all_domains(self, model, max_batches: int = 50, device: str = 'cuda') -> Dict:
        """Run validation on all domains and return comprehensive metrics"""
        
        results = {}
        
        # Adjust max_batches per domain based on dataset size
        domain_max_batches = {
            'english': min(max_batches, 60),  # C4 has ~62 batches
            'math': min(max_batches, 40),      # GSM8K has ~41 batches  
            'code': min(max_batches, 5)        # HumanEval has only ~5 batches
        }
        
        for domain in self.domains:
            print(f"\n{'='*50}")
            print(f"Validating on {domain.upper()} domain")
            print(f"{'='*50}")
            
            # Use domain-specific batch limit
            domain_batches = domain_max_batches.get(domain, max_batches)
            metrics = self.validate_domain(model, domain, domain_batches, device)
            results[domain] = metrics
            
            # Print immediate results
            if 'error' not in metrics:
                print(f"  Loss: {metrics['loss']:.4f}")
                print(f"  Perplexity: {metrics['perplexity']:.2f}")
                print(f"  Tokens evaluated: {metrics['total_tokens']:,}")
        
        # Calculate aggregate metrics
        valid_results = [r for r in results.values() if 'error' not in r]
        if valid_results:
            results['aggregate'] = {
                'avg_loss': np.mean([r['loss'] for r in valid_results]),
                'avg_perplexity': np.mean([r['perplexity'] for r in valid_results]),
                'loss_variance': np.var([r['loss'] for r in valid_results]),
                'domains_evaluated': len(valid_results)
            }
            
            # Calculate surprise ratios (how much worse each domain is vs average)
            avg_ppl = results['aggregate']['avg_perplexity']
            for domain in self.domains:
                if domain in results and 'perplexity' in results[domain]:
                    results[domain]['surprise_ratio'] = results[domain]['perplexity'] / avg_ppl
        
        return results
    
    def print_summary(self, results: Dict):
        """Print a nice summary of validation results"""
        
        print("\n" + "="*60)
        print("MULTI-DOMAIN VALIDATION SUMMARY")
        print("="*60)
        
        # Domain-specific results
        print("\nDomain-Specific Performance:")
        print("-"*40)
        
        for domain in self.domains:
            if domain in results and 'error' not in results[domain]:
                r = results[domain]
                print(f"\n{domain.upper()}:")
                print(f"  Perplexity: {r['perplexity']:.2f}")
                print(f"  Loss: {r['loss']:.4f}")
                if 'surprise_ratio' in r:
                    print(f"  Surprise Ratio: {r['surprise_ratio']:.2f}x")
                    if r['surprise_ratio'] > 1.5:
                        print(f"  ⚠️  Model struggles with {domain} - consider increasing training ratio")
                    elif r['surprise_ratio'] < 0.8:
                        print(f"  ✓ Model excels at {domain} - could reduce training ratio")
        
        # Aggregate results
        if 'aggregate' in results:
            print("\n" + "-"*40)
            print("Aggregate Metrics:")
            print(f"  Average Perplexity: {results['aggregate']['avg_perplexity']:.2f}")
            print(f"  Average Loss: {results['aggregate']['avg_loss']:.4f}")
            print(f"  Cross-domain Variance: {results['aggregate']['loss_variance']:.4f}")
            
            # Recommendations
            print("\n" + "-"*40)
            print("Training Mix Recommendations:")
            
            for domain in self.domains:
                if domain in results and 'surprise_ratio' in results[domain]:
                    ratio = results[domain]['surprise_ratio']
                    if ratio > 1.3:
                        print(f"  📈 Increase {domain} data (currently underperforming)")
                    elif ratio < 0.85:
                        print(f"  📉 Consider reducing {domain} data (already strong)")
                    else:
                        print(f"  ✅ {domain} ratio seems well-balanced")
        
        print("\n" + "="*60)
        
        return results


def integrate_with_trainer(trainer_instance):
    """Helper function to integrate multi-domain validation with existing trainer"""
    
    # Create multi-domain validator
    validator = MultiDomainValidator(
        tokenizer=trainer_instance.tokenizer,
        batch_size=trainer_instance.config.get('validation', {}).get('per_device_eval_batch_size', 32),
        max_samples_per_domain=1000
    )
    
    # Run validation
    results = validator.validate_all_domains(
        model=trainer_instance.model,
        max_batches=50,  # Efficient validation
        device=trainer_instance.device
    )
    
    # Print summary
    validator.print_summary(results)
    
    # Log to wandb if available
    if hasattr(trainer_instance, 'wandb_run') and trainer_instance.wandb_run:
        import wandb
        # Flatten results for wandb
        wandb_metrics = {}
        for domain in validator.domains:
            if domain in results and 'error' not in results[domain]:
                for key, value in results[domain].items():
                    if key != 'domain':
                        wandb_metrics[f"val/{domain}_{key}"] = value
        
        if 'aggregate' in results:
            for key, value in results['aggregate'].items():
                wandb_metrics[f"val/aggregate_{key}"] = value
        
        wandb.log(wandb_metrics, step=trainer_instance.global_step)
    
    return results


if __name__ == "__main__":
    # Test the validation system
    from data_pipeline import create_tokenizer
    
    print("Testing Multi-Domain Validation System")
    print("="*60)
    
    # Create tokenizer
    tokenizer = create_tokenizer()
    
    # Create validator
    validator = MultiDomainValidator(tokenizer, batch_size=8, max_samples_per_domain=100)
    
    print("\nDomains loaded:", list(validator.datasets.keys()))
    for domain, dataset in validator.datasets.items():
        print(f"  {domain}: {len(dataset)} samples")