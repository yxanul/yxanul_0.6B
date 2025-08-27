#!/usr/bin/env python3
"""
Performance Analysis and Optimization Suggestions for Yxanul Training

This script analyzes current training metrics and suggests optimizations.
"""

import torch
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pynvml

@dataclass
class PerformanceMetrics:
    """Current performance metrics from training"""
    tokens_per_sec: float = 19000
    gpu_utilization: float = 65  # %
    vram_usage: float = 22  # %
    batch_size: int = 1
    seq_length: int = 2048
    gradient_accumulation: int = 32
    model_params: float = 273.3e6
    vocab_size: int = 200005
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation
    
    @property
    def tokens_per_batch(self) -> int:
        return self.batch_size * self.seq_length
    
    @property
    def optimizer_frequency(self) -> int:
        """How many forward passes before optimizer step"""
        return self.gradient_accumulation


class PerformanceAnalyzer:
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        self.suggestions = []
    
    def analyze_bottleneck(self) -> str:
        """Identify the primary bottleneck"""
        if self.metrics.vram_usage < 30:
            return "memory_underutilized"
        elif self.metrics.gpu_utilization < 70:
            return "compute_underutilized"
        elif self.metrics.gpu_utilization > 95:
            return "compute_bound"
        else:
            return "balanced"
    
    def suggest_batch_size(self) -> int:
        """Calculate optimal batch size based on VRAM usage"""
        # RTX 5090 has 32GB VRAM
        total_vram_gb = 32
        current_usage_gb = total_vram_gb * (self.metrics.vram_usage / 100)
        
        # Leave 10% headroom for safety
        target_usage = 0.9
        available_vram = total_vram_gb * target_usage
        
        # Estimate how much more we can fit
        scale_factor = available_vram / current_usage_gb
        
        # Batch size scales linearly with memory for activations
        # But we need to be conservative due to vocab size
        suggested_batch = min(
            int(self.metrics.batch_size * scale_factor * 0.7),  # 0.7 safety factor
            8  # Max reasonable for 200k vocab
        )
        
        # Round to power of 2 for efficiency
        return 2 ** int(math.log2(max(suggested_batch, 1)))
    
    def suggest_gradient_accumulation(self, new_batch_size: int) -> int:
        """Adjust gradient accumulation for new batch size"""
        # Maintain similar effective batch size
        target_effective = 32  # Good for stability
        
        suggested = max(1, target_effective // new_batch_size)
        
        # Prefer factors of 2
        if suggested > 1:
            suggested = 2 ** int(math.log2(suggested) + 0.5)
        
        return suggested
    
    def calculate_expected_speedup(self, new_batch_size: int, new_grad_accum: int) -> float:
        """Estimate speedup from new configuration"""
        # Fewer gradient accumulation steps = less overhead
        overhead_reduction = self.metrics.gradient_accumulation / new_grad_accum
        
        # Larger batches = better GPU utilization
        utilization_improvement = math.sqrt(new_batch_size / self.metrics.batch_size)
        
        # Conservative estimate
        speedup = 1.0 + (overhead_reduction - 1.0) * 0.3 + (utilization_improvement - 1.0) * 0.5
        
        return speedup
    
    def generate_report(self) -> str:
        """Generate optimization report"""
        bottleneck = self.analyze_bottleneck()
        new_batch = self.suggest_batch_size()
        new_grad_accum = self.suggest_gradient_accumulation(new_batch)
        expected_speedup = self.calculate_expected_speedup(new_batch, new_grad_accum)
        
        report = []
        report.append("="*60)
        report.append("🔍 PERFORMANCE ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        # Current status
        report.append("📊 Current Training Metrics:")
        report.append(f"  • Tokens/sec: {self.metrics.tokens_per_sec:,}")
        report.append(f"  • GPU Utilization: {self.metrics.gpu_utilization}%")
        report.append(f"  • VRAM Usage: {self.metrics.vram_usage}%")
        report.append(f"  • Batch Size: {self.metrics.batch_size}")
        report.append(f"  • Gradient Accumulation: {self.metrics.gradient_accumulation}")
        report.append(f"  • Effective Batch Size: {self.metrics.effective_batch_size}")
        report.append("")
        
        # Bottleneck analysis
        report.append("🎯 Bottleneck Analysis:")
        if bottleneck == "memory_underutilized":
            report.append("  ⚠️  SEVERE MEMORY UNDERUTILIZATION DETECTED")
            report.append("  You're using only 22% of available VRAM!")
            report.append("  This is leaving significant performance on the table.")
        elif bottleneck == "compute_underutilized":
            report.append("  ⚠️  GPU COMPUTE UNDERUTILIZED")
            report.append("  GPU is not fully saturated with work.")
        else:
            report.append("  ✓ System is reasonably balanced")
        report.append("")
        
        # Recommendations
        report.append("💡 OPTIMIZATION RECOMMENDATIONS:")
        report.append("")
        
        if new_batch > self.metrics.batch_size:
            report.append(f"1. INCREASE BATCH SIZE: {self.metrics.batch_size} → {new_batch}")
            report.append(f"   Rationale: With only {self.metrics.vram_usage}% VRAM usage,")
            report.append(f"   we can safely increase batch size to better utilize GPU.")
            report.append("")
        
        if new_grad_accum < self.metrics.gradient_accumulation:
            report.append(f"2. REDUCE GRADIENT ACCUMULATION: {self.metrics.gradient_accumulation} → {new_grad_accum}")
            report.append(f"   Rationale: With larger batch size, we need less accumulation.")
            report.append(f"   This reduces optimizer overhead and improves throughput.")
            report.append("")
        
        # Additional optimizations
        report.append("3. ADDITIONAL OPTIMIZATIONS TO CONSIDER:")
        report.append("   • Enable torch.compile() for 10-20% speedup")
        report.append("   • Use Flash Attention 3 (if not already enabled)")
        report.append("   • Consider mixed precision (FP8 already enabled ✓)")
        report.append("   • Prefetch next batch during backward pass")
        report.append("")
        
        # Expected improvements
        report.append("📈 EXPECTED IMPROVEMENTS:")
        report.append(f"  • Estimated Speedup: {expected_speedup:.1f}x")
        report.append(f"  • New Tokens/sec: ~{int(self.metrics.tokens_per_sec * expected_speedup):,}")
        report.append(f"  • Time to 1B tokens: ~{1e9 / (self.metrics.tokens_per_sec * expected_speedup * 3600):.1f} hours")
        report.append("")
        
        # Implementation commands
        report.append("🚀 IMPLEMENTATION:")
        report.append("  To apply these optimizations, run:")
        report.append(f"  python train_te_v2.py --batch-size {new_batch} \\")
        report.append(f"                        --gradient-accumulation {new_grad_accum} \\")
        report.append("                        --config configs/yxanul_270m_fixed_2048.yaml")
        report.append("")
        
        # Monitoring advice
        report.append("📊 MONITORING:")
        report.append("  Watch for:")
        report.append("  • VRAM usage should increase to 70-80%")
        report.append("  • GPU utilization should reach 80-90%")
        report.append("  • Tokens/sec should increase by ~50%")
        report.append("  • Loss curve should remain stable")
        report.append("")
        
        # Warnings
        if new_batch > 4:
            report.append("⚠️  WARNING:")
            report.append(f"  Batch size {new_batch} is aggressive for 200k vocabulary.")
            report.append("  Monitor for OOM errors and reduce if needed.")
            report.append("")
        
        report.append("="*60)
        
        return "\n".join(report)


def check_gpu_capabilities():
    """Check GPU capabilities and features"""
    if not torch.cuda.is_available():
        return "No GPU available"
    
    info = []
    info.append("\n🖥️  GPU CAPABILITIES:")
    
    # Basic info
    gpu_name = torch.cuda.get_device_name()
    info.append(f"  Device: {gpu_name}")
    
    # Memory
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    info.append(f"  Total Memory: {total_mem:.1f} GB")
    
    # Compute capability
    major, minor = torch.cuda.get_device_capability()
    info.append(f"  Compute Capability: {major}.{minor}")
    
    # Check for specific features
    if major >= 8:
        info.append("  ✓ FP8 Support (Ampere/Ada/Hopper)")
    if major >= 9:
        info.append("  ✓ Flash Attention 3 Support")
    
    # Check TransformerEngine
    try:
        import transformer_engine as te
        info.append(f"  ✓ TransformerEngine v{te.__version__}")
    except:
        info.append("  ✗ TransformerEngine not available")
    
    return "\n".join(info)


def main():
    """Run performance analysis"""
    
    # Current metrics from training
    metrics = PerformanceMetrics(
        tokens_per_sec=19000,
        gpu_utilization=65,
        vram_usage=22,
        batch_size=1,
        seq_length=2048,
        gradient_accumulation=32,
        model_params=273.3e6,
        vocab_size=200005
    )
    
    # Run analysis
    analyzer = PerformanceAnalyzer(metrics)
    report = analyzer.generate_report()
    
    # Print report
    print(report)
    
    # Add GPU capabilities
    print(check_gpu_capabilities())
    
    # Memory calculation details
    print("\n📐 MEMORY CALCULATION DETAILS:")
    print(f"  Model Parameters: {metrics.model_params/1e6:.1f}M")
    print(f"  Model Size (BF16): {metrics.model_params * 2 / 1e9:.2f} GB")
    print(f"  Optimizer States (AdamW): {metrics.model_params * 8 / 1e9:.2f} GB")
    print(f"  Activation Memory (per sample): ~{metrics.seq_length * 640 * 4 * 28 / 1e6:.0f} MB")
    print(f"  Output Logits (per sample): {metrics.seq_length * metrics.vocab_size * 2 / 1e9:.2f} GB")
    print(f"  Total for batch={metrics.batch_size}: ~{32 * 0.22:.1f} GB used")


if __name__ == "__main__":
    main()