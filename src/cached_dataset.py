"""
Cached dataset wrapper to prevent reloading data from disk.
Loads once, keeps in memory, serves fast.
"""

import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
import gc

class CachedDataset(Dataset):
    """Wrapper that caches the entire dataset in memory after first load."""
    
    # Class-level cache to persist across dataloader recreations
    _cache = {}
    
    def __init__(
        self,
        base_dataset: Dataset,
        cache_key: str = "default",
        max_cache_size: int = 100000,
        device: str = "cpu"  # Keep cache on CPU to save GPU memory
    ):
        """
        Args:
            base_dataset: The underlying dataset to cache
            cache_key: Unique key for this dataset in the cache
            max_cache_size: Maximum number of samples to cache
            device: Device to store cached tensors
        """
        self.base_dataset = base_dataset
        self.cache_key = cache_key
        self.max_cache_size = max_cache_size
        self.device = device
        
        # Check if already cached
        if cache_key not in self._cache:
            print(f"[CachedDataset] Building cache for '{cache_key}'...")
            self._build_cache()
        else:
            print(f"[CachedDataset] Using existing cache for '{cache_key}' ({len(self._cache[cache_key])} samples)")
            
        self.cached_data = self._cache[cache_key]
        self.length = len(self.cached_data)
    
    def _build_cache(self):
        """Load all data into memory once."""
        cache = []
        
        # Determine how many samples to cache
        dataset_len = len(self.base_dataset)
        samples_to_cache = min(dataset_len, self.max_cache_size)
        
        print(f"[CachedDataset] Caching {samples_to_cache} samples...")
        
        # Load samples into cache
        for idx in range(samples_to_cache):
            if idx % 10000 == 0:
                print(f"  Cached {idx}/{samples_to_cache} samples...")
            
            sample = self.base_dataset[idx]
            
            # Move tensors to specified device and detach from graph
            cached_sample = {}
            for key, value in sample.items():
                if torch.is_tensor(value):
                    # Keep on CPU to save GPU memory, will move during training
                    cached_sample[key] = value.detach().cpu()
                else:
                    cached_sample[key] = value
            
            cache.append(cached_sample)
        
        # Store in class-level cache
        self._cache[self.cache_key] = cache
        
        # Force garbage collection to free memory from base dataset
        del self.base_dataset
        gc.collect()
        
        print(f"[CachedDataset] Cache complete! {len(cache)} samples ready.")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Return cached sample - already tokenized and ready."""
        if idx >= self.length:
            raise IndexError(f"Index {idx} out of range for cache size {self.length}")
        
        # Return a copy to avoid modifying cache
        sample = self.cached_data[idx]
        return {k: v.clone() if torch.is_tensor(v) else v for k, v in sample.items()}
    
    @classmethod
    def clear_cache(cls, cache_key: Optional[str] = None):
        """Clear cached data to free memory."""
        if cache_key:
            if cache_key in cls._cache:
                del cls._cache[cache_key]
                print(f"[CachedDataset] Cleared cache for '{cache_key}'")
        else:
            cls._cache.clear()
            print("[CachedDataset] Cleared all caches")
        gc.collect()


def create_cached_dataloader(
    dataset_name: str,
    tokenizer,
    batch_size: int = 1,
    max_length: int = 2048,
    num_workers: int = 0,  # Should be 0 for cached dataset
    split: str = "train",
    cache_key: Optional[str] = None,
    **kwargs
):
    """Create a dataloader with cached dataset for fast iteration."""
    
    from data_pipeline import create_dataloader
    
    # Create base dataloader
    base_dataloader, base_dataset = create_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        batch_size=1,  # Load one at a time for caching
        max_length=max_length,
        num_workers=0,  # No multiprocessing during caching
        split=split,
        **kwargs
    )
    
    # Create cache key if not provided
    if cache_key is None:
        cache_key = f"{dataset_name}_{max_length}_{split}"
    
    # Wrap in cached dataset
    cached_dataset = CachedDataset(
        base_dataset=base_dataset,
        cache_key=cache_key,
        max_cache_size=100000  # Cache up to 100k samples
    )
    
    # Create new dataloader with cached dataset
    from torch.utils.data import DataLoader
    
    cached_dataloader = DataLoader(
        cached_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # No workers needed - data is already in memory
        pin_memory=True,
        drop_last=True,
        persistent_workers=False
    )
    
    return cached_dataloader, cached_dataset