#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import hashlib
from collections import defaultdict, Counter
from importlib import import_module
from pathlib import Path
from typing import Any, List, Dict, Optional
import torch
import gc
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from multiprocessing import Process

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _resolve(dotted: str) -> Any:
    mod, attr = dotted.rsplit(".", 1)
    return getattr(import_module(mod), attr)

def create_dataset_id(json_path: str, task_name: str) -> str:
    """Create unique dataset ID based on dataset parameters."""
    content = f"{json_path}_{task_name}"
    return hashlib.md5(content.encode()).hexdigest()

def get_cache_path(output_base: str) -> str:
    """Get path for dataset indices cache file."""
    return os.path.join(output_base, "dataset_indices_cache.json")

def load_cached_indices(cache_path: str, dataset_id: str) -> Optional[List[int]]:
    """Load cached indices if available."""
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r') as f:
            cache = json.load(f)
        return cache.get(dataset_id)
    except:
        return None

def save_cached_indices(cache_path: str, dataset_id: str, indices: List[int]):
    """Save indices to cache."""
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
        except:
            cache = {}
    
    cache[dataset_id] = indices
    
    with open(cache_path, 'w') as f:
        json.dump(cache, f, indent=2)

def load_previous_cache_indices(previous_cache_path: str, dataset_id: str) -> Optional[List[int]]:
    """Load indices from previous cache to exclude them from new sampling."""
    if not os.path.exists(previous_cache_path):
        return None
    
    try:
        with open(previous_cache_path, 'r') as f:
            cache = json.load(f)
        return cache.get(dataset_id)
    except:
        return None

def get_dataset_sample_ids(dataset) -> List[str]:
    """Extract unique sample IDs from dataset for tracking."""
    sample_ids = []
    for item in dataset:
        if "instance" in item:
            # Create unique ID from key fields
            instance = item["instance"]
            if "GroundImage" in instance:
                ground_img = instance["GroundImage"]
                # Use file paths and coordinates as unique identifier
                sample_id = f"{ground_img.get('filename', '')}_" \
                           f"{ground_img.get('lat', '')}_" \
                           f"{ground_img.get('lon', '')}_" \
                           f"{ground_img.get('RelativeLocation', 'center')}"
                sample_ids.append(sample_id)
            else:
                sample_ids.append(f"sample_{len(sample_ids)}")
        else:
            sample_ids.append(f"sample_{len(sample_ids)}")
    
    return sample_ids


def create_balanced_indices_from_dataset(dataset, task_type: str, target_samples: int, 
                                        cache_path: str = None, dataset_id: str = None, 
                                        force_regenerate: bool = False,
                                        previous_cache_path: str = None) -> List[int]:
    """Create perfectly balanced dataset indices from processed dataset with caching.
    
    Args:
        dataset: The processed dataset to sample from
        task_type: Type of task (location, orientation, map_gauss, location_8_gauss)
        target_samples: Number of samples to select
        cache_path: Path to cache file (optional)
        dataset_id: Unique dataset identifier (optional)
        force_regenerate: If True, bypass cache and generate new indices (default: False)
        previous_cache_path: Path to previous cache to exclude already used indices (optional)
    """
    
    # Try to load from cache first (unless force_regenerate is True)
    if cache_path and dataset_id and not force_regenerate:
        cached_indices = load_cached_indices(cache_path, dataset_id)
        if cached_indices is not None:
            # Validate cached indices against current dataset
            max_index = max(cached_indices) if cached_indices else -1
            if max_index >= len(dataset):
                print(f"Cached indices invalid (max index {max_index} >= dataset size {len(dataset)}), regenerating...")
            else:
                print(f"Using cached indices for dataset {dataset_id[:8]}... ({len(cached_indices)} samples)")
                return cached_indices
    elif force_regenerate and cache_path and dataset_id:
        print(f"Force regenerating indices for dataset {dataset_id[:8]}... (generating new samples)")
    
    # Load previous cache to exclude already used indices
    excluded_indices = set()
    if previous_cache_path and dataset_id:
        previous_indices = load_previous_cache_indices(previous_cache_path, dataset_id)
        if previous_indices:
            excluded_indices = set(previous_indices)
            print(f"Excluding {len(excluded_indices)} previously used indices from dataset {dataset_id[:8]}...")
    
    print(f"Generating new balanced indices for dataset {dataset_id[:8] if dataset_id else 'unknown'}...")
    
    # Get categories and create mapping for valid indices (excluding previous ones)
    all_categories = []
    valid_indices = []  # Track which original indices have valid categories
    
    for idx, item in enumerate(dataset):
        category = None
        if task_type == "location":
            if "instance" in item and "GroundImage" in item["instance"] and "in_range" in item["instance"]["GroundImage"]:
                category = item["instance"]["GroundImage"]["in_range"]
        elif task_type == "orientation":
            if "instance" in item and "GroundImage" in item["instance"] and "orientation" in item["instance"]["GroundImage"]:
                category = item["instance"]["GroundImage"]["orientation"]
        elif task_type == "map_gauss":
            # Extract actual ground truth aerial index for map_gauss
            category = None
            if "instance" in item:
                instance = item["instance"]
                
                # Try multiple locations for aerial index
                if "correct_aerial_index" in instance:
                    category = instance["correct_aerial_index"]
                        
                if category is None and "GroundTruth" in instance and isinstance(instance["GroundTruth"], dict):
                    gt = instance["GroundTruth"]
                    if "correct_aerial_index" in gt:
                        category = gt["correct_aerial_index"]
                    elif "aerial_index" in gt:
                        category = gt["aerial_index"]
                            
                if category is None and "GroundImage" in instance and isinstance(instance["GroundImage"], dict):
                    if "correct_aerial_index" in instance["GroundImage"]:
                        category = instance["GroundImage"]["correct_aerial_index"]
                            
                if category is None and "AerialCandidates" in instance and isinstance(instance["AerialCandidates"], list):
                    # Find which candidate is marked as original (correct)
                    for i, candidate in enumerate(instance["AerialCandidates"]):
                        if isinstance(candidate, dict):
                            # Check for is_original (which means correct for map_gauss)
                            if candidate.get('is_original', False):
                                category = i
                                break
                            # Also check nested AerialImage for is_original
                            if 'AerialImage' in candidate and isinstance(candidate['AerialImage'], dict):
                                if candidate['AerialImage'].get('is_original', False):
                                    category = i
                                    break
                
                # Validate aerial index is in range 0-3
                if category is not None and (not isinstance(category, int) or category < 0 or category > 3):
                    category = None
        elif task_type == "location_8_gauss":
            # Extract relative location for location_8_gauss task
            if "instance" in item and "GroundImage" in item["instance"] and "RelativeLocation" in item["instance"]["GroundImage"]:
                category = item["instance"]["GroundImage"]["RelativeLocation"]
            
        # Only include indices that are not in excluded set
        if category is not None and idx not in excluded_indices:
            all_categories.append(category)
            valid_indices.append(idx)
    
    categories = all_categories
    
    if not categories:
        raise ValueError(f"No categories found in processed dataset")
    
    # Group valid indices by category
    category_indices = defaultdict(list)
    for valid_idx, category in enumerate(categories):
        # valid_idx is the index in the categories list
        # valid_indices[valid_idx] is the actual dataset index
        original_idx = valid_indices[valid_idx]
        category_indices[category].append(original_idx)
    
    # Shuffle within categories
    for indices in category_indices.values():
        random.shuffle(indices)
    
    unique_categories = list(category_indices.keys())
    n_categories = len(unique_categories)
    
    # Calculate samples per category
    samples_per_category = target_samples // n_categories
    remainder = target_samples % n_categories
    
    # Select balanced samples
    selected_indices = []
    for i, category in enumerate(unique_categories):
        target = samples_per_category + (1 if i < remainder else 0)
        available = len(category_indices[category])
        samples_to_take = min(target, available)
        selected_indices.extend(category_indices[category][:samples_to_take])
    
    random.shuffle(selected_indices)
    
    # Save to cache
    if cache_path and dataset_id:
        save_cached_indices(cache_path, dataset_id, selected_indices)
    
    return selected_indices

class BalancedDataset:
    """Simple wrapper for balanced sampling."""
    
    def __init__(self, original_dataset, selected_indices):
        self.original_dataset = original_dataset
        self.selected_indices = selected_indices
    
    def __len__(self):
        return len(self.selected_indices)
    
    def __getitem__(self, idx):
        return self.original_dataset[self.selected_indices[idx]]

def get_balanced_dataloader(root_dir: str, json_path: str, batch_size: int, 
                          target_samples: int, task_name: str, 
                          message_formatter, dataset_class, cache_path: str = None,
                          force_regenerate: bool = False, previous_cache_path: str = None):
    """Create balanced dataloader from processed dataset with caching."""
    dataset_id = create_dataset_id(json_path, task_name)
    
    # First load the original dataset with all processing/treatment
    original_dataset = dataset_class(
        root_dir=root_dir,
        json_path=json_path,
        message_formatter=message_formatter
    )
    
    # Detect task type from dataset class and task_name for balanced sampling
    dataset_class_name = dataset_class.__module__ + "." + dataset_class.__name__
    task_name_lower = task_name.lower()
    
    if "location_8_gauss" in dataset_class_name:
        task_type = "location_8_gauss"
    elif "orin" in task_name_lower or "orientation" in task_name_lower:
        task_type = "orientation"
    elif "map" in task_name_lower and "gauss" in task_name_lower:
        task_type = "map_gauss"
    else:
        task_type = "location"
    
    # Now create balanced indices based on the processed dataset with caching
    selected_indices = create_balanced_indices_from_dataset(
        original_dataset, task_type, target_samples, cache_path, dataset_id, force_regenerate, previous_cache_path
    )
    
    balanced_dataset = BalancedDataset(original_dataset, selected_indices)
    
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        return {
            "message": [item["message"] for item in batch],
            "instance": [item["instance"] for item in batch],
            "temp_dirs": [item.get("temp_dir") or item.get("temp_dirs") for item in batch]
        }
    
    return DataLoader(balanced_dataset, batch_size=batch_size, shuffle=False, 
                     num_workers=0, collate_fn=collate_fn, pin_memory=False)

def run_instance(model, inst_cfg, task_cfg, global_cfg, batch_size, target_samples, force_regenerate=False, previous_cache_path=None):
    """Run single instance with balanced sampling and caching."""
    DataWriter = _resolve(task_cfg.data_writer_class)
    LMMMessage = _resolve(task_cfg.lmm_message_class)
    Dataset = _resolve(task_cfg.dataset_class)
    
    task_folder = os.path.join(global_cfg.output_base, task_cfg.output_dir)
    os.makedirs(task_folder, exist_ok=True)
    writer_output_path = os.path.join(task_folder, model.config.name + '-' + inst_cfg.output)
    writer = DataWriter(writer_output_path)
    
    print(f"Task: {task_cfg.name}")
    
    # Get cache path
    cache_path = get_cache_path(global_cfg.output_base)
    
    # Create balanced dataloader with caching
    dataloader = get_balanced_dataloader(
        root_dir=inst_cfg.root_dir,
        json_path=inst_cfg.json_path,
        batch_size=batch_size,
        target_samples=target_samples,
        task_name=task_cfg.name,
        message_formatter=LMMMessage(),
        dataset_class=Dataset,
        cache_path=cache_path,
        force_regenerate=force_regenerate,
        previous_cache_path=previous_cache_path
    )

    for batch in tqdm(dataloader, desc=f"{task_cfg.name}:{inst_cfg.output}"):
        writer(model(batch["message"]), batch["instance"], batch["temp_dirs"])
    writer.write()
    
    gc.collect()

def run_pipeline(pipe, cfg):
    """Run pipeline with balanced sampling."""
    Model = _resolve(pipe.model.class_path)
    model = Model(pipe.model)
    target_samples = getattr(cfg, 'global_samples_per_dataset', 50)
    force_regenerate = getattr(cfg, 'force_regenerate', False)  # Default to False
    previous_cache_path = getattr(cfg, 'previous_cache_path', None)  # Path to previous cache
    
    for task_cfg in pipe.tasks:
        for inst_cfg in task_cfg.instances:
            run_instance(model, inst_cfg, task_cfg, cfg, pipe.model.batch_size, target_samples, force_regenerate, previous_cache_path)

def run_pipeline(pipe, cfg):
    """Run pipeline with balanced sampling."""
    from multiprocessing import Process, Semaphore
    
    Model = _resolve(pipe.model.class_path)
    model = Model(pipe.model)
    target_samples = getattr(cfg, 'global_samples_per_dataset', 50)
    force_regenerate = getattr(cfg, 'force_regenerate', False)  # Default to False
    previous_cache_path = getattr(cfg, 'previous_cache_path', None)  # Path to previous cache
    
    # Create semaphore to limit concurrent processes to 25
    max_concurrent = 25
    semaphore = Semaphore(max_concurrent)
    
    def run_instance_with_semaphore(*args):
        semaphore.acquire()
        try:
            run_instance(*args)
        finally:
            semaphore.release()
    
    for task_cfg in pipe.tasks:
        # Collect all instance processes for this task
        instance_processes = []
        
        for inst_cfg in task_cfg.instances:
            # Create a process for each instance
            p = Process(target=run_instance_with_semaphore, 
                       args=(model, inst_cfg, task_cfg, cfg, pipe.model.batch_size, 
                             target_samples, force_regenerate, previous_cache_path))
            p.start()
            instance_processes.append(p)
        
        # Wait for all instances in this task to complete
        for p in instance_processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Instance processing failed in task {task_cfg.name}")
            

@hydra.main(config_path="conf", config_name="conf_random8_AAAI", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function for balanced dataset evaluation.
    
    To force regenerate indices (bypass cache), use:
    python main_closeai2.py force_regenerate=true
    """
    Path(cfg.output_base).mkdir(parents=True, exist_ok=True)
    random.seed(42)
    
    # Check if force_regenerate is enabled
    force_regenerate = getattr(cfg, 'force_regenerate', False)
    if force_regenerate:
        print("🔄 Force regenerate enabled - all cached indices will be bypassed")
    
    for pipe in cfg.pipelines:
        p = Process(target=run_pipeline, args=(pipe, cfg))
        p.start()
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Pipeline {pipe.model.name} failed")

if __name__ == "__main__":
    main()
