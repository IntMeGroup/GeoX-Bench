#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Re-prediction script for missing predictions in LMM benchmark results.
Based on main_closeai.py architecture but specifically targets missing instances.
"""

import os
import json
import random
import hashlib
from collections import defaultdict, Counter
from importlib import import_module
from pathlib import Path
from typing import Any, List, Dict, Optional, Set, Tuple
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

def load_cached_indices(cache_path: str) -> Dict[str, List[int]]:
    """Load all cached indices."""
    if not os.path.exists(cache_path):
        return {}
    
    try:
        with open(cache_path, 'r') as f:
            return json.load(f)
    except:
        return {}

def analyze_missing_predictions(output_file_path: str) -> Tuple[Set[str], Set[str]]:
    """Analyze a JSON file for missing and error predictions.
    
    Returns:
        Tuple of (missing_instance_ids, error_instance_ids)
    """
    missing_ids = set()
    error_ids = set()
    
    try:
        with open(output_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both array format and instances format
        if isinstance(data, list):
            instances = data
        elif 'instances' in data:
            instances = data['instances']
        else:
            return missing_ids, error_ids
        
        for instance in instances:
            instance_id = instance.get('image_id', 'unknown_id')
            prediction_obj = instance.get('prediction', {})
            
            # Handle different prediction structures
            if isinstance(prediction_obj, dict):
                prediction = prediction_obj.get('message', '')
            else:
                prediction = str(prediction_obj) if prediction_obj else ''
            
            # Check for empty prediction
            if prediction == '' or prediction is None:
                missing_ids.add(str(instance_id))
            # Check for error in prediction
            elif isinstance(prediction, str) and 'error' in prediction:
                error_ids.add(str(instance_id))
    
    except Exception as e:
        print(f"Error analyzing {output_file_path}: {e}")
    
    return missing_ids, error_ids

def find_dataset_for_missing_instances(missing_ids: Set[str], dataset, cached_indices: List[int]) -> List[int]:
    """Find dataset indices for missing instance IDs."""
    missing_dataset_indices = []
    
    for cache_idx, dataset_idx in enumerate(cached_indices):
        if dataset_idx < len(dataset):
            item = dataset[dataset_idx]
            instance = item.get("instance", {})
            
            # Extract instance ID using the same logic as the original system
            instance_id = str(instance.get('image_id', 'unknown_id'))
            
            # If no image_id, try to construct from other fields
            if instance_id == 'unknown_id' and "GroundImage" in instance:
                ground_img = instance["GroundImage"]
                if "filename" in ground_img:
                    filename = ground_img["filename"]
                    if "/" in filename:
                        instance_id = filename.split("/")[-1].replace(".jpg", "").replace(".png", "")
                    else:
                        instance_id = filename.replace(".jpg", "").replace(".png", "")
            
            if instance_id in missing_ids:
                missing_dataset_indices.append(dataset_idx)
                print(f"Found missing instance {instance_id} at dataset index {dataset_idx} (cache index {cache_idx})")
    
    return missing_dataset_indices

class MissingInstanceDataset:
    """Dataset wrapper that only includes missing instances."""
    
    def __init__(self, original_dataset, missing_indices):
        self.original_dataset = original_dataset
        self.missing_indices = missing_indices
    
    def __len__(self):
        return len(self.missing_indices)
    
    def __getitem__(self, idx):
        return self.original_dataset[self.missing_indices[idx]]

def get_missing_instance_dataloader(root_dir: str, json_path: str, batch_size: int, 
                                  task_name: str, message_formatter, dataset_class, 
                                  missing_ids: Set[str], cache_path: str):
    """Create dataloader for missing instances only."""
    
    # Load the original dataset
    original_dataset = dataset_class(
        root_dir=root_dir,
        json_path=json_path,
        message_formatter=message_formatter
    )
    
    # Get the cached indices for this dataset
    dataset_id = create_dataset_id(json_path, task_name)
    cached_indices = load_cached_indices(cache_path).get(dataset_id, [])
    
    if not cached_indices:
        print(f"No cached indices found for dataset {dataset_id}")
        return None
    
    # Find dataset indices for missing instances
    missing_dataset_indices = find_dataset_for_missing_instances(missing_ids, original_dataset, cached_indices)
    
    if not missing_dataset_indices:
        print(f"No missing instances found in dataset for {json_path}")
        return None
    
    print(f"Creating dataloader for {len(missing_dataset_indices)} missing instances")
    
    missing_dataset = MissingInstanceDataset(original_dataset, missing_dataset_indices)
    
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        return {
            "message": [item["message"] for item in batch],
            "instance": [item["instance"] for item in batch],
            "temp_dirs": [item.get("temp_dir") or item.get("temp_dirs") for item in batch]
        }
    
    return DataLoader(missing_dataset, batch_size=batch_size, shuffle=False, 
                     num_workers=10, collate_fn=collate_fn, pin_memory=True)

def run_missing_instance_prediction(model, inst_cfg, task_cfg, global_cfg, batch_size, 
                                  missing_ids: Set[str], error_ids: Set[str], output_subdir: str):
    """Run prediction for missing instances and save to new directory."""
    DataWriter = _resolve(task_cfg.data_writer_class)
    LMMMessage = _resolve(task_cfg.lmm_message_class)
    Dataset = _resolve(task_cfg.dataset_class)
    
    # Create output directory for re-predictions
    repred_folder = os.path.join(global_cfg.output_base, output_subdir, task_cfg.output_dir)
    os.makedirs(repred_folder, exist_ok=True)
    writer_output_path = os.path.join(repred_folder, model.config.name + '-' + inst_cfg.output)
    
    print(f"Re-predicting for {task_cfg.name}:{inst_cfg.output}")
    print(f"Missing instances: {len(missing_ids)}, Error instances: {len(error_ids)}")
    print(f"Output: {writer_output_path}")
    
    all_problematic_ids = missing_ids.union(error_ids)
    
    if not all_problematic_ids:
        print("No problematic instances found, skipping...")
        return
    
    # Get cache path
    cache_path = get_cache_path(global_cfg.output_base)
    
    # Create dataloader for missing instances
    dataloader = get_missing_instance_dataloader(
        root_dir=inst_cfg.root_dir,
        json_path=inst_cfg.json_path,
        batch_size=batch_size,
        task_name=task_cfg.name,
        message_formatter=LMMMessage(),
        dataset_class=Dataset,
        missing_ids=all_problematic_ids,
        cache_path=cache_path
    )
    
    if dataloader is None:
        print("No dataloader created, skipping...")
        return
    
    # Initialize writer
    writer = DataWriter(writer_output_path)
    
    # Run predictions
    for batch in tqdm(dataloader, desc=f"Re-predicting {task_cfg.name}:{inst_cfg.output}"):
        predictions = model(batch["message"])
        writer(predictions, batch["instance"], batch["temp_dirs"])
    
    writer.write()
    
    torch.cuda.empty_cache()
    gc.collect()
    
    print(f"Re-prediction completed: {writer_output_path}")

def analyze_model_predictions(model_name: str, output_base: str) -> Dict[str, Tuple[Set[str], Set[str]]]:
    """Analyze all prediction files for a model and return missing/error instances per file."""
    model_issues = {}
    
    # Scan all JSON files for the model
    base_dir = Path(output_base)
    for json_file in base_dir.rglob(f'{model_name}-*.json'):
        if json_file.name == 'dataset_indices_cache.json':
            continue
        
        missing_ids, error_ids = analyze_missing_predictions(str(json_file))
        if missing_ids or error_ids:
            model_issues[str(json_file)] = (missing_ids, error_ids)
    
    return model_issues

def extract_task_and_instance_from_path(file_path: str, cfg: DictConfig) -> Tuple[Optional[Any], Optional[Any]]:
    """Extract task config and instance config from file path."""
    file_path = Path(file_path)
    
    # Extract task directory from path
    task_dir = file_path.parent.name
    
    # Extract instance name from filename (remove model name prefix)
    filename = file_path.name
    model_name = filename.split('-')[0]
    instance_name_with_ext = filename.replace(f'{model_name}-', '')  # Keep .json
    instance_name_without_ext = instance_name_with_ext.replace('.json', '')  # Without .json
    
    # Find matching task and instance in config
    for task_name, task_cfg in cfg.tasks.items():
        if hasattr(task_cfg, 'output_dir') and task_cfg.output_dir == task_dir:
            for inst_cfg in task_cfg.instances:
                # Try both with and without .json extension
                if inst_cfg.output == instance_name_with_ext or inst_cfg.output == instance_name_without_ext:
                    return task_cfg, inst_cfg
    
    return None, None

def run_reprediction_for_model(model_cfg, cfg: DictConfig, output_subdir: str = "re_predictions"):
    """Run re-prediction for all missing instances of a specific model."""
    Model = _resolve(model_cfg.class_path)
    model = Model(model_cfg)
    model_name = model_cfg.name
    
    print(f"\n🔄 Processing model: {model_name}")
    
    # Analyze all prediction files for this model
    model_issues = analyze_model_predictions(model_name, cfg.output_base)
    
    if not model_issues:
        print(f"No missing predictions found for model {model_name}")
        return
    
    print(f"Found {len(model_issues)} files with issues for {model_name}")
    
    # Process each problematic file
    for file_path, (missing_ids, error_ids) in model_issues.items():
        print(f"\nProcessing: {file_path}")
        print(f"  Missing: {len(missing_ids)}, Errors: {len(error_ids)}")
        
        # Extract task and instance config from file path
        task_cfg, inst_cfg = extract_task_and_instance_from_path(file_path, cfg)
        
        if task_cfg is None or inst_cfg is None:
            print(f"  Could not find matching task/instance config for {file_path}")
            continue
        
        try:
            # Run re-prediction for this specific task/instance
            run_missing_instance_prediction(
                model, inst_cfg, task_cfg, cfg, model_cfg.batch_size,
                missing_ids, error_ids, output_subdir
            )
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

@hydra.main(config_path="conf", config_name="conf_closeai", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main function for re-prediction of missing instances."""
    
    Path(cfg.output_base).mkdir(parents=True, exist_ok=True)
    random.seed(42)
    
    # Get target model from config
    target_model = getattr(cfg, 'target_model', None)
    output_subdir = getattr(cfg, 'output_subdir', 're_predictions')
    
    print("🔄 Starting re-prediction for missing instances...")
    print(f"📁 Output subdirectory: {output_subdir}")
    
    if target_model:
        print(f"🎯 Targeting specific model: {target_model}")
        
        # Find the specific model config
        model_found = False
        for i in range(10):  # Check models.0 through models.9
            try:
                if hasattr(cfg, 'models') and str(i) in cfg.models:
                    model_cfg = cfg.models[str(i)]
                    if model_cfg and model_cfg.name == target_model:
                        run_reprediction_for_model(model_cfg, cfg, output_subdir)
                        model_found = True
                        break
            except Exception as e:
                print(f"Error checking model {i}: {e}")
                continue
        
        if not model_found:
            print(f"❌ Model {target_model} not found in configuration")
    else:
        print("🔄 Processing all available models...")
        
        # Process all models defined in the config
        for i in range(10):  # Check models.0 through models.9
            try:
                if hasattr(cfg, 'models') and str(i) in cfg.models:
                    model_cfg = cfg.models[str(i)]
                    if model_cfg:
                        try:
                            run_reprediction_for_model(model_cfg, cfg, output_subdir)
                        except Exception as e:
                            print(f"❌ Error processing model {model_cfg.name}: {e}")
            except:
                continue
    
    print("\n✅ Re-prediction process completed!")

if __name__ == "__main__":
    main()
