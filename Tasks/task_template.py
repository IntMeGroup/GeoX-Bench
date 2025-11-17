import torch
from torch.utils.data import Dataset
import json
import re
from typing import Dict, List
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import tempfile
import random
from torch.utils.data import DataLoader, Subset

class LMMMessage:
    """Generic prompt builder that returns Swift‑compatible payloads."""

    def __call__(self, aerial_image_path: str, ground_image_path: str) -> Dict:
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "<image><image>You are provided with two images:\n\n"
                        "1. Remote sensing (aerial) image (top is North).\n"
                        "2. Ground‑level image taken from the center of that map.\n\n"
                        "Respond to the user’s request below."
                    ),
                },
                {
                    "role": "user",
                    "content": "Hi",
                },
            ],
            "images": [aerial_image_path, ground_image_path],
        }
        

class DataWriter:
    def __init__(self, output_json: str):
        self.output_json = output_json
        self.output_data = []
        self.executor = ThreadPoolExecutor(max_workers=4)


    def __call__(self, message_list, 
                 instancec: Dict,
                 temp_dirs_to_cleanup: List):
        for i, msg in enumerate(message_list):
            # Combine the sample with prediction details.
            row_dict = instancec[i]
            entry = {
                'prediction': {
                    "message": msg,
                    "prediction_element": self.extract_explanation_fullview(msg)
                }
            }
            row_dict.update(entry)
            self.output_data.append(row_dict)
        self.executor.submit(self.cleanup_temp_dirs, temp_dirs_to_cleanup)

    def extract_explanation_fullview(self, llm_response: str) -> str:
        """
        Given the LLM response text (which should contain structured JSON),
        parse out the 'direction' field (one of N, NE, E, SE, S, SW, W, NW).
        """
        direction_pattern = r'"direction"\s*:\s*"([NSEW]{1,2})"'
        match = re.search(direction_pattern, llm_response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def write(self):
        with open(self.output_json, 'w') as f:
            json.dump(self.output_data, f, indent=4)
        self.executor.shutdown(wait=True)


    def cleanup_temp_dirs(self, temp_dirs: List[str]):
        """
        Function that removes a list of temporary directories.
        """
        for temp_dir in temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Error cleaning up {temp_dir}: {e}")




class GADataset(Dataset):
    def __init__(self, root_dir: str, 
                 json_path: str,
                 panorama: bool = False,
                 north_degree: int = 0,
                 message_formatter: LMMMessage = None):

        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = data
        self.message_formatter = message_formatter
        self.root_dir = root_dir

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        For each sample, this method creates a persistent temporary directory (using mkdtemp),
        copies the source images into it, and formats the LLM message using the temporary file paths.
        The path to the temporary directory is returned along with the message so that it can later
        be cleaned up after the model has processed the images.
        """
        # Locate the original images
        orig_aerial = os.path.join(self.root_dir, self.samples[idx]['AerialImage']['file_path'])
        orig_ground = os.path.join(self.root_dir, self.samples[idx]['GroundImage']['file_path'])
        
        # Create a persistent temporary directory and copy images into it
        tmp_aerial, tmp_ground, temp_dir = self.create_temp_image_paths_persistent(orig_aerial, 
                                                                                   orig_ground,
                                                                                   sample_instance=self.samples[idx])
        
        # Create the LLM message using the temporary paths
        message = self.message_formatter(tmp_aerial, tmp_ground)
        
        return {
            "message": message,
            "instance": self.samples[idx],
            "temp_dir": temp_dir  # include the temp directory path for later cleanup
        }

    def create_temp_image_paths_persistent(self, 
                                           original_aerial: str, 
                                           original_ground: str,
                                           sample_instance) -> tuple:
        """
        Copies the provided images into a persistent temporary folder (created with mkdtemp)
        and returns the temporary file paths along with the temporary directory path.
        """
        temp_dir = tempfile.mkdtemp()  # creates a persistent temporary directory
        def add_suffix_to_filename(filename, *suffixes):
            """
            Add suffixes to a filename before its extension.
            
            Args:
                filename: Original filename
                *suffixes: One or more suffixes to add before the extension
            
            Returns:
                Modified filename with suffixes added before extension
            """
            basename = os.path.basename(filename)
            name, ext = os.path.splitext(basename)
            return f"{name}_{'_'.join(suffixes)}{ext}"

        # Use the function for both aerial and ground images
        tmp_aerial = os.path.join(temp_dir, add_suffix_to_filename(original_aerial, "aerial", "tmp"))
        tmp_ground = os.path.join(temp_dir, add_suffix_to_filename(original_ground, "ground", "tmp"))
        self.image_opeartion(original_aerial, tmp_aerial, original_ground, tmp_ground, sample_instance)
        return tmp_aerial, tmp_ground, temp_dir
    
    def image_opeartion(self, 
                        original_aerial, 
                        tmp_aerial, 
                        original_ground, 
                        tmp_ground,
                        sample_instance): 

        shutil.copy2(original_aerial, tmp_aerial)
        shutil.copy2(original_ground, tmp_ground)
        return None


# ------------------------------------------------------------------------------
# DataLoader utility function
# ------------------------------------------------------------------------------
def get_dataloader(root_dir: str,
                   json_path: str,
                   batch_size: int = 4,
                   sample_size: int = 0,
                   panorama: bool = False,
                   north_degree: int = 0):

    dataset = GADataset(root_dir=root_dir, 
                        json_path=json_path,
                        panorama=panorama,
                        north_degree=north_degree)
    
    if sample_size > 0:
        random_indices = random.sample(range(len(dataset)), sample_size)
        dataset = Subset(dataset, random_indices)

    def collate_fn(batch):
        messages = [item["message"] for item in batch]
        instances = [item["instance"] for item in batch]
        temp_dirs = [item["temp_dir"] for item in batch]
        return {
            "message": messages,
            "instance": instances,
            "temp_dirs": temp_dirs  # pass on the temporary directories for later cleanup
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

