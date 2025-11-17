import os
import re
import json
import shutil
import tempfile
import random
import argparse
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor


import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm

# ------------------------------------------------------------------------------
# LLM Message Formatter
# ------------------------------------------------------------------------------
class LMMMessage:
    def __init__(self):
        pass
    
    def __call__(self, aerial_image_path: str, ground_image_path: str) -> List[Dict]:
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are provided with two images:\n\n"
                            "1. Remote sensing (aerial) image:\n"
                            "- This image serves as a map.\n"
                            "- The user’s ground-level photo is taken from the center of this map.\n"
                            "- The top of this aerial image always corresponds to North.\n\n"
                            "2. Ground-level camera image:\n"
                            "- A photo taken by the user positioned at the center of the aerial image.\n\n"
                            "Your objective is to determine, based on these two images, the cardinal direction "
                            "(N, NE, E, SE, S, SW, W, NW) in which the user was facing when they captured the ground-level photo.\n\n"
                            "Your answer must be returned in a structured format (e.g., JSON) so that it can be parsed "
                            "to obtain a single directional value from: N, NE, E, SE, S, SW, W, NW.\n\n"
                            "You may include an explanation, but ensure that your final output contains a clear, easily-extracted field, for example:\n"
                            '{\n  "explanation": "Additional details or reasoning here...",\n  "direction": "N"\n}'
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Given these images (the aerial map and the ground-level photo), please provide your best assessment "
                            "of the direction the user faced when taking the ground-level photo.\n\n"
                            "Remember:\n"
                            "- Provide explanatory details under a separate field, for example: \"explanation\".\n"
                            "- Output your answer in a structured format that includes the field \"direction\" "
                            "(one of N, NE, E, SE, S, SW, W, NW).\n"
                        ),
                    },
                    {"type": "image", "image": aerial_image_path},
                    {"type": "image", "image": ground_image_path},
                ],
            },
        ]


# ------------------------------------------------------------------------------
# Helper function to create persistent temporary copies of images.
# ------------------------------------------------------------------------------
def create_temp_image_paths_persistent(original_aerial: str, original_ground: str):
    """
    Copies the provided images into a persistent temporary folder (created with mkdtemp)
    and returns the temporary file paths along with the temporary directory path.
    """
    temp_dir = tempfile.mkdtemp()  # creates a persistent temporary directory
    tmp_aerial = os.path.join(temp_dir, os.path.basename(original_aerial))
    tmp_ground = os.path.join(temp_dir, os.path.basename(original_ground))
    
    shutil.copy2(original_aerial, tmp_aerial)
    shutil.copy2(original_ground, tmp_ground)
    
    return tmp_aerial, tmp_ground, temp_dir


# ------------------------------------------------------------------------------
# Custom Dataset
# ------------------------------------------------------------------------------
class OrientationDataset(Dataset):
    def __init__(self, root_dir: str, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        self.samples = data
        self.message_formatter = LMMMessage()
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
        tmp_aerial, tmp_ground, temp_dir = create_temp_image_paths_persistent(orig_aerial, orig_ground)
        
        # Create the LLM message using the temporary paths
        message = self.message_formatter(tmp_aerial, tmp_ground)
        
        return {
            "message": message,
            "instance": self.samples[idx],
            "temp_dir": temp_dir  # include the temp directory path for later cleanup
        }


# ------------------------------------------------------------------------------
# DataLoader utility function
# ------------------------------------------------------------------------------
def get_dataloader(root_dir: str,
                   json_path: str,
                   batch_size: int = 4,
                   sample_size: int = 0):

    dataset = OrientationDataset(root_dir=root_dir, json_path=json_path)
    
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


# ------------------------------------------------------------------------------
# DataWriter for handling LLM outputs.
# ------------------------------------------------------------------------------

def cleanup_temp_dirs(temp_dirs: List[str]):
    """
    Function that removes a list of temporary directories.
    """
    for temp_dir in temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"Error cleaning up {temp_dir}: {e}")


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
                    "orientation": self.extract_explanation_fullview(msg)
                }
            }
            row_dict.update(entry)
            self.output_data.append(row_dict)
        self.executor.submit(cleanup_temp_dirs, temp_dirs_to_cleanup)

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
