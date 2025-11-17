
from .task_template import DataWriter, GADataset, LMMMessage
from .utils import CenterDivImageCroper
import re
from copy import deepcopy
import random
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import shutil
from typing import Dict, List
import os

class OrintationDataWriter(DataWriter):
    def __init__(self, output_json):
        super().__init__(output_json)
    
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

class OrintationLMMMessage:
    def __init__(self):
        pass
    
    def __call__(self, aerial_image_path: str, ground_image_path: str) -> List[Dict]:
        return {'messages':[
            {
                "role": "system",
                "content": 
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
            },
            {
                "role": "user",
                "content": 
                            "<image><image>Given these images (the aerial map and the ground-level photo), please provide your best assessment "
                            "of the direction the user faced when taking the ground-level photo.\n\n"
                            "Remember:\n"
                            "- Provide explanatory details under a separate field, for example: \"explanation\".\n"
                            "- Output your answer in a structured format that includes the field \"direction\" "
                            "(one of N, NE, E, SE, S, SW, W, NW).\n"
            }
        ],
        'images': [aerial_image_path, ground_image_path]
        }
        
class Orintation4LMMMessage:
    def __init__(self):
        pass
    
    def __call__(self, aerial_image_path: str, ground_image_path: str) -> List[Dict]:
        return {'messages':[
            {
                "role": "system",
                "content": 
                            "You are provided with two images:\n\n"
                            "1. Remote sensing (aerial) image:\n"
                            "- This image serves as a map.\n"
                            "- The user’s ground-level photo is taken from the center of this map.\n"
                            "- The top of this aerial image always corresponds to North.\n\n"
                            "2. Ground-level camera image:\n"
                            "- A photo taken by the user positioned at the random location inside of the aerial image.\n\n"
                            "Your objective is to determine, based on these two images, the cardinal direction "
                            "(N, E, S, W) in which the user was facing when they captured the ground-level photo.\n\n"
                            "Your answer must be returned in a structured format (e.g., JSON) so that it can be parsed "
                            "to obtain a single directional value from: N, E, S, W.\n\n"
                            "You may include an explanation, but ensure that your final output contains a clear, easily-extracted field, for example:\n"
                            '{\n  "explanation": "Additional details or reasoning here...",\n  "direction": "N"\n}'
            },
            {
                "role": "user",
                "content": 
                            "<image><image>Given these images (the aerial map and the ground-level photo), please provide your best assessment "
                            "of the direction the user faced when taking the ground-level photo.\n\n"
                            "Remember:\n"
                            "- Provide explanatory details under a separate field, for example: \"explanation\".\n"
                            "- Output your answer in a structured format that includes the field \"direction\" "
                            "(one of N, E, S, W).\n"
            }
        ],
        'images': [aerial_image_path, ground_image_path]
        }
        


class OrientationRandomDataset(GADataset):
    def __init__(self, 
                 root_dir, 
                 json_path, 
                 batch_size=1, 
                 sample_size=1, 
                 message_formatter=None,
                 panorama=False,
                 north_degree=0):
        super().__init__(root_dir, json_path, batch_size, sample_size, message_formatter)
        div_list = ["center", "left", "right", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"]
        randomed_sample = []
        self.center_image_div = CenterDivImageCroper()
        for sample in self.samples:
            for div in div_list:
                new_sample = deepcopy(sample)
                new_sample["GroundImage"]["RelativeLocation"] = div
                randomed_sample.append(new_sample)
        self.samples = randomed_sample
     
    def image_opeartion(self, 
                        original_aerial, 
                        tmp_aerial, 
                        original_ground, 
                        tmp_ground,
                        sample_instance): 
        shutil.copy2(original_ground, tmp_ground) 

        aerial_image = self.center_image_div(original_aerial, sample_instance["GroundImage"]['RelativeLocation'])
        aerial_image.save(tmp_aerial)
        return None
            



def get_dataloader(root_dir: str,
                   json_path: str,
                   batch_size: int = 4,
                   sample_size: int = 0,
                   panorama: bool = False,
                   north_degree: int = 0,
                   message_formatter:LMMMessage = None,
                   dataset:Dataset = None) -> DataLoader:

    dataset = dataset(root_dir=root_dir, 
                      json_path=json_path,
                      panorama=panorama,
                      north_degree=north_degree,
                      message_formatter=message_formatter)
    
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
