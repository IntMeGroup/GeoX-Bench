from .task_template import DataWriter, GADataset, LMMMessage
from .utils import CenterDivImageCroper
import re
from copy import deepcopy
import random
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import shutil
from typing import Dict, List, Optional
import os

class InRnageDataWriter(DataWriter):
    def __init__(self, output_json):
        super().__init__(output_json)
    
    def extract_explanation_fullview(self, llm_response: str) -> str:
        """
        Given the LLM response text (which should contain structured JSON),
        parse out the 'is_in_range' field (true/false).
        
        Returns:
            bool: True if 'is_in_range' is set to true, False if set to false,
                or None if the field is missing.
        """
        # Look for: "is_in_range": true/false
        pattern = r'"is_in_range"\s*:\s*(true|false)'
        match = re.search(pattern, llm_response, re.IGNORECASE)
        if match:
            return match.group(1).lower() == 'true'
        return None

class LocationLMMMessage:
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
                            "- The ground-level photo might be taken somewhere within the boundaries of this map.\n"
                            "- The top of this aerial image always corresponds to North.\n\n"
                            "2. Ground-level camera image:\n"
                            "- A photo that may or may not be taken from within the coverage area of the aerial image.\n\n"
                            "Your objective is to determine, based on these two images, whether the ground-level photo "
                            "was taken within the coverage area (range) of the aerial image.\n\n"
                            "Please return your answer in the following structured format (as JSON-like text) so that it can be parsed:\n"
                            '{\n  "explanation": "Your reasoning...",\n  "is_in_range": true\n}\n'
                            "where 'is_in_range' is a boolean indicating whether the ground image is within the aerial map range.\n\n"
             },
            {
                "role": "user",
                "content": 
                            "<image><image>Given these images (the aerial map and the ground-level photo), please determine if the ground-level photo "
                            "was taken within the coverage area of the aerial image.\n\n"
                            "Remember:\n"
                            "- Provide explanatory details under a separate field, for example: \"explanation\".\n"
                            '- Output your answer in a structured format that includes the field "is_in_range" '
                            "(true/false)."
                     }
        ],
            'images': [aerial_image_path, ground_image_path]
            }
            

class Location9LMMMessage:
    def __init__(self):
        pass

    def __call__(self, aerial_image_path: str, ground_image_path: str) -> List[Dict]:
        return {
            'messages': [
                {
                    "role": "system",
                    "content":
                        "You are provided with two images:\n\n"
                        "1. Remote-sensing (aerial) image:\n"
                        "- Serves as a map whose top always points North.\n\n"
                        "2. Ground-level camera image:\n"
                        "- A photo that may or may not have been taken within the map’s coverage area.\n\n"
                        "Your task is to decide, from these two images, **where** the ground-level photo was taken relative to "
                        "the aerial image.\n\n"
                        "Return the answer in the following JSON-like format (plain text):\n"
                        "{\n"
                        '  "explanation": "Your reasoning...",\n'
                        '  "location": "<one of the strings below>"\n'
                        "}\n\n"
                        "Where **location** must be exactly one value from:\n"
                        '    [\"not exist\", \"center\", \"left\", \"right\", \"top\", \"bottom\", '
                        '\"top left\", \"top right\", \"bottom left\", \"bottom right\"]\n\n'
                        "Use **\"not exist\"** if the ground photo is outside the map’s coverage."
                },
                {
                    "role": "user",
                    "content":
                        "<image><image>Given these images (the aerial map and the ground-level photo), indicate the most likely "
                        "location of the ground-level photo relative to the aerial map.\n\n"
                        "Remember:\n"
                        "- Put your detailed reasoning in the \"explanation\" field.\n"
                        "- Output the JSON-like structure exactly as specified, with \"location\" set to one allowed string."
                }
            ],
            'images': [aerial_image_path, ground_image_path]
        }
                    
          

class InRangeDataset(GADataset):
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
