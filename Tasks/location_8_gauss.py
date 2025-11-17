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
        parse out the 'location' field.
        
        Returns:
            str: The location value (e.g., "center", "left", "right", "top", "bottom", 
                "top left", "top right", "bottom left", "bottom right"),
                or None if the field is missing.
        """
        # Look for: "location": "value"
        pattern = r'"location"\s*:\s*"(.*?)"'
        match = re.search(pattern, llm_response, re.IGNORECASE)

        if match:
            return match.group(1)
        return None

class LocationLMMMessage:
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
                        "- A photo taken within the map's coverage area.\n\n"
                        "Your task is to decide, from these two images, **where** the ground-level photo was taken relative to "
                        "the aerial image.\n\n"
                        "Return the answer in the following JSON-like format (plain text):\n"
                        "{\n"
                        '  "explanation": "Your reasoning...",\n'
                        '  "location": "<one of the strings below>"\n'
                        "}\n\n"
                        "Where **location** must be exactly one value from:\n"
                        '    [\"center\", \"left\", \"right\", \"top\", \"bottom\", '
                        '\"top left\", \"top right\", \"bottom left\", \"bottom right\"]\n\n'
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
        dataset = Subset(dataset, range(sample_size))

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
