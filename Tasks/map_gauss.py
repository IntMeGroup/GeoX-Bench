from .task_template import DataWriter, GADataset, LMMMessage
from .utils import CenterDivImageCroper
import re
from copy import deepcopy
import random
import cv2
from torch.utils.data import DataLoader, Subset, Dataset
from PIL import Image
import shutil
from typing import Dict, List, Optional
import os
from tempfile import mkdtemp
from shutil import copyfile
from typing import List, Dict

import pdb

class MapGaussDataWriter(DataWriter):
    """Utility class that extracts the aerial‑image index selected by the LMM.

    The model is now expected to output a JSON‑like snippet such as:

        {
            "explanation": "your reasoning…",
            "matched_aerial_index": 2
        }

    where **matched_aerial_index** is an *integer* referring to the position of
    the correct aerial image in the list that was supplied to the model.  The
    numbering is **zero‑based** – i.e. 0, 1, 2, 3 for four aerial images.
    """

    def __init__(self, output_json: str):
        super().__init__(output_json)

    # ------------------------------------------------------------------
    #  Field extraction helpers
    # ------------------------------------------------------------------
    def extract_explanation_fullview(self, llm_response: str) -> Optional[int]:
        """Parse *matched_aerial_index* from the LMM response.

        Args:
            llm_response: The raw text returned by the model.

        Returns:
            The integer index (0‑based) of the aerial image that the model
            thinks corresponds to the ground photograph, or **None** if the
            field cannot be found / parsed.
        """
        pattern = r"\"matched_aerial_index\"\s*:\s*(\d+)"
        match = re.search(pattern, llm_response)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    # Convenience alias to keep previous name intact (optional)

# -----------------------------------------------------------------------------
#  Prompt / Message builder
# -----------------------------------------------------------------------------

class MapGaussLMMMessage:
    """Constructs the multi‑modal prompt for the large multi‑modal model.

    Usage::

        builder = MapGaussLMMMessage()
        prompt_dict = builder(aerial_paths, ground_path)
    """

    def __call__(self, aerial_image_paths: List[str], ground_image_path: str) -> Dict:
        if len(aerial_image_paths) != 4:
            raise ValueError("Exactly four aerial image paths must be provided.")

        # Order of images: the four aerials (index 0‑3) followed by the ground‑level
        # photo (index 4).  The model must reference the aerials using their
        # index in this order.
        all_images = aerial_image_paths + [ground_image_path]

        system_msg = (
            "You are provided with **five** images in total:\n\n"
            "1–4: *Aerial (remote‑sensing) images* that serve as candidate maps.\n"
            "   • The *top of every aerial image* corresponds to **North**.\n"
            "   • Exactly **one** of these four aerial images covers the location\n"
            "     where the ground‑level photograph was taken.\n\n"
            "5: *Ground‑level photograph*.\n\n"
            "Your task is to inspect **all five** images and decide which aerial\n"
            "image (index 0, 1, 2, or 3) corresponds to the same physical\n"
            "location shown in the ground‑level photo.  Only *one* aerial image\n"
            "is correct.\n\n"
            "Return your answer in *valid* JSON‑like text using the following\n"
            "schema – do **not** include any additional keys::\n\n"
            "  {\n"
            "    \"explanation\": \"<brief reasoning – max 3 sentences>\",\n"
            "    \"matched_aerial_index\": <integer 0‑3>\n"
            "  }\n\n"
            "Guidelines:\n"
            "• Be concise but cite at least one visible feature (e.g. road shape,\n"
            "  building layout, natural landmark) that supports your choice.\n"
            "• Use **zero‑based** indexing: the *first* aerial image is 0, the\n"
            "  *second* is 1, etc.\n"
            "• Do **not** wrap the JSON in markdown formatting fences.\n"
        )

        user_msg = (
            "<image><image><image><image><image>\n\n"
            "Based on the four aerial images and the ground‑level photograph,\n"
            "select the aerial image that covers the same location as the ground\n"
            "photo.  Provide your answer in the required structured format."
        )

        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "images": all_images,
        }

          


class MapGaussDataset(GADataset):
    """
    Simple (non-augmentation) dataset loader that
    – copies every aerial-candidate image and the ground image into a
      persistent tmp dir, and
    – appends '_aerial_tmp' / '_ground_tmp' to each filename to prevent
      clashes.
    """

    # -------------------------- helpers -------------------------------- #
    @staticmethod
    def _add_suffix_to_filename(path: str, *suffixes) -> str:
        """
        Insert one or more suffixes before the file extension.

        Example
        -------
        >>> _add_suffix_to_filename('/abs/path/img.png', 'aerial', 'tmp')
        'img_aerial_tmp.png'
        """
        basename = os.path.basename(path)
        name, ext = os.path.splitext(basename)
        return f"{name}_{'_'.join(suffixes)}{ext}"

    # -------------------------- ctor ----------------------------------- #
    def __init__(self,
                 root_dir,
                 json_path,
                 batch_size=1,
                 sample_size=1,
                 message_formatter=None,
                 panorama=False,
                 north_degree=0):
        super().__init__(root_dir,
                         json_path,
                         batch_size,
                         sample_size,
                         message_formatter)

    # -------------------------- main loader ---------------------------- #
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # ---- locate originals ---------------------------------------- #
        orig_aerials = [
            os.path.join(self.root_dir,
                         cand['AerialImage']['file_path'])
            for cand in sample['AerialCandidates']
        ]
        orig_ground = os.path.join(self.root_dir,
                                   sample['GroundImage']['file_path'])

        # ---- create tmp dir ------------------------------------------ #
        temp_dir = mkdtemp()
        tmp_aerials = []

        # ---- copy aerial candidates w/ suffix ------------------------ #
        for orig in orig_aerials:
            tmp_name = self._add_suffix_to_filename(orig, 'aerial', 'tmp')
            dest = os.path.join(temp_dir, tmp_name)
            orig = orig.strip()
            copyfile(orig, dest)
            tmp_aerials.append(dest)

        # ---- copy ground image w/ suffix ----------------------------- #
        tmp_ground_name = self._add_suffix_to_filename(orig_ground,
                                                       'ground', 'tmp')
        tmp_ground = os.path.join(temp_dir, tmp_ground_name)
        copyfile(orig_ground, tmp_ground)

        # ---- build LLM message --------------------------------------- #
        message = self.message_formatter(tmp_aerials, tmp_ground)

        return {
            'message': message,
            'instance': sample,
            'temp_dir': temp_dir
        }




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