import os
from swift.llm import PtEngine, RequestConfig, InferRequest
from typing import List, Dict
from .model import InferenceModel, InferenceModelConfig

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import gc

from PIL import Image
import requests
import copy
import torch

import sys
import warnings


class LLaVAOneVision(InferenceModel):
    """
    Thin wrapper around ms‑swift’s PtEngine so that the external API
    (constructor + __call__) remains identical to the original
    Hugging Face implementation.
    """
    image_token = DEFAULT_IMAGE_TOKEN

    def __init__(self, config: InferenceModelConfig):
        self.config = config
        self.test_mode = config.test_mode
        self.model_path = config.model_path
        warnings.filterwarnings("ignore")
        model_name = "llava_qwen"
        device_map = "auto"
        llava_model_args = {
                "multimodal": True,
            }
        overwrite_config = {}
        overwrite_config["image_aspect_ratio"] = "pad"
        llava_model_args["overwrite_config"] = overwrite_config
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(self.model_path, 
                                                                              None, 
                                                                              model_name, 
                                                                              device_map=device_map, 
                                                                              **llava_model_args)

        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    def __call__(self, messages: List[Dict]) -> List[str]:
        """
        Parameters
        ----------
        messages : List[Dict]
            Same schema you already use:
            [{'role':'system','content':'...'}, {'role':'user','content':'<image> …'}]

        Returns
        -------
        List[str]
            A single‑element list containing the model reply, matching the
            original behaviour so you don’t have to touch downstream code.
        """
        if self.test_mode:
            return ['{"explanation": "Dummy output for testing", "direction": "N"}']

        # Re‑use your helper to pull out the referenced vision assets
        messages = self.replace_image_tokens(self.conversation_surgey(messages))[0]
        # aerial_image = Image.open(messages['images'][0])
        # ground_image = Image.open(messages['images'][1])

        # images = [aerial_image, ground_image]
        images = [Image.open(img_path) for img_path in messages['images']]
        image_tensors = process_images(images, self.image_processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensors]

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.system = f"<|im_start|>system\n{messages['messages'][0]['content']}"
        conv.append_message(conv.roles[0], messages['messages'][1]['content'])
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        image_sizes = [image.size for image in images]

        # Generate response
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        return [text_outputs[0]]
    
    def conversation_surgey(self, 
                            conversation: List[Dict]) -> List[str]:
        """
        This function is used to get the conversation survey.
        """
        return conversation

