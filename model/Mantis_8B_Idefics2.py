import os
from typing import List, Dict

import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

from .model import InferenceModel, InferenceModelConfig


class MantisInterleave(InferenceModel):
    """
    Wraps a Mantis-8B Vision2Seq model into the same interface as LlavaInterleaveQwen.
    Expects two images in each user turn and a text query.
    """

    image_token = ""

    def __init__(self, config: InferenceModelConfig):
        self.model_path = config.model_path
        self.config = config
        self.test_mode = config.test_mode

        # Load processor & model just like in Segment 1
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            device_map="auto",
        )

        # You can tweak these generation settings if you like
        self.generation_kwargs = {
            "max_new_tokens": 1024,
            "num_beams": 1,
            "do_sample": False,
        }

    def __call__(self, messages: List[Dict]) -> List[str]:
        # If test mode is on, bypass real inference
        if self.test_mode:
            return ['{"explanation": "Dummy output for testing", "direction": "N"}']

        messages = self.replace_image_tokens(self.conversation_surgey(messages))[0]
        aerial_image = Image.open(messages['images'][0])
        ground_image = Image.open(messages['images'][1])

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": messages['messages'][0]['content']},
                ],
                
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": messages['messages'][1]['content']},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        # print(prompt)

        # --- 5. Tokenize text+images ---
        inputs = self.processor(
            text=prompt,
            images=[aerial_image, ground_image],
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # --- 6. Generate ---
        generated_ids = self.model.generate(**inputs, **self.generation_kwargs)

        response = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("ASSISTANT: ", response[0])
        return [response[0]]