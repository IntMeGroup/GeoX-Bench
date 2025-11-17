import os
from swift.llm import PtEngine, RequestConfig, InferRequest
from typing import List, Dict
from .model import InferenceModel, InferenceModelConfig
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

class LlavaInterleaveQwen(InferenceModel):
    """
    LlavaInterleaveQwen7B model class.
    """
    image_token = ""

    def __init__(self, config: InferenceModelConfig):
        self.model_path = config.model_path
        self.config = config
        self.test_mode = config.test_mode
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def __call__(self, messages: List[Dict]) -> List[str]:
        if self.test_mode:
            return ['{"explanation": "Dummy output for testing", "direction": "N"}']

        messages = self.replace_image_tokens(self.conversation_surgey(messages))[0]
        # aerial_image = Image.open(messages['images'][0])
        # ground_image = Image.open(messages['images'][1])
        images = [Image.open(img_path) for img_path in messages['images']]

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
                    {"type": "text", "text": messages['messages'][1]['content']},
                    {"type": "image"},
                    {"type": "image"},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # 4. Tokenize & preprocess (images + prompt)
        inputs = self.processor(
            # images=[aerial_image, ground_image],
            images = images,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, torch.float16)

        # 5. Generate and decode
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )
        # skip the first two special tokens before decoding
        decoded = self.processor.decode(output_ids[0][:], skip_special_tokens=True)
        # Remove everything before and including the user content
        user_content = messages['messages'][1]['content']
        if user_content in decoded:
            decoded = decoded[decoded.index(user_content) + len(user_content):]
        decoded = decoded.replace("assistant\n", "")
        return [decoded]
