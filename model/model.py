import os
from swift.llm import PtEngine, RequestConfig, InferRequest
from typing import List, Dict
import gc
from dataclasses import dataclass
import torch

@dataclass
class InferenceModelConfig:
    """
    Configuration for the InferenceModel.
    """
    name: str
    class_path: str
    model_size: str 
    test_mode: bool
    model_path: str

class InferenceModel:
    """
    Thin wrapper around ms‑swift’s PtEngine so that the external API
    (constructor + __call__) remains identical to the original
    Hugging Face implementation.
    """
    image_token = ''

    def __init__(self, config: InferenceModelConfig):
        self.config = config
        self.test_mode = config.test_mode
        self.model_path = config.model_path

        if not self.test_mode:
            # Swift recommends these env‑vars; keep them here so callers
            # don’t have to remember.
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            os.environ.setdefault("MAX_PIXELS", "1003520")
            os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
            os.environ.setdefault("FPS_MAX_FRAMES", "12")

            # One engine instance for all subsequent calls
            if 'deepseek' in self.model_path:
                self.engine = PtEngine(self.model_path, max_batch_size=10)
            else:
                self.engine = PtEngine(self.model_path, max_batch_size=10, attn_impl='flash_attn')
            # You can expose these knobs if you need more control
            self.request_cfg = RequestConfig(max_tokens=512, temperature=0)
        else:
            self.engine = None
            self.request_cfg = None

    # ------------------------------------------------------------------ #
    # INFERENCE                                                          #
    # ------------------------------------------------------------------ #
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
        messages = self.replace_image_tokens(self.conversation_surgey(messages))
        infer_req = [InferRequest(**message) for message in messages]
        resps = self.engine.infer(infer_req, self.request_cfg)
        message_contents = [resp.choices[0].message.content for resp in resps]
        # message_contents = [message.content for message in resp.choices[0].message]
        return message_contents
    
    def replace_image_tokens(self, obj):
        """
        Recursively replace "<image>" → "image_token" in any string within
        a nested structure of dicts and lists.
        """
        if isinstance(obj, str):
            return obj.replace("<image>", self.image_token)
        elif isinstance(obj, dict):
            return {k: self.replace_image_tokens(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.replace_image_tokens(item) for item in obj]
        else:
            # non-str scalars (int, float, None, etc.) are left untouched
            return obj
    
    def conversation_surgey(self, 
                            conversation: List[Dict]) -> List[str]:
        """
        This function is used to get the conversation survey.
        """
        return conversation


import os

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
class LoRAInferenceModel(InferenceModel):
    """
    Thin wrapper around ms‑swift’s PtEngine so that the external API
    (constructor + __call__) remains identical to the original
    Hugging Face implementation.
    """

    def __init__(self, config: InferenceModelConfig):
        self.config = config
        self.test_mode = config.test_mode
        self.model_path = config.model_path
        self.lora_checkpoint = config.lora_checkpoint
        self.default_system = None

        if not self.test_mode:
            # ------------------------------------------------------------------
            # 1. environment tweaks (unchanged)
            # ------------------------------------------------------------------
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
            os.environ.setdefault("MAX_PIXELS", "1003520")
            os.environ.setdefault("VIDEO_MAX_PIXELS", "50176")
            os.environ.setdefault("FPS_MAX_FRAMES", "12")

            # ------------------------------------------------------------------
            # 2. base model + (optional) LoRA
            # ------------------------------------------------------------------
            base_model, tokenizer = get_model_tokenizer(self.model_path)

            if self.lora_checkpoint:            # ← NEW: apply LoRA if supplied
                lora_dir = safe_snapshot_download(self.lora_checkpoint)
                base_model = Swift.from_pretrained(base_model, lora_dir)

            # ------------------------------------------------------------------
            # 3. template & engine
            # ------------------------------------------------------------------
            template_type = getattr(config, "template_type", None) or base_model.model_meta.template
            template = get_template(template_type, tokenizer, default_system=self.default_system)

            self.engine = PtEngine.from_model_template(
                base_model, template, max_batch_size=4
            )
            self.request_cfg = RequestConfig(max_tokens=512, temperature=0)

        else:                                   # test-mode: stub everything out
            self.engine = None
            self.request_cfg = None