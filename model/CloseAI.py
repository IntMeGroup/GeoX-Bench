import base64
import asyncio
import aiohttp
from openai import OpenAI
from typing import List, Dict, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from PIL import Image
import io

@dataclass
class CloseAIConfig:
    """
    Configuration for the CloseAI model.
    """
    name: str
    class_path: str
    model_size: str
    test_mode: bool
    model_path: str  # Used as model name in OpenAI client
    batch_size: int = 8  # Maximum concurrency
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0
    max_tokens: int = 512
    max_retries: int = 5  # Maximum number of retry attempts
    base_delay: float = 1.0  # Base delay for exponential backoff (seconds)

class CloseAI:
    """
    OpenAI-compatible model wrapper that supports concurrent processing with retry logic.
    Uses model_path from config as the model name in OpenAI client.
    """
    
    def __init__(self, config):
        self.config = config
        self.test_mode = getattr(config, 'test_mode', False)
        self.model_name = getattr(config, 'model_path', 'gpt-4o')
        self.max_concurrency = getattr(config, 'batch_size', 8)
        self.max_retries = getattr(config, 'max_retries', 5)
        self.base_delay = getattr(config, 'base_delay', 1.0)
        
        # Use defaults for missing config fields, falling back to env vars for sensitive data
        self.base_url = getattr(config, 'base_url', None) or os.getenv('CLOSEAI_BASE_URL')
        self.api_key = getattr(config, 'api_key', None) or os.getenv('CLOSEAI_API_KEY')
        self.temperature = getattr(config, 'temperature', 0)
        self.max_tokens = getattr(config, 'max_tokens', 8000)

        if not self.base_url:
            raise ValueError("CloseAI base URL must be provided via config or CLOSEAI_BASE_URL env var")
        if not self.api_key:
            raise ValueError("CloseAI API key must be provided via config or CLOSEAI_API_KEY env var")
        
        if not self.test_mode:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )
        else:
            self.client = None
            
    # def encode_image(self, image_path: str) -> str:
    #     """Encode image to base64 string."""
    #     with open(image_path, "rb") as image_file:
    #         return base64.b64encode(image_file.read()).decode('utf-8')
    

    def encode_image(self, image_path: str) -> str:
        """Resize image to 512x512 and encode to base64 string."""
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize to 512x512 using high-quality resampling
            # img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
            # print(img.size)
            img_resized = img
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            img_resized.save(buffer, format='JPEG', quality=100)
            buffer.seek(0)
            
            # Encode to base64
            return base64.b64encode(buffer.getvalue()).decode('utf-8')


    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        Returns True for connection errors, timeouts, and rate limiting.
        """
        import requests
        from openai import OpenAIError, RateLimitError, APITimeoutError, APIConnectionError
        
        # OpenAI specific errors that are retryable
        if isinstance(error, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True
        
        # General connection errors
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
        
        # Requests library errors
        if hasattr(requests, 'RequestException') and isinstance(error, requests.RequestException):
            return True
        
        # HTTP status codes that indicate temporary issues
        if hasattr(error, 'status_code'):
            # 429 (rate limit), 502 (bad gateway), 503 (service unavailable), 504 (gateway timeout)
            retryable_status_codes = [429, 502, 503, 504]
            if error.status_code in retryable_status_codes:
                return True
        
        # Check error message for common transient error patterns
        error_str = str(error).lower()
        transient_patterns = [
            'connection', 'timeout', 'rate limit', 'temporarily unavailable',
            'service unavailable', 'bad gateway', 'gateway timeout',
            'too many requests', 'server error'
        ]
        
        return any(pattern in error_str for pattern in transient_patterns)
    
    def _calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for exponential backoff with jitter.
        """
        import random
        
        # Exponential backoff: base_delay * (2 ** attempt)
        delay = self.base_delay * (2 ** attempt)
        
        # Add jitter to avoid thundering herd problem
        jitter = random.uniform(0.1, 0.3) * delay
        
        return delay + jitter
    
    def _process_single_message(self, message: Dict) -> str:
        """Process a single message with the OpenAI client with retry logic."""
        if self.test_mode:
            return '{"explanation": "Dummy output for testing", "direction": "N"}'
        
        last_error = None
        
        for attempt in range(self.max_retries + 1):  # +1 because we want to try max_retries + 1 times total
            try:
                # Extract message content and images
                messages = message.get("messages", [])
                images = message.get("images", [])
                
                # Convert images to base64 and insert into messages
                processed_messages = []
                image_index = 0
                
                for msg in messages:
                    if "<image>" in msg.get("content", ""):
                        # Process any message (system or user) with image placeholders
                        content = msg["content"]
                        content_parts = []
                        
                        # Split content by <image> tokens and insert images
                        parts = content.split("<image>")
                        for i, part in enumerate(parts):
                            if part:  # Add text part if not empty
                                content_parts.append({"type": "text", "text": part})
                            if i < len(parts) - 1 and image_index < len(images):  # Add image if available
                                base64_image = self.encode_image(images[image_index])
                                content_parts.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                })
                                image_index += 1
                        
                        processed_messages.append({
                            "role": msg["role"],
                            "content": content_parts
                        })
                    else:
                        # For messages without <image> markers, just copy as-is
                        processed_messages.append(msg)

                # Make API call
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=processed_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                print(response)
                
                return response.choices[0].message.content
                
            except Exception as e:
                last_error = e
                
                # Check if this is the last attempt or if error is not retryable
                if attempt == self.max_retries or not self._is_retryable_error(e):
                    print(f"Error processing message (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    return f'{{"error": "{str(e)}"}}'
                
                # Calculate delay and wait before retrying
                delay = self._calculate_delay(attempt)
                print(f"Retryable error on attempt {attempt + 1}/{self.max_retries + 1}: {e}")
                print(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        # This should never be reached, but just in case
        return f'{{"error": "{str(last_error)}"}}'
    
    def __call__(self, messages: List[Dict]) -> List[str]:
        """
        Process multiple messages with concurrent execution.
        
        Parameters
        ----------
        messages : List[Dict]
            List of message dictionaries containing:
            - messages: List of chat messages
            - images: List of image paths
        
        Returns
        -------
        List[str]
            List of model responses
        """
        if not messages:
            return []
            
        results = []
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_message, msg): i 
                for i, msg in enumerate(messages)
            }
            
            # Initialize results list with correct size
            results = [None] * len(messages)
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    print(f"Error processing message at index {index}: {e}")
                    results[index] = f'{{"error": "{str(e)}"}}'
        
        return results
