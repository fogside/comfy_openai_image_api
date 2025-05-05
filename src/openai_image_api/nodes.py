from inspect import cleandoc
import base64
import numpy as np
import torch
from PIL import Image
import io
from openai import OpenAI
import os

# ANSI escape codes for colors
RED = "\033[91m"
RESET = "\033[0m"

class OpenAIImageAPI:
    """
    A node for generating images using OpenAI's Image API
    
    This node allows users to generate or edit images using OpenAI's DALL-E 3 or GPT-Image-1 models.
    It supports various output sizes, quality settings, and can work with both single and multiple input images.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A beautiful image"
                }),
                "model": (["gpt-image-1"],),
                "size": (["1024x1024", "1536x1024", "1024x1536"],),
                "quality": (["low", "medium", "high"],),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "image/OpenAI"

    def generate_image(self, prompt, model, size, quality, image=None):
        # print(f"{RED}generate_image: {prompt}, {model}, {size}, {quality}, {image}{RESET}")

        # Read API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        try:
            if image is None or (isinstance(image, torch.Tensor) and image.numel() == 0):
                # If no input image, use generate API
                result = client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality
                )
            else:
                # Convert ComfyUI image tensor(s) to PIL Images
                images = []
                if len(image.shape) == 4:  # Batch of images
                    for i, img in enumerate(image):
                        # Convert PyTorch tensor to NumPy array
                        img_np = img.detach().cpu().numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        pil_image = Image.fromarray(img_np)
                        # Convert PIL Image to bytes with filename
                        img_byte_arr = io.BytesIO()
                        pil_image.save(img_byte_arr, format='PNG')
                        img_byte_arr_value = img_byte_arr.getvalue()
                        images.append((f"image_{i}.png", img_byte_arr_value))
                else:
                    # Single image
                    # Convert PyTorch tensor to NumPy array
                    img_np = image.detach().cpu().numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_np)
                    # Convert PIL Image to bytes with filename
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    img_byte_arr_value = img_byte_arr.getvalue()
                    images.append(("image_0.png", img_byte_arr_value))
                
                # Call edit API
                result = client.images.edit(
                    model=model,
                    image=images,
                    prompt=prompt,
                    size=size,
                    quality=quality
                )
            
            # Get the generated image
            image_base64 = result.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # Convert to PIL Image
            generated_image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            image_np = np.array(generated_image).astype(np.float32) / 255.0
            image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
            
            # Convert to torch tensor
            image_tensor = torch.from_numpy(image_np)
            
            return (image_tensor,)
            
        except Exception as e:
            error_message = f"{str(e)}"
            print(f"{RED}Error calling OpenAI Image API: {error_message}{RESET}")
            # Raise an exception to signal the error to the ComfyUI frontend
            raise RuntimeError(error_message) from e

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "OpenAI Image API": OpenAIImageAPI
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAI Image API": "OpenAI Image API with gpt-image-1"
}
