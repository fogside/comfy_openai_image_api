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
                "size": (["auto", "1024x1024", "1536x1024", "1024x1536"], {"default": "auto"}),
                "quality": (["low", "medium", "high"],),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "image/OpenAI"

    def generate_image(self, prompt, model, size, quality, image=None, mask=None):
        # print(f"{RED}generate_image: {prompt}, {model}, {size}, {quality}, {image}, {mask}{RESET}")

        # Read API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable not set")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        try:
            if image is None or (isinstance(image, torch.Tensor) and image.numel() == 0):
                # If no input image, use generate API
                if size == "auto":
                    raise RuntimeError("Size 'auto' is not supported for image generation (no input image). Please select a specific size.")
                
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
                
                # Process mask if provided
                mask_bytes = None
                if mask is not None:
                    if len(image.shape) != 4 or image.shape[0] != 1:
                         raise RuntimeError("Mask input requires a single image input.")
                    if mask.shape[1:3] != image.shape[1:3]:
                        raise RuntimeError(f"Mask shape {mask.shape[1:3]} must match image shape {image.shape[1:3]}.")
                    
                    # Convert mask tensor to RGBA PIL Image
                    mask_tensor_squeezed = mask.squeeze(0) # Remove batch dim
                    height, width = mask_tensor_squeezed.shape
                    
                    # Create RGBA numpy array (H, W, C)
                    # OpenAI expects transparent areas (alpha=0) to be inpainted.
                    # ComfyUI mask is typically 1 for keep, 0 for replace.
                    # So, alpha = 1 - comfy_mask
                    rgba_mask_np = np.zeros((height, width, 4), dtype=np.uint8)
                    rgba_mask_np[:, :, 3] = ((1.0 - mask_tensor_squeezed.cpu().numpy()) * 255).astype(np.uint8)
                    
                    mask_pil = Image.fromarray(rgba_mask_np, 'RGBA')
                    
                    # Convert PIL Image to bytes
                    mask_byte_arr = io.BytesIO()
                    mask_pil.save(mask_byte_arr, format='PNG')
                    mask_bytes = mask_byte_arr.getvalue()

                # Call edit API
                edit_args = {
                    "model": model,
                    "image": images[0], # Pass the tuple (filename, image_bytes)
                    "prompt": prompt,
                    "quality": quality
                }
                if size != "auto": # Only add size if not auto
                    edit_args["size"] = size
                    
                if mask_bytes:
                    edit_args["mask"] = ("mask.png", mask_bytes) # Pass as tuple (filename, mask_bytes)
                
                result = client.images.edit(**edit_args)
            
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
