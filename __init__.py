"""Top-level package for openai_image_api."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """Xin"""
__email__ = "unicough.github@gmail.com"
__version__ = "0.0.1"

from .src.openai_image_api.nodes import NODE_CLASS_MAPPINGS
from .src.openai_image_api.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
