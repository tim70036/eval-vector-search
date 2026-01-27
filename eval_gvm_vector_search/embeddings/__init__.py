"""Embedding providers package"""

from .base import BaseEmbedding
from .gemini import GeminiEmbedding

__all__ = [
    "BaseEmbedding",
    "GeminiEmbedding",
]
