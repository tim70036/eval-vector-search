"""Abstract base class for embedding providers"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """
    Abstract base class for all embedding providers.
    
    Each provider implements different embedding APIs (Gemini, OpenAI, Cohere, etc.)
    but provides a consistent interface for generating query embeddings.
    """
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding vector for a query string.
        
        Args:
            query: The text query to embed
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        pass
