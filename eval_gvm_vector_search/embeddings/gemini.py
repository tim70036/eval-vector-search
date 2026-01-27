"""Gemini embedding provider implementation"""

from typing import List, Optional
from loguru import logger

from google import genai
from google.genai import types

from .base import BaseEmbedding


class GeminiEmbedding(BaseEmbedding):
    """
    Gemini embedding provider using Google GenAI SDK.
    
    Uses the google-genai library (not the deprecated google-generativeai).
    Supports models like 'models/embedding-001' and 'gemini-embedding-001'.
    """
    
    def __init__(self, api_key: str, model: str, dimension: Optional[int] = None):
        """
        Initialize Gemini embedding provider.
        
        Args:
            api_key: Gemini API key
            model: Model name (e.g., 'models/embedding-001', 'gemini-embedding-001')
            dimension: Optional output dimensionality (e.g., 256, 512, 768).
                      If None, uses the model's default dimension.
        """
        if not api_key:
            raise ValueError("Gemini API key is required")
        if not model:
            raise ValueError("Gemini model name is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.dimension = dimension
        
        if dimension:
            logger.info(f"Initialized GeminiEmbedding with model: {model}, dimension: {dimension}")
        else:
            logger.info(f"Initialized GeminiEmbedding with model: {model} (default dimension)")
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding vector for a query using Gemini API.
        
        Args:
            query: The text query to embed
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            Exception: If API call fails
        """
        try:
            logger.debug(f"Generating Gemini embedding for query: {query[:100]}...")
            
            # Configure embedding request with optional dimension
            if self.dimension:
                config = types.EmbedContentConfig(
                    output_dimensionality=self.dimension,
                    task_type='SEMANTIC_SIMILARITY'
                )
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=query,
                    config=config
                )
            else:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=query
                )
            
            embedding = response.embeddings[0].values
            
            logger.debug(f"Generated embedding with dimension: {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to generate Gemini embedding: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
