"""Factory for creating retrieval sources"""

from typing import Dict
from loguru import logger

from eval_gvm_vector_search.config import Settings
from .base import BaseRetrievalSource
from .vector_search_source import VectorSearchSource
from .agent_api_source import AgentAPISource


class SourceFactory:
    """Factory for creating retrieval sources based on configuration."""
    
    @staticmethod
    def create(source_config: Dict, settings: Settings) -> BaseRetrievalSource:
        """
        Create a retrieval source based on configuration.
        
        Args:
            source_config: Source configuration dictionary with at least "source_type" key
            settings: Application settings
            
        Returns:
            Instance of BaseRetrievalSource
            
        Raises:
            ValueError: If source_type is unknown or required fields are missing
        """
        source_type = source_config.get("source_type")
        
        if not source_type:
            raise ValueError("source_config must have 'source_type' field")
        
        if source_type == "vector_search":
            return SourceFactory._create_vector_search_source(source_config, settings)
        
        elif source_type == "agent_api":
            return SourceFactory._create_agent_api_source(source_config, settings)
        
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    @staticmethod
    def _create_vector_search_source(source_config: Dict, settings: Settings) -> VectorSearchSource:
        """
        Create a VectorSearchSource instance.
        
        Args:
            source_config: Configuration with keys:
                - label: Source label
                - index_name: Vector search index name
                - search_configs: List of search configurations
                - use_gemini_embeddings: Optional flag to enable Gemini embeddings
            settings: Application settings
            
        Returns:
            VectorSearchSource instance
        """
        required_fields = ["label", "index_name", "search_configs"]
        for field in required_fields:
            if field not in source_config:
                raise ValueError(f"vector_search source requires '{field}' field")
        
        logger.info(f"Creating VectorSearchSource: {source_config['label']}")
        
        # Check if Gemini embeddings should be used
        embedding_provider = None
        if source_config.get("use_gemini_embeddings"):
            if not settings.gemini_api_key or not settings.gemini_embed_model:
                raise ValueError(
                    "Gemini embeddings enabled but EVAL_GEMINI_API_KEY or "
                    "EVAL_GEMINI_EMBED_MODEL not set in environment"
                )
            
            from eval_gvm_vector_search.embeddings import GeminiEmbedding
            
            # Get dimension from source config only (not from global settings)
            dimension = source_config.get("gemini_embed_dimension")
            dimension = dimension if dimension and dimension > 0 else None  # 0 or None means use default
            
            if dimension:
                logger.info(f"Enabling Gemini embeddings with model: {settings.gemini_embed_model}, dimension: {dimension}")
            else:
                logger.info(f"Enabling Gemini embeddings with model: {settings.gemini_embed_model} (default dimension)")
            
            embedding_provider = GeminiEmbedding(
                api_key=settings.gemini_api_key,
                model=settings.gemini_embed_model,
                dimension=dimension
            )
        
        return VectorSearchSource(
            endpoint_name=settings.vector_search_endpoint,
            index_name=source_config["index_name"],
            label=source_config["label"],
            search_configs=source_config["search_configs"],
            num_results=settings.num_results,
            embedding_provider=embedding_provider
        )
    
    @staticmethod
    def _create_agent_api_source(source_config: Dict, settings: Settings) -> AgentAPISource:
        """
        Create an AgentAPISource instance for Recommend Products API.
        
        Args:
            source_config: Configuration with keys:
                - label: Source label
                - api_endpoint: Optional (from EVAL_RECOMMEND_PRODUCTS_API_ENDPOINT)
                - api_token: Optional (from EVAL_RECOMMEND_PRODUCTS_API_TOKEN)
                - customer_uuid: Optional (from EVAL_RECOMMEND_PRODUCTS_CUSTOMER_UUID)
                - configs: List of configurations (can be [{}])
                - timeout: Optional request timeout
            settings: Application settings
            
        Returns:
            AgentAPISource instance
        """
        # Get API endpoint
        api_endpoint = source_config.get("api_endpoint") or settings.recommend_products_api_endpoint
        if not api_endpoint:
            raise ValueError(
                "agent_api source requires 'api_endpoint' in source_config or "
                "EVAL_RECOMMEND_PRODUCTS_API_ENDPOINT in settings"
            )
        
        # Get API token
        api_token = source_config.get("api_token") or settings.recommend_products_api_token
        if not api_token:
            raise ValueError(
                "agent_api source requires 'api_token' in source_config or "
                "EVAL_RECOMMEND_PRODUCTS_API_TOKEN in settings"
            )
        
        # Get customer UUID
        customer_uuid = source_config.get("customer_uuid") or settings.recommend_products_customer_uuid
        if not customer_uuid:
            raise ValueError(
                "agent_api source requires 'customer_uuid' in source_config or "
                "EVAL_RECOMMEND_PRODUCTS_CUSTOMER_UUID in settings"
            )
        
        required_fields = ["label"]
        for field in required_fields:
            if field not in source_config:
                raise ValueError(f"agent_api source requires '{field}' field")
        
        logger.info(f"Creating AgentAPISource: {source_config['label']}")
        
        # Get optional parameters
        timeout = source_config.get("timeout", settings.agent_api_timeout)
        configs = source_config.get("configs", [{}])  # Default to single empty config
        
        return AgentAPISource(
            api_endpoint=api_endpoint,
            label=source_config["label"],
            configs=configs,
            timeout=timeout,
            num_results=settings.num_results,
            api_token=api_token,
            customer_uuid=customer_uuid
        )
