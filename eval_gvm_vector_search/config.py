"""Configuration using Pydantic Settings for environment-based config"""

import os
from typing import Any
from pydantic import Field
from pydantic_settings import BaseSettings
from databricks.vector_search.reranker import DatabricksReranker

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Databricks Configuration (from env or .env file)
    vector_search_endpoint: str = Field(..., description="Databricks vector search endpoint name")
    databricks_llm_judge_endpoint: str = Field(..., description="Databricks LLM endpoint name for judging")
    
    # MLflow Configuration for Databricks
    mlflow_experiment_name: str = Field(..., description="MLflow experiment name (e.g., /Users/username/vector-search-eval)")
    mlflow_max_workers: int = Field(..., description=" This controls the data-level concurrency, or how many data items (rows in your evaluation dataset) are evaluated in parallel. Prevent overloading the LLM judge endpoint.")
    mlflow_max_scorer_workers: int = Field(..., description=": This controls the scorer-level concurrency, or how many scorers run in parallel for each data item. P                    revent overloading the LLM judge endpoint.")
    # Databricks Configuration for MLflow authentication
    databricks_host: str = Field(..., description="Databricks workspace URL (e.g., https://your-workspace.cloud.databricks.com)")
    databricks_token: str = Field(..., description="Databricks personal access token")
    
    # Agent API Configuration (optional)
    agent_api_endpoint: str = Field(default="", description="Agent API endpoint URL (optional)")
    agent_api_timeout: int = Field(default=30, description="Agent API request timeout in seconds")
    
    # Recommend Products API Configuration (optional)
    recommend_products_api_endpoint: str = Field(default="", description="Recommend Products API endpoint URL (optional)")
    recommend_products_api_token: str = Field(default="", description="Recommend Products API bearer token (optional)")
    recommend_products_customer_uuid: str = Field(default="", description="Recommend Products API customer UUID (optional)")
    
    # Evaluation Configuration
    num_results: int = Field(..., description="Number of results to retrieve per query")
    
    # Gemini Embeddings Configuration (optional)
    gemini_api_key: str = Field(default="", description="Gemini API key for custom embeddings (optional)")
    gemini_embed_model: str = Field(default="", description="Gemini embedding model name (e.g., 'models/embedding-001', 'gemini-embedding-001')")
    
    class Config:
        env_file = ".env"
        env_prefix = "EVAL_"
        case_sensitive = False

    def model_post_init(self, __context: Any) -> None:
        """Set Databricks environment variables for MLflow authentication"""
        # Always set DATABRICKS_HOST (required by MLflow)
        os.environ["DATABRICKS_HOST"] = self.databricks_host

        os.environ["MLFLOW_GENAI_EVAL_MAX_WORKERS"] = str(self.mlflow_max_workers)
        os.environ["MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS"] = str(self.mlflow_max_scorer_workers)
        
        # Set authentication method (token or profile)
        if self.databricks_token:
            os.environ["DATABRICKS_TOKEN"] = self.databricks_token


# Unified source configurations (test matrix)
# Each source can be vector search, agent API, or other retrieval mechanisms
# This replaces the previous VECTOR_INDEXES and SEARCH_CONFIGS with a unified structure
SOURCE_CONFIGS = [
    # # Vector search source: baseline index
    # {
    #     "source_type": "vector_search",
    #     "label": "baseline",
    #     "index_name": "aigc_prod.intent_engine.content_article_gold_index",
    #     "search_configs": [
    #         {
    #             "query_type": "HYBRID"
    #         },
    #         {
    #             "query_type": "HYBRID",
    #             "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
    #         },
    #         {
    #             "query_type": "ANN"
    #         },
    #         {
    #             "query_type": "ANN",
    #             "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
    #         },
    #         {
    #             "query_type": "FULL_TEXT"
    #         },
    #         {
    #             "query_type": "FULL_TEXT",
    #             "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
    #         },
    #     ]
    # },

    # Vector search source: baseline index with Gemini embeddings
    {
        "source_type": "vector_search",
        "label": "baseline",
        "index_name": "aigc_prod.intent_engine.content_article_gemini_v1_gold_index",
        "use_gemini_embeddings": True,
        "gemini_embed_model": "gemini-embedding-001",
        "gemini_embed_dimension": 768,
        "search_configs": [
            {
                "query_type": "HYBRID"
            },
            {
                "query_type": "HYBRID",
                "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
            },
            {
                "query_type": "ANN"
            },
            {
                "query_type": "ANN",
                "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
            },
            {
                "query_type": "FULL_TEXT"
            },
            {
                "query_type": "FULL_TEXT",
                "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
            },
        ]
    },

    # Vector search source: improved embeddings index
    # {
    #     "source_type": "vector_search",
    #     "label": "embed_search_summary",
    #     "index_name": "aigc_prod.intent_engine.content_article_gold_v2_index",
    #     "search_configs": [
    #         {
    #             "query_type": "HYBRID"
    #         },
    #         {
    #             "query_type": "HYBRID",
    #             "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
    #         },
    #         {
    #             "query_type": "ANN"
    #         },
    #         {
    #             "query_type": "ANN",
    #             "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
    #         },
    #         {
    #             "query_type": "FULL_TEXT"
    #         },
    #         {
    #             "query_type": "FULL_TEXT",
    #             "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])
    #         },
    #     ]
    # },    



    # Recommend Products API source (uncomment and configure when ready to use)
    # {
    #     "source_type": "agent_api",
    #     "label": "recommend_products_api",
    #     # Uses env vars: EVAL_RECOMMEND_PRODUCTS_API_ENDPOINT,
    #     #                EVAL_RECOMMEND_PRODUCTS_API_TOKEN,
    #     #                EVAL_RECOMMEND_PRODUCTS_CUSTOMER_UUID
    #     # Everything else (article="", k=10) is hardcoded in AgentAPISource
    #     "configs": [{}]  # Single empty config - all params hardcoded
    # },
]
