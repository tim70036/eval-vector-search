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
    vector_search_index: str = Field(..., description="Databricks vector search index name")
    databricks_llm_judge_endpoint: str = Field(..., description="Databricks LLM endpoint name for judging")
    
    # MLflow Configuration for Databricks
    mlflow_experiment_name: str = Field(..., description="MLflow experiment name (e.g., /Users/username/vector-search-eval)")
    mlflow_max_workers: int = Field(..., description=" This controls the data-level concurrency, or how many data items (rows in your evaluation dataset) are evaluated in parallel. Prevent overloading the LLM judge endpoint.")
    mlflow_max_scorer_workers: int = Field(..., description=": This controls the scorer-level concurrency, or how many scorers run in parallel for each data item. Prevent overloading the LLM judge endpoint.")
    # Databricks Configuration for MLflow authentication
    databricks_host: str = Field(..., description="Databricks workspace URL (e.g., https://your-workspace.cloud.databricks.com)")
    databricks_token: str = Field(..., description="Databricks personal access token")
    # Evaluation Configuration
    num_results: int = Field(..., description="Number of results to retrieve per query")
    
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
        

# Hardcoded search configurations (test matrix)
# These represent different search strategies to evaluate
# Note: columns ["title", "content"] are hardcoded in vector_search.py
SEARCH_CONFIGS = [
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
