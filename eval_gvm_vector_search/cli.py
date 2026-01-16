#!/usr/bin/env python3
"""
CLI entry point for running MLflow vector search evaluation.

Usage:
    python run_evaluation.py

Configuration is loaded from environment variables or .env file.
Required: EVAL_DATABRICKS_LLM_JUDGE_ENDPOINT
"""

import json
from pathlib import Path
from loguru import logger

from eval_gvm_vector_search.config import Settings
from eval_gvm_vector_search.eval_runner import run_evaluation
from eval_gvm_vector_search.logging_config import setup_logging

# Setup logging
setup_logging()


def load_eval_queries(queries_path: str = "data/eval_queries.json") -> list:
    """Load evaluation queries from JSON file."""
    queries = json.loads(Path(queries_path).read_text())
    logger.info(f"Loaded {len(queries)} queries from {queries_path}")
    return queries


def main():
    """Main entry point for evaluation"""
    logger.info("MLflow Vector Search Evaluation")
    
    settings = Settings()
    logger.info("Configuration loaded")
    
    eval_queries = load_eval_queries()
    run_evaluation(settings, eval_queries)


if __name__ == "__main__":
    main()
