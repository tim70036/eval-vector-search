"""Main evaluation runner using MLflow GenAI evaluate"""

import mlflow
import mlflow.genai
import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger

from eval_gvm_vector_search.config import Settings, SEARCH_CONFIGS
from eval_gvm_vector_search.vector_search import VectorSearchWrapper
from eval_gvm_vector_search.evaluator import (
    create_retrieval_relevance_judge,
    create_result_quality_judge,
    create_ranking_quality_judge
)



def get_config_name(config: dict) -> str:
    """
    Generate readable run name from search configuration.
    
    Args:
        config: Search configuration dictionary
        
    Returns:
        Readable string like "HYBRID_with_rerank" or "ANN_no_rerank"
    """
    query_type = config["query_type"]
    has_reranker = "with_rerank" if "reranker" in config else "no_rerank"
    return f"{query_type}_{has_reranker}"


def run_evaluation(settings: Settings, eval_queries: List[Dict]):
    """
    Run evaluation for all search configurations using MLflow GenAI evaluate.
    
    Args:
        settings: Application settings from Pydantic
        eval_queries: List of evaluation queries with metadata
    """
    logger.info(f"Starting evaluation with {len(eval_queries)} queries")
    logger.info(f"MLflow Experiment: {settings.mlflow_experiment_name}")
    logger.info(f"LLM Judge Endpoint: {settings.llm_judge_endpoint}")
    logger.info(f"Vector Search Index: {settings.vector_search_index}")
    logger.info(f"Number of results per query: {settings.num_results}")
            
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(settings.mlflow_experiment_name)
    
    vs_wrapper = VectorSearchWrapper(
        endpoint_name=settings.vector_search_endpoint,
        index_name=settings.vector_search_index
    )
    
    if not vs_wrapper.test_connection():
        logger.warning("Vector search connection test failed. Continuing anyway...")
    else:
        logger.info("Vector search connection successful")
    
    retrieval_relevance_judge = create_retrieval_relevance_judge(settings.llm_judge_endpoint)
    result_quality_judge = create_result_quality_judge(settings.llm_judge_endpoint)
    ranking_quality_judge = create_ranking_quality_judge(settings.llm_judge_endpoint)
    
    scorers = [
        retrieval_relevance_judge,
        result_quality_judge,
        ranking_quality_judge
    ]
    logger.info(f"Created {len(scorers)} custom judges")
    
    # Prepare evaluation dataframe
    eval_df = pd.DataFrame([
        {"inputs": {"query": q["query_text"]}}
        for q in eval_queries
    ])
    
    logger.info(f"Evaluating {len(SEARCH_CONFIGS)} search configurations")
    
    for idx, search_config in enumerate(SEARCH_CONFIGS, 1):
        config_name = get_config_name(search_config)
        logger.info(f"[{idx}/{len(SEARCH_CONFIGS)}] Evaluating: {config_name}")
        
        config_with_num = {**search_config, "num_results": settings.num_results}
        
        try:
            with mlflow.start_run(run_name=config_name):
                mlflow.log_params({
                    "config_name": config_name,
                    "query_type": search_config["query_type"],
                    "has_reranker": "reranker" in search_config,
                    "num_results": settings.num_results,
                    "num_queries": len(eval_queries),
                    "columns": "title,content"  # Hardcoded article schema
                })
                
                predict_fn = vs_wrapper.create_predict_function(config_with_num)
                
                results = mlflow.genai.evaluate(
                    data=eval_df,
                    predict_fn=predict_fn,
                    scorers=scorers
                )
                
                # Extract and aggregate judge scores from assessments
                judge_metrics = {}
                if "eval_results" in results.tables:
                    eval_table = results.tables["eval_results"]
                    
                    for judge_name in ["retrieval_relevance", "result_quality", "ranking_quality"]:
                        scores = []
                        
                        for assessments in eval_table.get('assessments', []):
                            if isinstance(assessments, list):
                                for assessment in assessments:
                                    if assessment.get('assessment_name') == judge_name:
                                        value = assessment.get('feedback', {}).get('value')
                                        if value is not None:
                                            try:
                                                scores.append(float(value))
                                            except (ValueError, TypeError):
                                                pass
                        
                        if scores:
                            judge_metrics[judge_name] = {
                                'mean': float(np.mean(scores)),
                                'min': float(np.min(scores)),
                                'max': float(np.max(scores)),
                                'median': float(np.median(scores)),
                                'variance': float(np.var(scores)),
                                'p90': float(np.percentile(scores, 90))
                            }
                            
                            for agg_name, agg_value in judge_metrics[judge_name].items():
                                mlflow.log_metric(f"{judge_name}/{agg_name}", agg_value)
                
                # Log summary
                relevance_mean = judge_metrics.get('retrieval_relevance', {}).get('mean', 0.0)
                quality_mean = judge_metrics.get('result_quality', {}).get('mean', 0.0)
                ranking_mean = judge_metrics.get('ranking_quality', {}).get('mean', 0.0)
                
                logger.info(f"Results for {config_name}: Relevance={relevance_mean:.2f}/5, Quality={quality_mean:.2f}/5, Ranking={ranking_mean:.2f}/5")
                
                # Log eval results table with query metadata directly to MLflow
                if "eval_results" in results.tables:
                    eval_table = results.tables["eval_results"].copy()
                    eval_table.insert(1, "query_text", [q["query_text"] for q in eval_queries])
                    eval_table.insert(2, "config_name", config_name)
                    
                    artifact_file = f"eval_results_{config_name}.json"
                    mlflow.log_table(data=eval_table, artifact_file=artifact_file)
                
                logger.info(f"Logged to MLflow run: {mlflow.active_run().info.run_id}")
                
        except Exception:
            logger.exception(f"Error evaluating {config_name}")
            continue
    
    logger.info(f"Evaluation complete! Results logged to: {settings.mlflow_experiment_name}")
    logger.info(f"View results at: {settings.databricks_host} → Machine Learning → Experiments")
