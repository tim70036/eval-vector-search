"""Main evaluation runner using MLflow GenAI evaluate"""

import mlflow
import mlflow.genai
import pandas as pd
import numpy as np
from typing import List, Dict
from loguru import logger

from eval_gvm_vector_search.config import Settings, SOURCE_CONFIGS
from eval_gvm_vector_search.sources import SourceFactory
from eval_gvm_vector_search.scorers import scorer_registry



def run_evaluation(settings: Settings, eval_queries: List[Dict]):
    """
    Run evaluation for all search configurations using MLflow GenAI evaluate.
    
    Args:
        settings: Application settings from Pydantic
        eval_queries: List of evaluation queries with metadata
    """
    logger.info(f"Starting evaluation with {len(eval_queries)} queries")
    logger.info(f"MLflow Experiment: {settings.mlflow_experiment_name}")
    logger.info(f"LLM Judge Endpoint: {settings.databricks_llm_judge_endpoint}")
    logger.info(f"Number of sources: {len(SOURCE_CONFIGS)}")
    
    # Calculate total runs
    total_runs = sum(len(source_config.get("search_configs", source_config.get("configs", []))) 
                     for source_config in SOURCE_CONFIGS)
    logger.info(f"Total runs: {total_runs}")
    logger.info(f"Number of results per query: {settings.num_results}")
            
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(settings.mlflow_experiment_name)
    
    # Get all registered scorers from the registry
    scorer_instances = scorer_registry.get_all_scorers(settings.databricks_llm_judge_endpoint)
    scorers = [scorer.create_judge() for scorer in scorer_instances]
    scorer_names = scorer_registry.get_scorer_names()
    
    logger.info(f"Loaded {len(scorers)} scorers from registry: {scorer_names}")
    
    # Prepare evaluation dataframe
    eval_df = pd.DataFrame([
        {"inputs": {"query": q["query_text"]}}
        for q in eval_queries
    ])
    
    # Iterate through all sources
    for source_idx, source_config in enumerate(SOURCE_CONFIGS, 1):
        source_type = source_config.get("source_type")
        source_label = source_config.get("label")
        
        logger.info(f"Source [{source_idx}/{len(SOURCE_CONFIGS)}]: {source_label} ({source_type})")
        
        try:
            # Create source using factory
            source = SourceFactory.create(source_config, settings)
            
            # Get all configurations for this source
            configs = source.get_configs()
            
            # Iterate through configurations
            for config_idx, config in enumerate(configs, 1):
                config_name = source.get_config_name(config)
                logger.info(f"[Source {source_idx}/{len(SOURCE_CONFIGS)}, Config {config_idx}/{len(configs)}] Evaluating: {config_name}")
                
                try:
                    with mlflow.start_run(run_name=config_name):
                        # Log parameters
                        params = {
                            "config_name": config_name,
                            "source_type": source_type,
                            "source_label": source_label,
                            "num_results": settings.num_results,
                            "num_queries": len(eval_queries),
                        }
                        
                        # Add source-specific parameters
                        if source_type == "vector_search":
                            params["index_name"] = source_config.get("index_name", "")
                            params["query_type"] = config.get("query_type", "")
                            params["has_reranker"] = "reranker" in config
                            params["columns"] = "title,content"
                        elif source_type == "agent_api":
                            params["api_endpoint"] = source_config.get("api_endpoint", settings.recommend_products_api_endpoint)
                            params["customer_uuid"] = source_config.get("customer_uuid", settings.recommend_products_customer_uuid)
                        
                        mlflow.log_params(params)
                        
                        # Create predict function
                        predict_fn = source.create_predict_function(config)
                        
                        # Run evaluation
                        results = mlflow.genai.evaluate(
                            data=eval_df,
                            predict_fn=predict_fn,
                            scorers=scorers
                        )
                        
                        # Extract and aggregate judge scores from assessments (dynamic based on registry)
                        judge_metrics = {}
                        if "eval_results" in results.tables:
                            eval_table = results.tables["eval_results"]
                            
                            # Dynamically extract metrics for all registered scorers
                            for judge_name in scorer_names:
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
                                    q1 = float(np.percentile(scores, 25))
                                    q3 = float(np.percentile(scores, 75))
                                    iqr = q3 - q1
                                    fail_count = sum(1 for score in scores if score <= 2)
                                    fail_rate = fail_count / len(scores)
                                    
                                    judge_metrics[judge_name] = {
                                        'median': float(np.median(scores)),
                                        'mean': float(np.mean(scores)),
                                        'IQR': iqr,
                                        'fail-rate': fail_rate
                                    }
                                    
                                    for agg_name, agg_value in judge_metrics[judge_name].items():
                                        mlflow.log_metric(f"{judge_name}/{agg_name}", agg_value)
                        
                        # Log summary (dynamic based on all scorers)
                        summary_parts = [f"{config_name}:"]
                        for judge_name in scorer_names:
                            mean_score = judge_metrics.get(judge_name, {}).get('mean', 0.0)
                            summary_parts.append(f"{judge_name}={mean_score:.2f}/5")
                        
                        logger.info("Results for " + ", ".join(summary_parts))
                        
                        # Log eval results table with query metadata directly to MLflow
                        if "eval_results" in results.tables:
                            eval_table = results.tables["eval_results"].copy()
                            # Extract query_text from inputs column (each row may have multiple results per query)
                            if "inputs" in eval_table.columns:
                                query_texts = eval_table["inputs"].apply(
                                    lambda x: x.get("query", "") if isinstance(x, dict) else ""
                                )
                            else:
                                query_texts = pd.Series([""] * len(eval_table))
                            eval_table.insert(1, "query_text", query_texts)
                            eval_table.insert(2, "config_name", config_name)
                            
                            artifact_file = f"eval_results_{config_name}.json"
                            mlflow.log_table(data=eval_table, artifact_file=artifact_file)
                        
                        logger.info(f"Logged to MLflow run: {mlflow.active_run().info.run_id}")
                        
                except Exception:
                    logger.exception(f"Error evaluating {config_name}")
                    continue
                    
        except Exception:
            logger.exception(f"Error creating source {source_label}")
            continue
    
    logger.info(f"Evaluation complete! Results logged to: {settings.mlflow_experiment_name}")
    logger.info(f"View results at: {settings.databricks_host} → Machine Learning → Experiments")
