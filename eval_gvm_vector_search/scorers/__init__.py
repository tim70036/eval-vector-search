"""
Scorer package for vector search evaluation.

This package provides a flexible, extensible architecture for evaluation judges.

To add a new scorer:
1. Create a new file in this directory (e.g., my_new_scorer.py)
2. Inherit from BaseScorer and implement required methods
3. Import and register it below

To remove a scorer:
1. Remove or comment out its registration line below
"""

from eval_gvm_vector_search.scorers.base import BaseScorer
from eval_gvm_vector_search.scorers.registry import scorer_registry, ScorerRegistry
from eval_gvm_vector_search.scorers.retrieval_relevance import RetrievalRelevanceScorer
from eval_gvm_vector_search.scorers.result_quality import ResultQualityScorer
from eval_gvm_vector_search.scorers.ranking_quality import RankingQualityScorer

# Auto-register all scorers
# Add/remove scorers here - changes automatically propagate through the system
scorer_registry.register(RetrievalRelevanceScorer)
scorer_registry.register(ResultQualityScorer)
scorer_registry.register(RankingQualityScorer)

# Public exports
__all__ = [
    "BaseScorer",
    "ScorerRegistry",
    "scorer_registry",
    "RetrievalRelevanceScorer",
    "ResultQualityScorer",
    "RankingQualityScorer",
]
