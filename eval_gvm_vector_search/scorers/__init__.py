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
from eval_gvm_vector_search.scorers.intent_fit import IntentFitScorer
from eval_gvm_vector_search.scorers.latent_interest_match import LatentInterestMatchScorer
from eval_gvm_vector_search.scorers.diversity import DiversityScorer
from eval_gvm_vector_search.scorers.engagement import EngagementScorer
from eval_gvm_vector_search.scorers.ranking import RankingScorer
from eval_gvm_vector_search.scorers.credibility import CredibilityScorer

# Auto-register all scorers
# Add/remove scorers here - changes automatically propagate through the system
scorer_registry.register(IntentFitScorer)
scorer_registry.register(LatentInterestMatchScorer)
scorer_registry.register(DiversityScorer)
scorer_registry.register(EngagementScorer)
scorer_registry.register(RankingScorer)
scorer_registry.register(CredibilityScorer)

# Public exports
__all__ = [
    "BaseScorer",
    "ScorerRegistry",
    "scorer_registry",
    "IntentFitScorer",
    "LatentInterestMatchScorer",
    "DiversityScorer",
    "EngagementScorer",
    "RankingScorer",
    "CredibilityScorer",
]
