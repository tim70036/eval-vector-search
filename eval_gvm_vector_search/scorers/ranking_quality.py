"""Ranking quality scorer for evaluating result ordering quality"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class RankingQualityScorer(BaseScorer):
    """
    Evaluates whether search results are well-ranked.
    
    Ensures most relevant and engaging results appear in top positions.
    """
    
    @property
    def name(self) -> str:
        return "ranking_quality"
    
    @property
    def description(self) -> str:
        return "Evaluates whether results are well-ranked by relevance and engagement"
    
    @property
    def instructions(self) -> str:
        return """
Evaluate whether the search results are well-ranked.

REQUEST: {{ inputs }}

RETRIEVED RESULTS (in ranking order):
{{ outputs }}

CUSTOM RANKING RULES:
- Most relevant results must appear in top positions
- More engaging and eye-catching results should be ranked higher
- Ranking should follow logical relevance order
- Less relevant results should be ranked lower

Provide a score from 1-5 where:
1 = Poor ranking, relevant results buried
2 = Weak ranking with some good results lower
3 = Average ranking, could be improved
4 = Good ranking, most relevant on top
5 = Excellent ranking, perfect relevance order

Return ONLY the numeric score (1, 2, 3, 4, or 5).
"""
