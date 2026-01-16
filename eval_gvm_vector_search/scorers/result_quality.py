"""Result quality scorer for evaluating search result quality and usefulness"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class ResultQualityScorer(BaseScorer):
    """
    Evaluates the overall quality and usefulness of retrieved results.
    
    Focuses on engagement and clickability based on query intent.
    """
    
    @property
    def name(self) -> str:
        return "result_quality"
    
    @property
    def description(self) -> str:
        return "Evaluates quality, usefulness, and engagement of search results"
    
    @property
    def instructions(self) -> str:
        return """
Evaluate the quality and usefulness of the retrieved search results.

REQUEST: {{ inputs }}

RETRIEVED RESULTS:
{{ outputs }}

CUSTOM QUALITY RULES:
- Results should be eye-catching and engaging
- Results should be able induced users to click and read the article based on the query intent

Provide a score from 1-5 where:
1 = Poor quality, not useful
2 = Low quality with limited usefulness
3 = Acceptable quality, moderately useful
4 = Good quality, very useful
5 = Excellent quality, highly useful and accurate

Return ONLY the numeric score (1, 2, 3, 4, or 5).
"""
