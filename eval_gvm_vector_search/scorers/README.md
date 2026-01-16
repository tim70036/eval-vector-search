# Scorer Architecture

This package provides a flexible, extensible architecture for evaluation judges (scorers) used in vector search evaluation.

## 🎯 Design Principles

- **No hardcoded scorer names**: Everything is registry-based
- **Easy to add/remove**: One file per scorer + registration
- **Auto-propagation**: Changes automatically flow through the entire system
- **Single source of truth**: Each scorer defines its own name, description, and instructions

## 📁 File Structure

```
scorers/
├── __init__.py                 # Auto-registration of all scorers
├── README.md                   # This file
├── base.py                     # BaseScorer abstract class
├── registry.py                 # ScorerRegistry for managing scorers
├── intent_fit.py               # Intent Fit scorer (意圖吻合度)
├── latent_interest_match.py    # Latent Interest Match scorer (潛在興趣吻合)
├── diversity.py                # Diversity scorer (多樣性/互補性)
├── engagement.py               # Engagement scorer (吸睛但不誤導)
├── ranking.py                  # Ranking scorer (排序品質)
└── credibility.py              # Credibility scorer (不誤導與可信度)
```

## ➕ How to Add a New Scorer

### Step 1: Create a new scorer file

Create a new file (e.g., `diversity_scorer.py`):

```python
"""Diversity scorer for evaluating result diversity"""

from eval_gvm_vector_search.scorers.base import BaseScorer


class DiversityScorer(BaseScorer):
    """
    Evaluates whether search results show good topic diversity.
    """
    
    @property
    def name(self) -> str:
        return "diversity"  # Used in MLflow metrics
    
    @property
    def description(self) -> str:
        return "Evaluates topic diversity in search results"
    
    @property
    def instructions(self) -> str:
        return """
Evaluate the diversity of the retrieved search results.

REQUEST: {{ inputs }}

RETRIEVED RESULTS:
{{ outputs }}

Criteria:
- Results should cover different aspects/angles of the topic
- Avoid redundancy and repetition
- Include varied perspectives when appropriate

Provide a score from 1-5 where:
1 = No diversity, highly repetitive
2 = Low diversity
3 = Moderate diversity
4 = Good diversity
5 = Excellent diversity, well-balanced coverage

Return ONLY the numeric score (1, 2, 3, 4, or 5).
"""
```

### Step 2: Register in `__init__.py`

Add these lines to `scorers/__init__.py`:

```python
from eval_gvm_vector_search.scorers.diversity_scorer import DiversityScorer

# In the registration section:
scorer_registry.register(DiversityScorer)

# In __all__:
__all__ = [
    # ... existing exports ...
    "DiversityScorer",
]
```

### That's it! 🎉

Your new scorer will automatically:
- ✅ Be used in all evaluations
- ✅ Generate MLflow metrics (e.g., `diversity/mean`, `diversity/p90`)
- ✅ Appear in evaluation logs
- ✅ Be included in result summaries

## ➖ How to Remove a Scorer

### Option 1: Comment out registration (temporary)

In `scorers/__init__.py`:

```python
# scorer_registry.register(RankingScorer)  # Disabled
```

### Option 2: Delete the file (permanent)

1. Delete the scorer file (e.g., `ranking.py`)
2. Remove its import and registration from `__init__.py`

The system will automatically adapt without any other changes needed!

## 🔧 Advanced: Scorer with Custom Configuration

You can extend scorers with custom parameters:

```python
class CustomScorer(BaseScorer):
    def __init__(self, llm_endpoint: str, custom_param: str = "default"):
        super().__init__(llm_endpoint)
        self.custom_param = custom_param
    
    @property
    def instructions(self) -> str:
        return f"""
        Custom rule: {self.custom_param}
        
        REQUEST: {{{{ inputs }}}}
        RESULTS: {{{{ outputs }}}}
        
        Score from 1-5...
        """
```

Then in `__init__.py`, create a custom instance:

```python
# For scorers with custom parameters, instantiate manually:
def _create_custom_scorers(llm_endpoint: str):
    return [
        CustomScorer(llm_endpoint, custom_param="special_value")
    ]

# Register in a different way if needed, or modify registry to support factories
```

## 📊 Registry API

```python
from eval_gvm_vector_search.scorers import scorer_registry

# Get all scorer instances
scorers = scorer_registry.get_all_scorers(llm_endpoint="my-endpoint")

# Get scorer names (for metric extraction)
names = scorer_registry.get_scorer_names()
# Returns: ['intent_fit', 'latent_interest_match', 'diversity', 'engagement', 'ranking', 'credibility']

# Get scorer info
info = scorer_registry.get_scorers_info()
# Returns: {'retrieval_relevance': 'Evaluates...', ...}

# Check number of registered scorers
count = len(scorer_registry)
```

## 🎓 Best Practices

1. **Keep scorers focused**: Each scorer should evaluate one specific aspect
2. **Clear naming**: Use descriptive names that will make sense in MLflow metrics
3. **Consistent scoring**: Always use 1-5 scale for comparability
4. **Good instructions**: Be specific about what you're evaluating and how to score
5. **Test independently**: Each scorer should work independently of others

## 🚀 Example: Full Workflow

```python
# 1. User creates new_scorer.py with NewScorer class
# 2. User edits __init__.py:
#    - Import: from eval_gvm_vector_search.scorers.new_scorer import NewScorer
#    - Register: scorer_registry.register(NewScorer)
# 3. Run evaluation → new scorer automatically used
# 4. MLflow logs metrics: new_scorer/mean, new_scorer/min, etc.
# 5. Done! No other code changes needed anywhere.
```

---

**Questions?** Check the `base.py` for the abstract interface or look at existing scorers for examples.
