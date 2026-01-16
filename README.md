# MLflow Vector Search Evaluation

Finding the perfect search configuration shouldn't be guesswork. This toolkit systematically evaluates different Databricks vector search strategies—comparing HYBRID, ANN, and FULL_TEXT approaches with and without reranking—to discover which configuration delivers the most relevant, credible, and engaging results for your users.

Powered by MLflow GenAI's evaluation framework and custom LLM judges, this tool measures search quality across multiple dimensions: relevance, credibility, diversity, engagement, and intent alignment. Run comprehensive evaluations, compare configurations side-by-side in MLflow, and make data-driven decisions about your search infrastructure.

## Features

- **MLflow GenAI Evaluate**: Uses `mlflow.genai.evaluate` with custom judges
- **Custom Evaluation Rules**: Insert your own domain-specific rules into judge instructions
- **Multiple Search Configs**: Compare HYBRID/ANN/FULL_TEXT with/without reranking
- **Environment-Based Config**: Pydantic Settings with .env file

## Setup

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e .
```

## Configuration

Create a `.env` file in the project root:

```bash
# Copy template
cp env.template .env

# Edit .env with your values
EVAL_DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
EVAL_DATABRICKS_TOKEN=  # Optional if using DATABRICKS_CONFIG_PROFILE
EVAL_VECTOR_SEARCH_ENDPOINT=aigc-vector-search-endpoint
EVAL_VECTOR_SEARCH_INDEX=aigc_prod.intent_engine.content_article_gold_index
EVAL_DATABRICKS_LLM_JUDGE_ENDPOINT=databricks-meta-llama-3-1-70b-instruct
EVAL_MLFLOW_EXPERIMENT_NAME=/Users/your.email@company.com/vector-search-evaluation
EVAL_MLFLOW_MAX_WORKERS=3  # Maximum number of workers for MLflow evaluation. Prevent overloading the LLM judge endpoint.
EVAL_MLFLOW_MAX_SCORER_WORKERS=3  # Maximum number of workers for running scorers/judges in parallel. Controls concurrency for scorer execution to prevent overloading the LLM judge endpoint.
EVAL_NUM_RESULTS=10
```

## Usage

```bash
# Run evaluation with uv
uv run eval-vector-search

# Or with python -m
python -m eval_gvm_vector_search.cli

# View results in Databricks
# Go to your Databricks workspace → Machine Learning → Experiments
# Find your experiment and use "Compare Runs" to see all 6 configurations
```

## Customizing Scorers

The evaluation uses a registry-based scorer system. Each scorer is a custom LLM judge that evaluates search results on specific dimensions (intent fit, diversity, credibility, engagement, ranking quality).

### Adding a New Scorer

1. Create a new file in `eval_gvm_vector_search/scorers/` (e.g., `my_scorer.py`):

```python
from eval_gvm_vector_search.scorers.base import BaseScorer

class MyScorer(BaseScorer):
    @property
    def name(self) -> str:
        return "my_scorer"
    
    @property
    def description(self) -> str:
        return "Evaluates my custom criteria"
    
    @property
    def instructions(self) -> str:
        return """
        REQUEST: {{ inputs }}
        RETRIEVED RESULTS: {{ outputs }}
        
        Evaluate and return a score from 1-5.
        """
```

2. Register it in `scorers/__init__.py`:

```python
from eval_gvm_vector_search.scorers.my_scorer import MyScorer
scorer_registry.register(MyScorer)
```

The new scorer will automatically be used in all evaluations and generate MLflow metrics.

### Removing a Scorer

Comment out or remove the registration line in `scorers/__init__.py`. See existing scorers in `eval_gvm_vector_search/scorers/` for examples.
