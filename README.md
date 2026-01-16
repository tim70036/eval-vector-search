# MLflow Vector Search Evaluation

Evaluate Databricks vector search configurations using MLflow GenAI with custom LLM judges.

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

## Customizing Evaluation Rules

Edit `eval_gvm_vector_search/evaluator.py` to insert your custom rules in the judge instructions:

```python
CUSTOM EVALUATION RULES:
- [INSERT YOUR DOMAIN-SPECIFIC RULES HERE]
- Results must mention specific keywords
- Technical accuracy requirements
- Brand guidelines compliance
```

## Project Structure

```
eval-gvm-vector-search/
├── pyproject.toml                    # Package configuration
├── uv.lock                           # Dependency lock file
├── .python-version                   # Python version
├── .gitignore                        # Git ignore rules
├── env.template                      # Environment variables template
├── eval_gvm_vector_search/           # Main package
│   ├── __init__.py
│   ├── cli.py                       # CLI entry point
│   ├── config.py                    # Pydantic Settings
│   ├── evaluator.py                 # Custom LLM judges
│   ├── vector_search.py             # Vector search wrapper
│   ├── eval_runner.py               # Main evaluation logic
│   └── logging_config.py            # Loguru logging configuration
└── data/
    └── eval_queries.json            # Evaluation queries
```
