# MLflow Multi-Source Retrieval Evaluation

Finding the perfect retrieval solution shouldn't be guesswork. This toolkit systematically evaluates different retrieval sources—comparing Databricks vector search (HYBRID, ANN, FULL_TEXT with/without reranking) against agent APIs—to discover which configuration delivers the most relevant, credible, and engaging results for your users.

Powered by MLflow GenAI's evaluation framework and custom LLM judges, this tool measures retrieval quality across multiple dimensions: relevance, credibility, diversity, engagement, and intent alignment. Run comprehensive evaluations, compare sources side-by-side in MLflow, and make data-driven decisions about your search infrastructure.

## Features

- **Multi-Source Support**: Compare vector search, agent APIs, and custom retrieval sources
- **Extensible Architecture**: Easy to add new retrieval sources with abstract base class
- **MLflow GenAI Evaluate**: Uses `mlflow.genai.evaluate` with custom judges
- **Custom Evaluation Rules**: Insert your own domain-specific rules into judge instructions
- **Multiple Search Configs**: Compare HYBRID/ANN/FULL_TEXT with/without reranking
- **Flexible Agent API Adapter**: Configurable request/response formatting for REST APIs
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

### Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template
cp env.template .env

# Edit .env with your values
EVAL_DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
EVAL_DATABRICKS_TOKEN=  # Optional if using DATABRICKS_CONFIG_PROFILE
EVAL_VECTOR_SEARCH_ENDPOINT=aigc-vector-search-endpoint
EVAL_DATABRICKS_LLM_JUDGE_ENDPOINT=databricks-meta-llama-3-1-70b-instruct
EVAL_MLFLOW_EXPERIMENT_NAME=/Users/your.email@company.com/vector-search-evaluation
EVAL_MLFLOW_MAX_WORKERS=3  # Maximum number of workers for MLflow evaluation
EVAL_MLFLOW_MAX_SCORER_WORKERS=3  # Maximum number of workers for running scorers/judges in parallel
EVAL_NUM_RESULTS=10

# Optional: Agent API configuration
EVAL_AGENT_API_ENDPOINT=https://your-agent-endpoint.com/search
EVAL_AGENT_API_TIMEOUT=30

# Optional: Recommend Products API configuration
EVAL_RECOMMEND_PRODUCTS_API_ENDPOINT=https://agent.aigc.mlytics.co/api/v1/recommend-products
EVAL_RECOMMEND_PRODUCTS_API_TOKEN=your-bearer-token
EVAL_RECOMMEND_PRODUCTS_CUSTOMER_UUID=your-customer-uuid
```

### Source Configuration

The evaluation sources are configured in `eval_gvm_vector_search/config.py` via the `SOURCE_CONFIGS` list. This unified configuration replaces the previous separate `VECTOR_INDEXES` and `SEARCH_CONFIGS`.

**Example: Vector Search Sources**

```python
{
    "source_type": "vector_search",
    "label": "baseline",
    "index_name": "aigc_prod.intent_engine.content_article_gold_index",
    "search_configs": [
        {"query_type": "HYBRID"},
        {"query_type": "HYBRID", "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])},
        {"query_type": "ANN"},
        {"query_type": "ANN", "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])},
        {"query_type": "FULL_TEXT"},
        {"query_type": "FULL_TEXT", "reranker": DatabricksReranker(columns_to_rerank=["title", "content"])},
    ]
}
```

**Example: Agent API Source**

```python
{
    "source_type": "agent_api",
    "label": "agent_api",
    "api_endpoint": "https://your-agent-endpoint.com/search",  # Or use EVAL_AGENT_API_ENDPOINT
    "configs": [
        {"config_name": "default"},
        {"config_name": "with_filter", "filter": "category:tech"},  # Custom configs
    ]
}
```

**Example: Recommend Products API Source (hardcoded formatting, articles only)**

```python
{
    "source_type": "agent_api",
    "label": "recommend_products_api",
    # Uses env vars: EVAL_RECOMMEND_PRODUCTS_API_ENDPOINT,
    #                EVAL_RECOMMEND_PRODUCTS_API_TOKEN,
    #                EVAL_RECOMMEND_PRODUCTS_CUSTOMER_UUID
    # Hardcoded: article="", k=10, articles only
    "configs": [{}]  # Single empty config
}
```

## Architecture

The refactored system uses a modular, extensible architecture:

```
eval_gvm_vector_search/
├── sources/
│   ├── base.py                    # Abstract BaseRetrievalSource
│   ├── vector_search_source.py    # Databricks Vector Search implementation
│   ├── agent_api_source.py        # Flexible REST API agent (supports Bearer auth)
│   └── factory.py                 # SourceFactory for instantiation
├── scorers/                       # Custom LLM judges (unchanged)
├── config.py                      # Unified SOURCE_CONFIGS + custom formatters
└── eval_runner.py                 # Simplified evaluation loop
```

**Key Components:**

- **BaseRetrievalSource**: Abstract interface that all sources implement
- **VectorSearchSource**: Wraps Databricks Vector Search API
- **AgentAPISource**: Recommend Products API source with hardcoded formatting (article="", k=10, articles only)
- **SourceFactory**: Creates source instances based on configuration
- **SOURCE_CONFIGS**: Unified configuration list for all retrieval sources

## Usage

```bash
# Run evaluation with uv
uv run eval-vector-search

# Or with python -m
python -m eval_gvm_vector_search.cli

# View results in Databricks
# Go to your Databricks workspace → Machine Learning → Experiments
# Find your experiment and use "Compare Runs" to see all configurations
```

## Adding Custom Retrieval Sources

The system is designed to be easily extensible. You can add new retrieval sources (e.g., Elasticsearch, Pinecone, custom APIs) by implementing the `BaseRetrievalSource` interface.

### Steps to Add a Custom Source

1. **Create a new source class** in `eval_gvm_vector_search/sources/` (e.g., `my_custom_source.py`):

```python
from typing import Dict, List, Callable
from .base import BaseRetrievalSource

class MyCustomSource(BaseRetrievalSource):
    def __init__(self, endpoint: str, label: str, configs: List[Dict], **kwargs):
        self.endpoint = endpoint
        self._label = label
        self.configs = configs
    
    @property
    def source_label(self) -> str:
        return self._label
    
    def retrieve(self, query: str, config: Dict) -> str:
        # Your retrieval logic here
        results = your_api_call(query, **config)
        return self.format_results(results)
    
    def create_predict_function(self, config: Dict) -> Callable[[str], str]:
        return lambda query: self.retrieve(query, config)
    
    def get_config_name(self, config: Dict) -> str:
        return f"{self._label}_{config.get('name', 'default')}"
    
    def get_configs(self) -> List[Dict]:
        return self.configs
```

2. **Register in the factory** (`sources/factory.py`):

```python
from .my_custom_source import MyCustomSource

# In SourceFactory.create():
elif source_type == "my_custom":
    return SourceFactory._create_my_custom_source(source_config, settings)
```

3. **Add to SOURCE_CONFIGS** in `config.py`:

```python
{
    "source_type": "my_custom",
    "label": "my_custom_retrieval",
    "endpoint": "https://my-api.com",
    "configs": [
        {"name": "config1"},
        {"name": "config2"},
    ]
}
```

## Customizing Agent API

The `AgentAPISource` supports custom request/response formatting:

```python
def custom_request_formatter(query: str, config: Dict) -> Dict:
    return {
        "messages": [{"role": "user", "content": query}],
        "parameters": config
    }

def custom_response_parser(response_data: Any) -> str:
    return response_data["choices"][0]["message"]["content"]

# In SOURCE_CONFIGS:
{
    "source_type": "agent_api",
    "label": "custom_agent",
    "api_endpoint": "https://api.example.com/chat",
    "request_formatter": custom_request_formatter,
    "response_parser": custom_response_parser,
    "configs": [{"temperature": 0.7}]
}
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
