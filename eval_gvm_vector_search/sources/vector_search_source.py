"""Vector search source implementation for Databricks Vector Search"""

from typing import Dict, List, Callable, Optional
from databricks.vector_search.client import VectorSearchClient
import mlflow
from mlflow.entities import SpanType

from eval_gvm_vector_search.embeddings import BaseEmbedding

from .base import BaseRetrievalSource


class VectorSearchSource(BaseRetrievalSource):
    """
    Retrieval source for Databricks Vector Search.
    
    Wraps Databricks Vector Search API and formats results for MLflow evaluation.
    """
    
    def __init__(
        self,
        endpoint_name: str,
        index_name: str,
        label: str,
        search_configs: List[Dict],
        num_results: int,
        embedding_provider: Optional[BaseEmbedding] = None
    ):
        """
        Initialize vector search source.
        
        Args:
            endpoint_name: Databricks vector search endpoint name
            index_name: Databricks vector search index name
            label: Short label for this index (e.g., "baseline", "improved_embeddings")
            search_configs: List of search configuration dictionaries
            num_results: Number of results to retrieve per query
            embedding_provider: Optional custom embedding provider (e.g., GeminiEmbedding)
        """
        self.vsc = VectorSearchClient()
        self.index = self.vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        self.endpoint_name = endpoint_name
        self.index_name = index_name
        self._label = label
        self.search_configs = search_configs
        self.num_results = num_results
        self.embedding_provider = embedding_provider
    
    @property
    def source_label(self) -> str:
        """Get the label for this source."""
        return self._label
    
    @mlflow.trace(name="vector_search_retrieve", span_type=SpanType.RETRIEVER)
    def retrieve(self, query: str, config: Dict) -> str:
        """
        Execute vector search and format results for LLM judge.
        
        Args:
            query: The search query string
            config: Search configuration dictionary
                - query_type: "HYBRID", "ANN", or "FULL_TEXT"
                - num_results: Number of results to retrieve
                - reranker: Optional DatabricksReranker instance
        
        Returns:
            Formatted string of search results suitable for LLM evaluation
        """        
        # Hardcoded columns: title and content (article schema)
        columns = ["title", "content"]
        
        # Execute vector search with custom embeddings or Databricks embeddings
        if self.embedding_provider:
            # Use custom embedding provider (e.g., Gemini)
            query_vector = self.embedding_provider.embed_query(query)
            
            # Check if reranker is present in config
            has_reranker = "reranker" in config
            is_hybrid = config["query_type"] == "HYBRID"
            is_full_text = config["query_type"] == "FULL_TEXT"
            
            if has_reranker or is_hybrid or is_full_text:
                # With reranker or hybrid or full text: pass both query_vector and query_text
                # Reranker needs query_text for semantic understanding, hybrid and full text need query_text for keyword search.
                results = self.index.similarity_search(
                    query_vector=query_vector,
                    query_text=query,
                    columns=columns,
                    **config
                )
            else:
                # Without reranker: pass only query_vector
                results = self.index.similarity_search(
                    query_vector=query_vector,
                    columns=columns,
                    **config
                )
        else:
            # Use Databricks embeddings (current behavior)
            results = self.index.similarity_search(
                query_text=query,
                columns=columns,
                **config
            )
        
        # Format results for LLM evaluation (as context)
        formatted_results = []
        
        # Databricks Vector Search returns: {"result": {"data_array": [...], "row_count": N}}
        # Each row in data_array is a list/tuple with columns in the order specified
        data_array = results.get("result", {}).get("data_array", [])
        
        if not data_array:
            return "No results found."
        
        # Get column names from manifest to map array positions
        manifest_columns = results.get("manifest", {}).get("columns", [])
        column_names = [col.get("name") for col in manifest_columns]
        
        # Find column indices (columns are in the order requested + score at the end)
        try:
            title_idx = column_names.index("title")
            content_idx = column_names.index("content")
        except ValueError:
            raise ValueError(f"Required columns not in results. Available columns: {column_names}")
        
        for i, row in enumerate(data_array, 1):
            # Each row is a list/tuple: [title, content, score]
            title = row[title_idx] if len(row) > title_idx else "N/A"
            content = row[content_idx] if len(row) > content_idx else ""
            
            # The last element is typically the similarity score
            score = row[-1] if len(row) > len(column_names) else None
            
            # Truncate content to keep context manageable
            content_preview = content[:300] + "..." if len(content) > 300 else content
            
            result_text = (
                f"Result {i}:\n"
                f"  Title: {title}\n"
                f"  Content: {content_preview}\n"
            )
            if score is not None:
                result_text += f"  Similarity Score: {score:.4f}\n"
            
            formatted_results.append(result_text)
        
        final_results = "\n".join(formatted_results)
    
        return final_results
    
    def create_predict_function(self, config: Dict) -> Callable[[str], str]:
        """
        Create a predict function for mlflow.genai.evaluate.
        
        Args:
            config: Search configuration dictionary
                - query_type: "HYBRID", "ANN", or "FULL_TEXT"
                - num_results: Number of results to retrieve
                - reranker: Optional DatabricksReranker instance
        
        Returns:
            Function that accepts query as keyword argument
            and returns formatted search results (context string)
        """
        # Add num_results to config
        config_with_num = {**config, "num_results": self.num_results}
        
        def predict_fn(query: str) -> str:
            """
            Execute vector search for a query.
            
            Args:
                query: The search query string (passed from inputs dictionary)
                
            Returns:
                Formatted search results (context string)
            """
            return self.retrieve(query, config_with_num)
        
        return predict_fn
    
    def get_config_name(self, config: Dict) -> str:
        """
        Generate readable run name from search configuration and index label.
        
        Args:
            config: Search configuration dictionary
            
        Returns:
            Readable string like "baseline_gemini_HYBRID_with_rerank" or "baseline_ANN_no_rerank"
        """
        query_type = config["query_type"]
        has_reranker = "reranker" in config
        rerank_suffix = "with_rerank" if has_reranker else "no_rerank"
        
        # Add embedding provider info to run name if using custom embeddings
        if self.embedding_provider:
            return f"{self._label}_gemini_{query_type}_{rerank_suffix}"
        else:
            return f"{self._label}_{query_type}_{rerank_suffix}"
    
    def get_configs(self) -> List[Dict]:
        """
        Get all search configurations to evaluate for this index.
        
        Returns:
            List of search configuration dictionaries
        """
        return self.search_configs
