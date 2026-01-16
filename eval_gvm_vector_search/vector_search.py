"""Vector search wrapper for Databricks Vector Search"""

from databricks.vector_search.client import VectorSearchClient
from loguru import logger


class VectorSearchWrapper:
    """Wrapper for Databricks Vector Search that formats results for MLflow GenAI evaluation"""
    
    def __init__(self, endpoint_name: str, index_name: str):
        """
        Initialize vector search client and index.
        
        Args:
            endpoint_name: Databricks vector search endpoint name
            index_name: Databricks vector search index name
        """
        self.vsc = VectorSearchClient()
        self.index = self.vsc.get_index(
            endpoint_name=endpoint_name,
            index_name=index_name
        )
        self.endpoint_name = endpoint_name
        self.index_name = index_name
    
    def search(self, query_text: str, search_config: dict) -> str:
        """
        Execute search and format results for LLM judge.
        
        Args:
            query_text: The search query
            search_config: Dictionary with search configuration
                - query_type: "HYBRID", "ANN", or "FULL_TEXT"
                - num_results: Number of results to retrieve
                - reranker: Optional DatabricksReranker instance
        
        Returns:
            Formatted string of search results suitable for LLM evaluation
        """
        # Hardcoded columns: title and content (article schema)
        columns = ["title", "content"]
        
        # Execute vector search
        results = self.index.similarity_search(
            query_text=query_text,
            columns=columns,
            **search_config
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
        
        return "\n".join(formatted_results)
    
    def create_predict_function(self, search_config: dict):
        """
        Create a predict function for mlflow.genai.evaluate.
        
        Args:
            search_config: Dictionary with search configuration
                - query_type: "HYBRID", "ANN", or "FULL_TEXT"
                - num_results: Number of results to retrieve
                - reranker: Optional DatabricksReranker instance
        
        Returns:
            Function that accepts query as keyword argument
            and returns formatted search results (context string)
        """
        def predict_fn(query: str) -> str:
            """
            Execute vector search for a query.
            
            Args:
                query: The search query string (passed from inputs dictionary)
                
            Returns:
                Formatted search results (context string)
            """
            # Execute search and format results
            return self.search(query, search_config)
        
        return predict_fn
    
    def test_connection(self) -> bool:
        """
        Test the connection to vector search index.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            _ = self.index.similarity_search(
                query_text="test",
                num_results=1,
                columns=["title"]
            )
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
