"""Agent API source implementation for Recommend Products API"""

import requests
from typing import Dict, List, Any
from loguru import logger

from .base import BaseRetrievalSource


class AgentAPISource(BaseRetrievalSource):
    """
    Retrieval source for Recommend Products API.
    
    Hardcoded to work with the Recommend Products API that returns
    articles, books, and activities. Automatically filters to articles only.
    """
    
    def __init__(
        self,
        api_endpoint: str,
        label: str,
        configs: List[Dict],
        timeout: int = 30,
        num_results: int = 10,
        api_token: str = "",
        customer_uuid: str = ""
    ):
        """
        Initialize Recommend Products API source.
        
        Args:
            api_endpoint: API endpoint URL
            label: Short label for this source
            configs: List of configuration dictionaries (can be [{}] for single default)
            timeout: Request timeout in seconds (default: 30)
            num_results: Number of results (hardcoded to 10 in requests)
            api_token: Bearer token for authentication
            customer_uuid: Customer UUID for API requests
        """
        self.api_endpoint = api_endpoint
        self._label = label
        self.configs = configs if configs else [{}]  # Default to single empty config
        self.timeout = timeout
        self.num_results = num_results
        self.customer_uuid = customer_uuid
        
        # Configure requests session with Bearer token
        self.session = requests.Session()
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"
        
        self.session.headers.update(headers)
    
    @property
    def source_label(self) -> str:
        """Get the label for this source."""
        return self._label
    
    def _format_request(self, query: str) -> Dict:
        """
        Format request for Recommend Products API - hardcoded.
        
        Args:
            query: Search query string
            config: Configuration dictionary (ignored, all hardcoded)
            
        Returns:
            Request payload for Recommend Products API
        """
        return {
            "article": "",  # Hardcoded: no article context
            "question": query,
            "k": 5,  # Hardcoded: always 5 results
            "customer_uuid": self.customer_uuid,
            "product_types": ["articles"]
        }
    
    def _parse_response(self, response_data: Any) -> str:
        """
        Parse Recommend Products API response - articles only.
        
        Extracts only articles from the multi-vertical response.
        
        Args:
            response_data: Response JSON from API
            
        Returns:
            Formatted string of article results
        """
        results_by_vertical = response_data.get("results_by_vertical", [])
        
        if not results_by_vertical:
            return "No results found."
        
        # Find articles vertical
        articles_data = None
        for vertical_data in results_by_vertical:
            if vertical_data.get("vertical") == "articles":
                articles_data = vertical_data
                break
        
        if not articles_data:
            return "No article results found."
        
        products = articles_data.get("products", [])
        if not products:
            return "No article results found."
        
        # Format articles
        formatted_results = []
        for i, product in enumerate(products, 1):
            title = product.get("title", "N/A")
            description = product.get("description", "")
            relevance_score = product.get("relevance_score", 0.0)
            
            # Truncate description
            if description:
                description_preview = description[:300] + "..." if len(description) > 300 else description
            else:
                description_preview = "No description available"
            
            result_text = (
                f"Result {i}:\n"
                f"  Title: {title}\n"
                f"  Content: {description_preview}\n"
                f"  Relevance Score: {relevance_score:.4f}\n"
            )
            
            # Add metadata
            metadata = product.get("metadata", {})
            if metadata:
                authors = metadata.get("authors", [])
                categories = metadata.get("categories", [])
                publish_time = metadata.get("publish_time", "")
                
                if authors:
                    result_text += f"  Authors: {', '.join(authors)}\n"
                if categories:
                    result_text += f"  Categories: {', '.join(categories)}\n"
                if publish_time:
                    result_text += f"  Published: {publish_time}\n"
            
            formatted_results.append(result_text)
        
        return "\n".join(formatted_results) if formatted_results else "No article results found."
    
    def retrieve(self, query: str, config: Dict) -> str:
        """
        Call Recommend Products API and format results.
        
        Args:
            query: The search query string
            config: Configuration dictionary (unused, all hardcoded)
        
        Returns:
            Formatted string of article results
        """
        try:
            # Format request payload (hardcoded)
            payload = self._format_request(query)
            
            logger.debug(f"Calling Recommend Products API: {self.api_endpoint}")
            logger.debug(f"Payload: {payload}")
            
            # Make API request
            response = self.session.post(
                self.api_endpoint,
                json=payload,
                timeout=self.timeout
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response (articles only)
            response_data = response.json()
            formatted_results = self._parse_response(response_data)
            
            return formatted_results
            
        except requests.exceptions.Timeout:
            error_msg = f"API request timed out after {self.timeout}s"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        except requests.exceptions.HTTPError as e:
            error_msg = f"API HTTP error: {e.response.status_code} - {e.response.text[:200]}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"
        
        except Exception as e:
            error_msg = f"Unexpected error calling API: {str(e)}"
            logger.exception(error_msg)
            return f"Error: {error_msg}"
    
    def create_predict_function(self, config: Dict) -> Any:
        """
        Create a predict function for mlflow.genai.evaluate.
        
        Args:
            config: Configuration dictionary (unused)
        
        Returns:
            Function that accepts query and returns formatted article results
        """
        def predict_fn(query: str) -> str:
            """Execute API call for a query."""
            return self.retrieve(query, config)
        
        return predict_fn
    
    def get_config_name(self, config: Dict) -> str:
        """
        Generate readable run name.
        
        Args:
            config: Configuration dictionary (unused)
            
        Returns:
            Run name string
        """
        return self._label
    
    def get_configs(self) -> List[Dict]:
        """
        Get all configurations to evaluate.
        
        Returns:
            List of configuration dictionaries
        """
        return self.configs
