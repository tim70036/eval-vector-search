"""Abstract base class for retrieval sources"""

from abc import ABC, abstractmethod
from typing import Dict, List, Callable


class BaseRetrievalSource(ABC):
    """
    Abstract base class for all retrieval sources.
    
    Each source implements different retrieval mechanisms (vector search, agent API, etc.)
    but provides a consistent interface for MLflow evaluation.
    """
    
    @abstractmethod
    def retrieve(self, query: str, config: Dict) -> str:
        """
        Retrieve and format results for a given query.
        
        Args:
            query: The search query string
            config: Configuration dictionary specific to this retrieval
            
        Returns:
            Formatted string of results suitable for LLM evaluation
        """
        pass
    
    @abstractmethod
    def create_predict_function(self, config: Dict) -> Callable[[str], str]:
        """
        Create a predict function for mlflow.genai.evaluate.
        
        Args:
            config: Configuration dictionary for this specific run
            
        Returns:
            Function that accepts query as keyword argument
            and returns formatted results (context string)
        """
        pass
    
    @abstractmethod
    def get_config_name(self, config: Dict) -> str:
        """
        Generate a readable run name from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Readable string for MLflow run name
        """
        pass
    
    @abstractmethod
    def get_configs(self) -> List[Dict]:
        """
        Get all configurations to evaluate for this source.
        
        Returns:
            List of configuration dictionaries
        """
        pass
    
    @property
    @abstractmethod
    def source_label(self) -> str:
        """
        Get the label for this source (e.g., "baseline", "agent_api").
        
        Returns:
            Source label string
        """
        pass
