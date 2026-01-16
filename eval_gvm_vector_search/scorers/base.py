"""Base scorer abstract class for evaluation judges"""

from abc import ABC, abstractmethod
from typing import Any


class BaseScorer(ABC):
    """
    Abstract base class for all evaluation scorers.
    
    Each scorer encapsulates:
    - Scorer name (used for MLflow metrics)
    - Judge instructions/prompts
    - MLflow judge creation logic
    """
    
    def __init__(self, llm_endpoint: str):
        """
        Initialize scorer with LLM endpoint.
        
        Args:
            llm_endpoint: Databricks LLM endpoint name for judging
        """
        self.llm_endpoint = llm_endpoint
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this scorer (used in MLflow metrics).
        
        Returns:
            Scorer name (e.g., "retrieval_relevance")
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this scorer evaluates.
        
        Returns:
            Description string
        """
        pass
    
    @property
    @abstractmethod
    def instructions(self) -> str:
        """
        Judge instructions/prompt template.
        
        Should include placeholders for:
        - {{ inputs }}: User query
        - {{ outputs }}: Retrieved results
        
        Returns:
            Instruction template string
        """
        pass
    
    def create_judge(self) -> Any:
        """
        Create MLflow GenAI judge with this scorer's configuration.
        
        Returns:
            MLflow GenAI judge object
        """
        from mlflow.genai.judges import make_judge
        
        return make_judge(
            name=self.name,
            instructions=self.instructions,
            model=f"databricks:/{self.llm_endpoint}"
        )
