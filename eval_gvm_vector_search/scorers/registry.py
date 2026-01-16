"""Scorer registry for managing evaluation judges"""

from typing import List, Type, Dict
from loguru import logger

from eval_gvm_vector_search.scorers.base import BaseScorer


class ScorerRegistry:
    """
    Registry for managing all available scorers.
    
    Provides centralized scorer management:
    - Register new scorer classes
    - Get all scorer instances
    - Get scorer names for metric extraction
    
    Adding a new scorer is as simple as:
    1. Create scorer class inheriting from BaseScorer
    2. Call registry.register(NewScorer)
    """
    
    def __init__(self):
        """Initialize empty scorer registry."""
        self._scorer_classes: List[Type[BaseScorer]] = []
    
    def register(self, scorer_class: Type[BaseScorer]) -> None:
        """
        Register a scorer class.
        
        Args:
            scorer_class: Scorer class (not instance) to register
        """
        if not issubclass(scorer_class, BaseScorer):
            raise ValueError(f"{scorer_class.__name__} must inherit from BaseScorer")
        
        self._scorer_classes.append(scorer_class)
        logger.debug(f"Registered scorer: {scorer_class.__name__}")
    
    def get_all_scorers(self, llm_endpoint: str) -> List[BaseScorer]:
        """
        Get instances of all registered scorers.
        
        Args:
            llm_endpoint: Databricks LLM endpoint name for judging
            
        Returns:
            List of scorer instances
        """
        return [scorer_class(llm_endpoint) for scorer_class in self._scorer_classes]
    
    def get_scorer_names(self) -> List[str]:
        """
        Get names of all registered scorers.
        
        Useful for dynamic metric extraction without hardcoding names.
        
        Returns:
            List of scorer names
        """
        # Create temporary instances to get names
        # Note: We use a dummy endpoint since we only need the name
        temp_instances = [scorer_class("dummy") for scorer_class in self._scorer_classes]
        return [scorer.name for scorer in temp_instances]
    
    def get_scorers_info(self) -> Dict[str, str]:
        """
        Get information about all registered scorers.
        
        Returns:
            Dictionary mapping scorer names to descriptions
        """
        temp_instances = [scorer_class("dummy") for scorer_class in self._scorer_classes]
        return {scorer.name: scorer.description for scorer in temp_instances}
    
    def clear(self) -> None:
        """Clear all registered scorers (useful for testing)."""
        self._scorer_classes.clear()
    
    def __len__(self) -> int:
        """Return number of registered scorers."""
        return len(self._scorer_classes)


# Global registry instance
scorer_registry = ScorerRegistry()
