from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, context_documents: List[Dict[str, Any]], 
                      system_prompt: str = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _prepare_context(self,documents) : 
        pass

    @abstractmethod 
    def _get_default_system_prompt(self) : 
        pass 

