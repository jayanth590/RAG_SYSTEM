from abc import ABC, abstractmethod 
from typing import List 
import numpy as np 

class BaseVectorStore(ABC)  :
    
    @abstractmethod 
    def _create_index_if_not_exists(self):
        """Create index with proper mapping if it doesn't exist."""
        pass 

    @abstractmethod
    def add_documents(self, documents,metadata) :
        """Add documents with embeddings to the store.""" 
        pass 

    @abstractmethod 
    def _manual_insert(self,documents,metadata)  :
        """Perform similarity search using query embedding.""" 
        pass 
        
