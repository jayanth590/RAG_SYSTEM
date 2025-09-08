from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List
from .base_embeddings import BaseEmbeddingModel

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(BaseEmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._dimension = self.model.get_sentence_embedding_dimension()
        self.model_name = model_name
        logger.info(f"Initialized SentenceTransformer with model: {model_name}, dimension: {self._dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self._dimension
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        valid_texts = [text.strip() if text.strip() else " " for text in texts]
        
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True, batch_size=32)
        return embeddings.tolist()
    
    def embed_documents(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, convert_to_numpy=True)
    
    def get_embedding_dimension(self) -> int:
        return self._dimension
