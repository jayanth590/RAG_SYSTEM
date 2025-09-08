from abc import ABC, abstractmethod
from typing import List
import numpy as np

class BaseEmbeddings(ABC) :

    @abstractmethod
    def embed_documents(self, texts) : 
        '''Generating the embeddings of the chunked documents'''
        pass

    @abstractmethod
    def embed_query(self, query) : 
        '''Generating the embeddings of the query''' 
        pass 

    @abstractmethod
    def embed_text(self,text) : 
        '''Generating the embeddings of the complete text'''
        pass 
    
    @abstractmethod 
    def embed_texts(self,texts) :
        '''Generating the embeddings of the list of text'''
        pass 

    @abstractmethod
    def get_embedding_dimension() : 
        '''The dimnesion of the embeddings''' 
        pass 

    