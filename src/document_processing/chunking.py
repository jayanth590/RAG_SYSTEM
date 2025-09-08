from abc import ABC, abstractmethod
from typing import List,Dict,Any 
import re 
from transformers import AutoTokenizer
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter


class BaseChunker(ABC) : 
    
    @abstractmethod 
    def create_chunk(self, text) : 
        pass 

    
class RecursiveChunker(BaseChunker) : 
    
    def __init__(self, chunk_size:int=512, overlap:int=50) : 
        self.chunk_size = chunk_size 
        self.overlap = overlap 

    def create_chunk(self, text) : 
        logging.info(f"Loading documents from {files}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(text)
        #logging.info("Printing texts")
        #logging.info(texts)
        return chunks 
        

