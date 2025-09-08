from elasticsearch import Elasticsearch
from ..retrieval.hybrid_retrieval import Hybrid_search
import logging 
logger = logging.getLogger(__name__)
from ..generation.llm_generator import OpenAIGenerator

class InferencePipeline : 

    def __init__(self,config) : 
        self.es = Elasticsearch([config.es_url]) 
        self.index_name = config.index_name 
        self.llm_gen = OpenAIGenerator("gpt-4o")
        self.search= Hybrid_search(config) 


    def process_query(self, query:str,lexical_topk:int=3, semantic_top_k:int=3) : 
        try: 
            documents = self.search.hybrid_search(query, lexical_topk, semantic_top_k)
            logger.info(f"The  Number of Documents Extracted : {len(documents)}") 
            logger.info(f"The sample document : {documents[0]}")
            response =  self.llm_gen.generate(query,documents) 

            return response

        except Exception as e :
            logger.info(f"Exception is {e}") 
            return {"error" : e}

    

        
