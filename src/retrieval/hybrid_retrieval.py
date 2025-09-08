from langchain.vectorstores.elasticsearch import ElasticsearchStore
import logging
import torch
import os
import base64
import json
from elasticsearch import Elasticsearch
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)
class Hybrid_search : 
    def __init__(self, config) : 
        self.es = Elasticsearch([config.es_url ])
        self.index_name = config.index_name  
        self.embed_model = config.embedding_model_name 
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def get_embeddings(self, query:str) : 
        try:
            query = query.replace("\n", " ")
            return self.embeddings.embed_query(query)
        except Exception as e:
            logger.error(f"Error fetching embedding: {str(e)}")

    def lexical_search(self, query: str, top_k: int):

        try : 
            lexical_query = {
                "size": top_k,
                "query": {"match": {"text": query}},
                "_source": ["text", "metadata", "title", "url", "chunk_id", "source", "sections_in_chunk", "depth"]  #Extract metadata stored in _source
            }

            lexical_results = self.es.search(index=self.index_name, body=lexical_query)
            lexical_hits = lexical_results["hits"]["hits"]
            max_bm25_score = max([hit.get("_score", 0) for hit in lexical_hits], default=1.0)
            for hit in lexical_hits:
                hit["_normalized_score"] = (hit.get("_score", 0) / max_bm25_score) if max_bm25_score else 0.0
            return lexical_hits

        except Exception as e :
            logger.error(f"Error in fetching data using BM25 : {e}") 
            return {}
            



    def semantic_search(self,query: str, top_k: int):
        try : 
            query_embedding = self.get_embeddings(query)
            script_query = {
                "size": top_k,
                "query": {
                    "script_score": {

                        "query": {
                        "bool": {
                            "filter": [
                                {"exists": {"field": "vector"}}
                            ]
                        }
                    },
                        "script": {
                            "source": "cosineSimilarity(params.query_embedding, 'vector') + 1.0",
                            "params": {
                                "query_embedding": query_embedding,
                            }
                        }
                    }
                },
                "_source": ["text", "metadata", "title", "url", "chunk_id", "source", "sections_in_chunk", "depth"]
            }
            semantic_results = self.es.search(index=self.index_name, body=script_query)
            semantic_hits = semantic_results["hits"]["hits"]
            max_semantic_score = max([hit["_score"] for hit in semantic_hits], default=1.0)
            for hit in semantic_hits:
                hit["_normalized_score"] = hit["_score"] / max_semantic_score
            return semantic_hits
        
        except Exception as  e : 
            logger.error(f"Error in fetching data using Semantic Search : {e}")
            return {}


    def reciprocal_rank_fusion(self, query, lexical_hits, semantic_hits, k=60, top_k=2):
        try : 
            rrf_scores = {}
            for rank, hit in enumerate(lexical_hits, start=1):
                doc_id = hit["_id"]
                src = hit.get("_source", {})
                rrf_scores.setdefault(doc_id, {
                    "id": doc_id,
                    "text": src.get("text"),
                    "metadata": src.get("metadata"),
                    "title": src.get("title"),
                    "url": src.get("url"),
                    "lexical_score": 0,
                    "semantic_score": 0,
                    "rrf_score": 0,
                })
                rrf_scores[doc_id]["lexical_score"] = hit.get("_normalized_score", 0)
                rrf_scores[doc_id]["rrf_score"] += 1 / (k + rank)

            for rank, hit in enumerate(semantic_hits, start=1):
                doc_id = hit["_id"]
                src = hit.get("_source", {})
                rrf_scores.setdefault(doc_id, {
                    "id": doc_id,
                    "text": src.get("text"),
                    "metadata": src.get("metadata"),
                    "title": src.get("title"),
                    "url": src.get("url"),
                    "lexical_score": 0,
                    "semantic_score": 0,
                    "rrf_score": 0,
                })
                rrf_scores[doc_id]["semantic_score"] = hit.get("_normalized_score", 0)
                rrf_scores[doc_id]["rrf_score"] += 1 / (k + rank)

            sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

            # Print for debugging
            for i, doc in enumerate(sorted_results[:top_k], 1):
                print(f"{i}. id={doc['id']} rrf={doc['rrf_score']:.6f} text_snippet={doc['text'][:80] if doc['text'] else ''}")

            return sorted_results[:top_k]

        except Exception as e : 
            logger.error(f"Error in Reciprocal Rank Fusion : {e}")
            return {}



    def hybrid_search(self,query: str, lexical_top_k:int=5, semantic_top_k:int=5):
        lexical_hits = self.lexical_search(query, lexical_top_k)
        semantic_hits = self.semantic_search(query, semantic_top_k)
        rrf_results = self.reciprocal_rank_fusion(query, lexical_hits, semantic_hits, k=60, top_k=2)
        return rrf_results 

