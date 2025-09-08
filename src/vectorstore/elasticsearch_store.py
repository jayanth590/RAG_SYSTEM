# 

from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import SentenceTransformer
from langchain_elasticsearch.vectorstores import ElasticsearchStore
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from elasticsearch.helpers import bulk
logger = logging.getLogger(__name__)

class ElasticSearchStore:
    def __init__(self, config):
        self.config = config
        self.es_url = config.es_url
        self.index_name = config.index_name 
        self.embed_model = config.embedding_model_name
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embed_model,
            model_kwargs={'device': 'cuda'}, 
            encode_kwargs={'normalize_embeddings': True}
        )

        sample_embedding = self.embeddings.embed_query("test")
        self.embedding_dim = len(sample_embedding)
        self.es = Elasticsearch([self.es_url])
        self._create_index_if_not_exists()
        
        logger.info(f" ElasticsearchStore initialized: {self.es_url}")
    
    def _create_index_if_not_exists(self):
        try:
            if not self.es.indices.exists(index=self.index_name):
                logger.info(f" Creating index: {self.index_name}")
                mapping = {
                    "mappings": {
                        "properties": {
                            "text": {"type": "text"},
                            "vector": {   
                                "type": "dense_vector",
                                "dims": self.embedding_dim,
                                "index": True,
                                "similarity": "cosine"
                            },
                            "metadata": {"type": "object", "enabled": False},
                            "content": {"type": "text"},
                            "url": {"type": "keyword"},
                            "title": {"type": "text"},
                            "chunk_id": {"type": "keyword"},
                            "source": {"type": "keyword"},
                            "sections_in_chunk": {"type": "integer"},
                            "depth": {"type": "integer"},
                        }
                    },
                    "settings": {
                        "number_of_shards": 1,
                        "number_of_replicas": 0,
                    }
                }
                
                self.es.indices.create(index=self.index_name, body=mapping)
                logger.info(f"Index '{self.index_name}' created successfully")
            else:
                logger.info(f"Index '{self.index_name}' already exists")
                
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
    
    def add_documents(self, documents: List[str], metadata: List[Dict[str, Any]]) -> int:
        try:
            logger.info(f"Adding {len(documents)} documents to index")
            langchain_docs = []
            for content, meta in zip(documents, metadata):
                doc = Document(
                    page_content=content,
                    metadata=meta
                )
                langchain_docs.append(doc)
            if len(langchain_docs) > 0:
                try:
                    db = ElasticsearchStore.from_documents(
                        documents=langchain_docs, 
                        embedding=self.embeddings,  #Function to convert embeddings of documents  
                        es_url=self.es_url, 
                        index_name=self.index_name 
                    )
                    
                    logger.info(f"Successfully indexed {len(documents)} documents")
                    return 1
                    
                except Exception as e:
                    logger.error(f"Error with from_documents: {str(e)}")
                    # Fallback to manual insertion
                    return self._manual_insert(documents, metadata)
            else:
                logger.warning("No documents to add")
                return 0
                
        except Exception as e:
            logger.error(f" Error in add_documents: {str(e)}")
            return 0
    
    def _manual_insert(self, documents: List[str], metadata: List[Dict[str, Any]]) -> int:
        try:
            logger.info("Using manual insertion fallback")
            embeddings = self.embeddings.encode(documents)
            bulk_data = []
            for i, (content, meta, embedding) in enumerate(zip(documents, metadata, embeddings)):
                doc_id = meta.get('chunk_id', f'doc_{i}')
                bulk_data.append({
                    "index": {
                        "_index": self.index_name,
                        "_id": doc_id
                    }
                })
            
                doc_data = {
                    "text": content,  
                    "content": content,  
                    "vector": embedding.tolist(),
                    "metadata": meta,
                    "url": meta.get('url', ''),
                    "title": meta.get('title', ''),
                    "chunk_id": meta.get('chunk_id', f'doc_{i}'),
                    "source": meta.get('source', ''),
                    "sections_in_chunk": meta.get('sections_in_chunk', 1),
                    "depth": meta.get('depth', 0),
                    "created_at": meta.get('created_at', '2025-01-01T00:00:00Z')
                }
                bulk_data.append(doc_data)
            
            success, failed = bulk(self.es, bulk_data, refresh=True)
            
            logger.info(f"Manual insertion: {success} successful, {len(failed)} failed")
            return 1 if success > 0 else 0
            
        except Exception as e:
            logger.error(f"Manual insertion failed: {str(e)}")
            return 0
