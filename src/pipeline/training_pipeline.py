import logging
from typing import List, Dict, Any
from ..document_processing.web_scraper_2 import SmartDocumentScraper
from ..vectorstore.elasticsearch_store import ElasticSearchStore
logger = logging.getLogger(__name__)

class TrainingPipeline:
    
    def __init__(self, config):
        self.config = config
        self.web_scraper = SmartDocumentScraper(
            api_key=getattr(config, 'serper_api_key'),
            max_depth=getattr(config, 'max_depth'),
            delay=1,
            max_links_per_page=getattr(config, 'max_links_per_page'),
            max_tokens_per_chunk=getattr(config, 'max_tokens_per_chunk')
        )
        self.vector_store = ElasticSearchStore(config)
        
        logger.info("Training pipeline initialized")
    
    def process_urls(self, url: str) -> Dict[str, Any]:
        try:
            logger.info(f"Starting training pipeline for URL: {url}")
            
            results = {
                'processed_urls': 0,
                'total_chunks': 0,
                'failed_urls': [],
                'success_urls': [],
                'processing_stats': {}
            }
            
            logger.info("Step 1: Web scraping")
            scraped_docs = self.web_scraper.scrape_to_documents(url)
            
            if not scraped_docs:
                logger.warning("No documents successfully scraped")
                return results
            
            logger.info(f"Successfully scraped {len(scraped_docs)} documents")
            
            # Step 2: Process Document objects correctly
            logger.info("Step 2: Processing documents for indexing")
            
            all_chunks = []  # List of content
            all_chunks_metadata = []  # List of metadata dict
            
            for doc in scraped_docs:
                content = doc.page_content 
                metadata_dict = doc.metadata
                processed_metadata = {
                    'url': metadata_dict.get('url', ''),
                    'title': metadata_dict.get('title', ''),
                    'content_length': metadata_dict.get('content_length', len(content)),
                    'source': metadata_dict.get('url', ''),
                    'sections_in_chunk': metadata_dict.get('sections_in_chunk', 1),
                    'chunk_id': metadata_dict.get('chunk_id', f'chunk_{len(all_chunks)}'),
                    'depth': metadata_dict.get('depth', 0),
                    'created_at': metadata_dict.get('scraped_at', '2025-01-01T00:00:00Z')
                }
                
                all_chunks.append(content)
                all_chunks_metadata.append(processed_metadata)
            
            logger.info(f"Processed {len(all_chunks)} documents for indexing")
            results['total_chunks'] = len(all_chunks)
            
            if not all_chunks:
                logger.warning("No chunks created from documents")
                return results
            
            # Step 3: Index in Elasticsearch
            logger.info("Step 3: Indexing documents in Elasticsearch")
            success = self.vector_store.add_documents(all_chunks, all_chunks_metadata)
            
            if success:
                logger.info(f"Successfully indexed {len(all_chunks)} documents")
                results['processed_urls'] = 1
                results['success_urls'] = [url]
            else:
                logger.error("Failed to index documents")
                results['failed_urls'] = [url]
            
            results['processing_stats'] = {
                'total_documents': len(all_chunks),
                'avg_content_length': sum(len(content) for content in all_chunks) / len(all_chunks) if all_chunks else 0,
                'unique_urls': len(set(meta['url'] for meta in all_chunks_metadata))
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
    
    def process_single_url(self, url: str) -> Dict[str, Any]:
        return self.process_urls(url)
