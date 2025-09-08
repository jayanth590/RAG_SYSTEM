import sys
sys.path.append('.')
import logging
from src.pipeline.training_pipeline import TrainingPipeline
from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():

    print("Hi")
    
    url = "https://help.salesforce.com/s/articleView?id=data.c360_a_data_cloud.htm&type=5"
    
    logger.info(f"Starting training with {url} URLs")
    
    pipeline = TrainingPipeline(config)
    
    try:
        results = pipeline.process_urls(url)
        
        logger.info("Training Results:")
        logger.info(f"  - Processed URLs: {results['processed_urls']}")
        logger.info(f"  - Total chunks: {results['total_chunks']}")
        logger.info(f"  - Failed URLs: {len(results['failed_urls'])}")
        logger.info(f"  - Success URLs: {len(results['success_urls'])}")
        
        if results['failed_urls']:
            logger.warning(f"Failed URLs: {results['failed_urls']}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
