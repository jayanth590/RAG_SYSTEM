from pydantic_settings import BaseSettings
import os 
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass

@dataclass
class Config():
    # Elasticsearch settings
    host: str = "your_es_url"
    port: int = 9205
    es_url: str = f"http://{host}:{port}"
    index_name: str = "your_es_index"
    username: str = None
    password: str = None
    
    # Embedding settings
    embedding_model_name: str = "hkunlp/instructor-large"
    embedding_dimension: int = 768
    
    
    # Retrieval settings
    top_k_results: int = 5
    similarity_threshold: float = 0.87
    
    # Azure OpenAI settings
    AZURE_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_API_VERSION: str = os.getenv("API_VERSION")
    AZURE_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")
    llm_model: str = "gpt-4o"

    #Scraping settings 
    serper_api_key: str = "0b87708a5e0f5cce8eedc2a266eeec70"
    max_depth: int = 2
    delay: int = 1
    max_links_per_page: int = 10
    max_tokens_per_chunk: int = 200



# Create default config
config = Config()


    

    