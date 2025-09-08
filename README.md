# RAG System - Salesforce Data Cloud Documentation

A Retrieval-Augmented Generation (RAG) system for Salesforce Data Cloud documentation using web scraping, hybrid search, and LLM-powered response generation.

## Prerequisites

- **Python 3.11** (required version)
- **Azure OpenAI** account with API access
- **Elasticsearch cluster** access (remote server configured)

## Installation & Setup

### 1. Environment Setup

Create and activate a Python 3.11 virtual environment:

```bash
# Create Python 3.11 virtual environment
python3.11 -m venv rag_env

# Activate environment
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```


```bash
git clone https://github.com/jayanth590/RAG_SYSTEM.git
cd RAG_SYSTEM
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

#### a) Configure Settings (`config/settings.py`)

Your settings file should look like this:

```python
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
    
    # Scraping settings
    serper_api_key: str = "0b87708a5e0f5cce8eedc2a266eeec70"
    max_depth: int = 0
    delay: int = 1
    max_links_per_page: int = 5
    max_tokens_per_chunk: int = 200

# Create default config
config = Config()
```

#### b) Environment Variables (`.env`)

Create a `.env` file in the project root with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_key_here
API_VERSION=your_azure_openai_version
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
```

## Usage

### Step 1: Data Insertion

Index Salesforce Data Cloud documentation into Elasticsearch:

```bash
python3 scripts/train.py
```

This script will:
- Scrape Salesforce help documentation from the configured source URL
- Apply smart semantic chunking (200 tokens per chunk based on your config)
- Generate embeddings using the instructor-large model
- Index documents with metadata into Elasticsearch cluster
- Store data in the `your_es_index` index

**Expected Output:**
```
Starting training with URLs
Scraping completed: X documents created
Successfully indexed X documents in Elasticsearch
Training completed successfully!
```

### Step 2: Start Inference API

Launch the HTTP API server:

```bash
python3 scripts/inference_app.py
```

The API will be available at: **http://localhost:8000**

**Available Endpoints:**
- `POST /query` - Process single query

### Step 3: Test the System

Run the test simulator:

```bash
bash simulator.sh
```

This script sends sample queries to the API and displays responses.

**Manual Testing Examples:**

```bash

# Single query
curl -s -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "How does Salesforce Data Cloud work?",
       "top_k": 5
     }'
```

## API Reference

### Single Query Request Format

```json
{
  "query": "your question here",
  "top_k": 5,
}
```


### Response Format

```json
{
  "query": "How does Salesforce Data Cloud work?",
  "answer": "Generated response based on retrieved context...",
  "urls" : ["list of urls"],
  "retrieval_successful": true,
}
```




## Architecture Overview

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   Smart Web Scraper │───▶│ Elasticsearch Cluster│───▶│  Hybrid Retrieval   │
│ (200 tokens/chunk)  │    │                            │ (RRF: BM25+Vector)  │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                                                     │
┌─────────────────────┐    ┌──────────────────────┐                 │
│   FastAPI Server    │◀───│  Azure OpenAI GPT-4o │◀────────────────┘
│  (localhost:8000)   │    │   LLM Generator      │
└─────────────────────┘    └──────────────────────┘
```

**Key Components:**

1. **Smart Web Scraper**: 
   - Extracts Salesforce documentation
   - 200-token semantic chunks with overlap
   - Deduplication and content filtering

2. **Elasticsearch Cluster**:
   - Stores documents with instructor-large embeddings (768 dimensions)
   - Index: `rag_documents2`

3. **Hybrid Retrieval**:
   - Combines BM25 (lexical) and vector (semantic) search
   - Reciprocal Rank Fusion (RRF) for result combination
   - Similarity threshold: 0.7

4. **Azure OpenAI Integration**:
   - GPT-4o model for response generation
   - Context-aware answer generation
   - Token usage tracking

5. **HTTP API**:
   - FastAPI framework
   - Single and batch query processing
   - Health monitoring and error handling

## System Configuration Details

**Current Settings:**
- **Chunk Size**: 200 tokens (optimized for focused content)
- **Embedding Model**: instructor-large (768 dimensions)
- **Retrieval**: Top 5 results with 0.87 similarity threshold
- **Scraping Depth**: 0 (single page, no link following)
- **Rate Limiting**: 1 second delay between requests




## Performance Notes

**Indexing Performance:**
- 200-token chunks create more documents but enable precise retrieval
- instructor-large provides high-quality embeddings but requires more compute
- Remote Elasticsearch may have network latency considerations

**Query Performance:**
- Hybrid search balances accuracy and speed
- Top-k=5 provides good context without overwhelming the LLM
- GPT-4o offers high-quality responses with reasonable speed

**Resource Usage:**
- Embedding model requires ~2GB RAM when loaded
- Each query processes up to 5 context documents  
- Token usage tracked for cost monitoring

## Next Steps

1. **Scale Testing**: Test with larger document collections
2. **Query Optimization**: Experiment with different top_k values
3. **Custom Prompting**: Develop domain-specific system prompts
4. **Monitoring**: Implement logging and metrics collection
5. **Security**: Add authentication if deploying to production


