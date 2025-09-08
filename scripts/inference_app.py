from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import sys
sys.path.append('.')

from src.pipeline.inference_pipeline import InferencePipeline
from config.settings import config

# Setup logging
#logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RAG System API",
    description="RAG System with Web Scraping and Semantic Search",
    version="1.0.0"
)

inference_pipeline = None

@app.on_event("startup")
def startup_event():
    global inference_pipeline
    try:
        inference_pipeline = InferencePipeline(config)
        logger.info("Inference pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise

# Pydantic models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=5, ge=1, le=20)

class QueryResponse(BaseModel):
    query: str
    answer: str
    urls : List[str]
    retrieval_successful: bool


@app.get("/")
def root():
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "endpoints": ["/query"]
    }


@app.post("/query", response_model=QueryResponse)
def process_query(request: QueryRequest):
    if inference_pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = inference_pipeline.process_query(
            query=request.query
        )
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
