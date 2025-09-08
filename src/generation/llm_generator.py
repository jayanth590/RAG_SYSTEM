import asyncio
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import os 
from .base_generator import BaseGenerator
from config.settings import config

logger = logging.getLogger(__name__)

class OpenAIGenerator(BaseGenerator):
    def __init__(self, model:str="gpt-4o"):
        self.azure_client = AzureOpenAI(
            api_key=config.AZURE_API_KEY,
            api_version=config.AZURE_API_VERSION,
            azure_endpoint=config.AZURE_ENDPOINT
        )
        self.model = model
        
    
    def generate(self, query: str, context_documents: List[Dict[str, Any]], 
                      system_prompt: str = None) -> Dict[str, Any]:
        try:
            context,urls = self._prepare_context(context_documents)
            system_message = system_prompt or self._get_default_system_prompt()
            
            user_message = f"""
            Query: {query}
            
            Context from relevant documents:
            {context}
            
            Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the query, please say so clearly.
            """
            
            # Generate response
            response = self.azure_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            logger.info(f"******************************************************")
            logger.info(f"\n\nThe answer Generated : {answer}\n\n")
            response_data = {
                'query': query,
                'answer': answer,
                'urls' : urls,
                'retrieval_successful' : True
            }
            logger.info(f"*****************************************************")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            response_data = {
                'query': query,
                'answer': answer,
                'urls' : [],
                'retrieval_successful' : False
            }
            return response_data
    
    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        urls = set()
        for i, doc in enumerate(documents, 1):
            content = doc.get('text', '').strip()
            url = doc.get("metadata", {}).get("url") or doc.get("url")
            urls.add(url)
            rrf_score = doc.get("rrf_score")
            
            context_part = f"""
            Document {i} (Relevance: {rrf_score:.3f}):
            Source: {url}
            Content: {content}
            """
            context_parts.append(context_part.strip())
        
        urls = list(urls)
        return ("\n\n".join(context_parts), urls)
    
    def _get_default_system_prompt(self) -> str:

        return """
        You are a helpful AI assistant that answers questions based on provided context documents. 

        Guidelines:
        1. Use only the information provided in the context documents to answer questions
        2. If the context doesn't contain sufficient information, clearly state this
        3. Provide specific, accurate, and helpful responses
        4. Include relevant details from the context when appropriate
        5. If multiple sources provide different information, acknowledge this
        6. Be concise but comprehensive in your responses
        """
    