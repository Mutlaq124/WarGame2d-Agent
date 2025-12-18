"""
Groq API client for LLM inference.
"""

import os
from typing import Optional
import requests
import logging

logger = logging.getLogger(__name__)


class GroqClient:
    """Client for Groq API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not provided and not found in environment")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        logger.info(f"Initialized GroqClient with model: {self.model}")
        
    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
        """Call Groq API with system and user prompts."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            logger.debug(f"Groq API call successful (tokens: {response.json().get('usage', {}).get('total_tokens', 'N/A')})")
            return content
            
        except requests.exceptions.Timeout:
            logger.error("Groq API request timed out")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"Groq API HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except KeyError as e:
            logger.error(f"Unexpected Groq API response format: {e}")
            raise ValueError(f"Invalid response format from Groq API: {e}")