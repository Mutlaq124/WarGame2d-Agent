import json
import requests
from typing import Optional

class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.1:8b-instruct-q4_K_S",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/chat"
        
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,  
            "options": {
                "temperature": temperature,
                # default to infinite below not specify max tokens
                "num_predict": max_tokens  # Max tokens to generate
            }
        }
        
        try:
            response = requests.post(
                self.generate_url,
                json=payload,
                timeout=60  # Ollama can be slow on CPU
            )
            response.raise_for_status()
            
            data = response.json()
            return data["message"]["content"]
            
        except requests.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after 60 seconds.\n"
                f"Model '{self.model}' might be too large for your hardware."
            )
        except requests.RequestException as e:
            raise RuntimeError(f"Ollama API error: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid Ollama response format: {e}")
    
