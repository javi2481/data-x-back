"""
API adapters for different LLM and embedding providers.

This module provides adapter classes that handle provider-specific API formats
and translate them to a common interface.
"""

import httpx
import json
import re
from typing import List, Dict, Optional
from .config import ProviderConfig


class BaseAdapter:
    """Base adapter class for API providers."""
    
    def __init__(self, provider_config: ProviderConfig, api_key: str):
        self.config = provider_config
        self.api_key = api_key
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        raise NotImplementedError
    
    async def chat_completion(
        self, 
        model: str, 
        prompt: str, 
        timeout: float = 30.0
    ) -> str:
        """Send a chat completion request."""
        raise NotImplementedError
    
    async def embed_text(
        self, 
        model: str, 
        text: str, 
        timeout: float = 30.0
    ) -> List[float]:
        """Send an embedding request."""
        raise NotImplementedError


class SphinxAdapter(BaseAdapter):
    """Adapter for Sphinx-hosted services."""
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "connect-protocol-version": "1",
            "x-sphinx-version": "1.0.0",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    async def chat_completion(
        self, 
        model: str, 
        prompt: str, 
        timeout: float = 30.0
    ) -> str:
        url = f"{self.config.base_url.rstrip('/')}/sphinx.SphinxRuntimeService/ChatCompletion"
        headers = self.get_headers()
        
        data = {
            "modelName": model,
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                if 'message' not in response_json:
                    raise ValueError(f"Unexpected response format: {response_json}")
                
                return response_json['message']
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_LIBRARY_TOKEN. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 403:
                raise ValueError(
                    "Access forbidden. Please check your API token permissions. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"HTTP error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request failed: {str(e)}")
    
    async def embed_text(
        self, 
        model: str, 
        text: str, 
        timeout: float = 30.0
    ) -> List[float]:
        url = f"{self.config.base_url.rstrip('/')}/sphinx.SphinxRuntimeService/EmbedText"
        headers = self.get_headers()
        
        data = {
            "text": text,
            "modelName": model,
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                if 'embedding' not in response_json:
                    raise ValueError(f"Unexpected response format: {response_json}")
                
                return response_json['embedding']
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_LIBRARY_TOKEN. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 403:
                raise ValueError(
                    "Access forbidden. Please check your API token permissions. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"HTTP error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request failed: {str(e)}")
    
    async def get_connection_credentials(
        self,
        integration_name: str,
        timeout: float = 5.0
    ) -> Dict[str, str]:
        """
        Get connection credentials for integrations like Snowflake and Databricks.
        
        Args:
            integration_name: The name of the integration to get credentials for
            timeout: Timeout in seconds for the request (default: 5.0)
            
        Returns:
            Dict[str, str]: Dictionary containing connection credentials with keys.
            
            For Snowflake, the keys are:
            - username: Snowflake username
            - account_identifier: Snowflake account identifier
            - access_token: Snowflake programmatic access token
            
            For Databricks, the keys are:
            - workspace_url: Databricks workspace URL
            - access_token: Databricks programmatic access token
        """
        url = f"{self.config.base_url.rstrip('/')}/sphinx.SphinxRuntimeService/GetIntegrationCredentials"
        headers = self.get_headers()
        
        data = {
            "integrationName": integration_name
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                credentials_json = response_json['credentialsJson']
                raw_credentials = json.loads(credentials_json)
                
                # Convert camelCase keys to snake_case, with special handling for token
                def camel_to_snake(name: str) -> str:
                    # Insert underscores before uppercase letters and convert to lowercase
                    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
                    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
                
                result = {}
                for key, value in raw_credentials.items():
                    if key == "token":
                        # Special case: token becomes access_token
                        result["access_token"] = value
                    else:
                        # Convert camelCase to snake_case for all other keys
                        snake_key = camel_to_snake(key)
                        result[snake_key] = value
                
                return result
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_LIBRARY_TOKEN. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 403:
                raise ValueError(
                    "Access forbidden. Please check your API token permissions. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"HTTP error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request failed: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected response format: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing credentials JSON: {e}")

    async def get_secret_value(
        self,
        secret_name: str,
        secret_type: str = "user",
        timeout: float = 5.0
    ) -> str:
        """
        Get a secret value from the Sphinx secrets store.
        
        Args:
            secret_name: The name of the secret to retrieve
            secret_type: The type of secret ('user'). Defaults to 'user'.
            timeout: Timeout in seconds for the request (default: 5.0)
            
        Returns:
            The secret value as a string.
        """
        url = f"{self.config.base_url.rstrip('/')}/sphinx.SphinxRuntimeService/GetSecretValue"
        headers = self.get_headers()
        
        # Convert string type to protobuf enum value
        type_map = {
            "user": 1,  # SECRET_TYPE_USER
        }
        
        secret_type_lower = secret_type.lower()
        if secret_type_lower not in type_map:
            raise ValueError(
                f"Invalid secret type: '{secret_type}'. "
                f"Valid types are: {', '.join(type_map.keys())}"
            )
        
        data = {
            "name": secret_name,
            "type": type_map[secret_type_lower]
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                return response_json.get("value", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_LIBRARY_TOKEN. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 403:
                raise ValueError(
                    "Access forbidden. Please check your API token permissions. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 404:
                raise ValueError(
                    f"Secret '{secret_name}' not found. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"HTTP error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request failed: {str(e)}")


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API."""
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
    
    async def chat_completion(
        self, 
        model: str, 
        prompt: str, 
        timeout: float = 30.0
    ) -> str:
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"
        headers = self.get_headers()
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                return response_json['choices'][0]['message']['content']
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_LLM_API_KEY for OpenAI. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 403:
                raise ValueError(
                    "Access forbidden. Please check your OpenAI API key permissions. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"OpenAI API error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request to OpenAI failed: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected response format from OpenAI: {e}")
    
    async def embed_text(
        self, 
        model: str, 
        text: str, 
        timeout: float = 30.0
    ) -> List[float]:
        url = f"{self.config.base_url.rstrip('/')}/embeddings"
        headers = self.get_headers()
        
        data = {
            "model": model,
            "input": text,
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                return response_json['data'][0]['embedding']
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_EMBEDDING_API_KEY for OpenAI. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 403:
                raise ValueError(
                    "Access forbidden. Please check your OpenAI API key permissions. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"OpenAI API error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request to OpenAI failed: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected response format from OpenAI: {e}")


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude API."""
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
    
    async def chat_completion(
        self, 
        model: str, 
        prompt: str, 
        timeout: float = 30.0
    ) -> str:
        url = f"{self.config.base_url.rstrip('/')}/messages"
        headers = self.get_headers()
        
        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4096,
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                return response_json['content'][0]['text']
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_LLM_API_KEY for Anthropic. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            elif e.response.status_code == 403:
                raise ValueError(
                    "Access forbidden. Please check your Anthropic API key permissions. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"Anthropic API error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request to Anthropic failed: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected response format from Anthropic: {e}")
    
    async def embed_text(
        self, 
        model: str, 
        text: str, 
        timeout: float = 30.0
    ) -> List[float]:
        raise ValueError(
            "Anthropic does not support text embeddings. "
            "Please use a different provider for embeddings (e.g., set SPHINX_EMBEDDING_PROVIDER=openai)."
        )


class GoogleAdapter(BaseAdapter):
    """Adapter for Google Gemini API."""
    
    def get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
    
    async def chat_completion(
        self, 
        model: str, 
        prompt: str, 
        timeout: float = 30.0
    ) -> str:
        url = f"{self.config.base_url.rstrip('/')}/models/{model}:generateContent"
        headers = self.get_headers()
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                # Check if candidates exist
                if 'candidates' not in response_json or not response_json['candidates']:
                    # Check for safety blocks or other filters
                    if 'promptFeedback' in response_json:
                        block_reason = response_json['promptFeedback'].get('blockReason', 'UNKNOWN')
                        raise ValueError(
                            f"Google Gemini blocked the request. Reason: {block_reason}. "
                            "Try rephrasing your prompt or adjusting safety settings."
                        )
                    raise ValueError(
                        f"Google Gemini returned no candidates. Response: {response_json}"
                    )
                
                # Extract the text from the first candidate
                candidate = response_json['candidates'][0]
                if 'content' not in candidate or 'parts' not in candidate['content']:
                    raise ValueError(f"Unexpected candidate structure from Google: {candidate}")
                
                return candidate['content']['parts'][0]['text']
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 or e.response.status_code == 403:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_LLM_API_KEY for Google. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"Google API error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request to Google failed: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected response format from Google: {e}")
    
    async def embed_text(
        self, 
        model: str, 
        text: str, 
        timeout: float = 30.0
    ) -> List[float]:
        url = f"{self.config.base_url.rstrip('/')}/models/{model}:embedContent"
        headers = self.get_headers()
        
        data = {
            "model": f"models/{model}",
            "content": {
                "parts": [
                    {"text": text}
                ]
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                response_json = response.json()
                
                # Check if embedding exists
                if 'embedding' not in response_json:
                    raise ValueError(
                        f"Google Gemini returned no embedding. Response: {response_json}"
                    )
                
                if 'values' not in response_json['embedding']:
                    raise ValueError(
                        f"Google Gemini embedding missing values. Response: {response_json}"
                    )
                
                return response_json['embedding']['values']
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 or e.response.status_code == 403:
                raise ValueError(
                    "Authentication failed. Please check your SPHINX_EMBEDDING_API_KEY for Google. "
                    f"Status: {e.response.status_code}, Response: {e.response.text}"
                )
            else:
                raise ValueError(f"Google API error: {e.response.status_code}, Response: {e.response.text}")
        except httpx.RequestError as e:
            raise ValueError(f"Request to Google failed: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Unexpected response format from Google: {e}")


def get_adapter(provider_config: ProviderConfig, api_key: str) -> BaseAdapter:
    """
    Get the appropriate adapter for a provider based on its API format.
    
    This allows flexibility - e.g., you can use provider="openai" with a custom
    base_url that supports OpenAI-compatible endpoints.
    """
    adapter_map = {
        "sphinx": SphinxAdapter,
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "google": GoogleAdapter,
    }
    
    # Select adapter based on api_format (not provider name)
    # This allows cross-provider compatibility (e.g., OpenAI format with custom URL)
    adapter_class = adapter_map.get(provider_config.api_format, OpenAIAdapter)
    return adapter_class(provider_config, api_key)
