"""
Sphinx AI Library

A Python library for interacting with the Sphinx AI platform,
providing easy-to-use functions for chat completion, text embedding,
connection credentials, and secrets management.

Model Size Tiers:
    - S (Small): Fast, cost-effective, suitable for simple tasks
    - M (Medium): Balanced performance and capability
    - L (Large): Highest quality for complex reasoning
    
    The library automatically maps these sizes to appropriate models
    based on your configured provider, keeping your code independent
    of specific model names.

Example:
    import sphinxai
    
    # Chat completion
    response = await sphinxai.llm("Hello, world!", model_size="S")
    
    # Text embedding
    embedding = await sphinxai.embed_text("Hello, world!", model_size="S")
    
    # Connection credentials
    credentials = await sphinxai.get_connection_credentials("snowflake")
    
    # Secrets
    api_key = await sphinxai.get_user_secret_value("MY_API_KEY")
    
    # Configure provider programmatically
    sphinxai.set_llm_config(
        provider="openai",
        api_key="sk-...",
        models={"S": "gpt-4.1-nano", "M": "gpt-4.1-mini", "L": "gpt-4.1"}
    )
"""

# Import main functions from specialized modules
from .chat import llm, batch_llm
from .embedding import embed_text, batch_embed_text
from .connections import get_connection_credentials
from .secrets import get_user_secret_value
from .config import (
    set_llm_config,
    set_embedding_config,
    get_llm_config,
    get_embedding_config,
    reset_llm_config,
    reset_embedding_config,
    reset_config
)

# Public API - includes all main functions
__all__ = [
    "llm",
    "batch_llm", 
    "embed_text",
    "batch_embed_text",
    "get_connection_credentials",
    "get_user_secret_value",
    "set_llm_config",
    "set_embedding_config",
    "get_llm_config",
    "get_embedding_config",
    "reset_llm_config",
    "reset_embedding_config",
    "reset_config",
]

# Version information
__version__ = "0.0.3"