"""
Text embedding functionality for the Sphinx library.

This module provides functions for generating text embeddings,
including single requests and batch processing.
"""

import asyncio
from typing import List

from .config import load_runtime_config, EMBEDDING_MODEL_SIZES
from .validation import validate_embedding_model, validate_embedding_config, validate_model_size
from .adapters import get_adapter


async def embed_text(text: str, model_size: str = "S", timeout: float = 30.0) -> List[float]:
    """
    Generate text embeddings using the specified embedding model size.
    
    Args:
        text: The text to embed
        model_size: Model size tier ("S" = small/fast, "L" = large/high-quality).
                   The actual model used depends on your configured provider.
                   Note: Only S and L are supported for embeddings.
        timeout: Timeout in seconds for the request (default: 30.0)
        
    Returns:
        The embedding vector as a list of floats
        
    Raises:
        ValueError: If environment variables are missing or model is invalid
    """
    # Load runtime configuration
    runtime_config = load_runtime_config()
    
    # Validate inputs
    validate_model_size(model_size, EMBEDDING_MODEL_SIZES)
    validate_embedding_config(runtime_config)
    
    # Get provider config and model
    provider_config = runtime_config.get_embedding_provider_config()
    model_name = runtime_config.get_embedding_model(model_size)
    
    # Validate the model
    validate_embedding_model(model_name, provider_config)
    
    # Create adapter and make request
    adapter = get_adapter(provider_config, runtime_config.get_embedding_api_key())
    return await adapter.embed_text(model_name, text, timeout)


async def batch_embed_text(
    texts: List[str], 
    model_size: str = "S", 
    max_concurrent: int = 5,
    timeout: float = 30.0
) -> List[List[float]]:
    """
    Process multiple text embeddings concurrently with semaphore-based rate limiting.
    
    Args:
        texts: List of text strings to embed
        model_size: Model size tier ("S" = small/fast, "L" = large/high-quality).
                   The actual model used depends on your configured provider.
                   Note: Only S and L are supported for embeddings.
        max_concurrent: Maximum number of concurrent requests (default: 5)
        timeout: Timeout in seconds for each individual request (default: 30.0)
    
    Returns:
        List of embedding vectors in the same order as input texts.
        Failed requests will return empty lists.
        
    Raises:
        ValueError: If environment variables are missing or model is invalid
    """
    # Validate inputs
    if not texts:
        return []
    
    # Load runtime configuration
    runtime_config = load_runtime_config()
    
    # Validate inputs
    validate_model_size(model_size, EMBEDDING_MODEL_SIZES)
    validate_embedding_config(runtime_config)
    
    # Get provider config and model
    provider_config = runtime_config.get_embedding_provider_config()
    model_name = runtime_config.get_embedding_model(model_size)
    
    # Validate the model
    validate_embedding_model(model_name, provider_config)
    
    # Create adapter
    adapter = get_adapter(provider_config, runtime_config.get_embedding_api_key())
    
    # Create semaphore to limit concurrent requests
    sem = asyncio.Semaphore(max_concurrent)
    
    async def process_single_text(text: str) -> List[float]:
        """Process a single text with semaphore control."""
        async with sem:
            try:
                return await adapter.embed_text(model_name, text, timeout)
            except Exception:
                # Return empty list on error
                return []
    
    # Process all texts concurrently
    tasks = [process_single_text(text) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert any remaining exceptions to empty lists
    return [
        [] if isinstance(result, Exception) else result
        for result in results
    ]
