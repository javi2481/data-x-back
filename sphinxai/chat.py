"""
Chat completion functionality for the Sphinx library.

This module provides functions for interacting with chat completion models,
including single requests and batch processing.
"""

import asyncio
from typing import List

from .config import load_runtime_config, CHAT_MODEL_SIZES
from .validation import validate_chat_model, validate_llm_config, validate_model_size
from .adapters import get_adapter


async def llm(prompt: str, model_size: str = "S", timeout: float = 30.0) -> str:
    """
    Call an LLM with a given prompt and model size.
    
    Args:
        prompt: The user prompt / message
        model_size: Model size tier ("S" = small/fast, "M" = medium, "L" = large/capable).
                   The actual model used depends on your configured provider.
        timeout: Timeout in seconds for the request (default: 30.0)
        
    Returns:
        Response message content from the API
        
    Raises:
        ValueError: If environment variables are missing or model is invalid
    """
    # Load runtime configuration
    runtime_config = load_runtime_config()
    
    # Validate inputs
    validate_model_size(model_size, CHAT_MODEL_SIZES)
    validate_llm_config(runtime_config)
    
    # Get provider config and model
    provider_config = runtime_config.get_llm_provider_config()
    model_name = runtime_config.get_chat_model(model_size)
    
    # Validate the model
    validate_chat_model(model_name, provider_config)
    
    # Create adapter and make request
    adapter = get_adapter(provider_config, runtime_config.get_llm_api_key())
    return await adapter.chat_completion(model_name, prompt, timeout)


async def batch_llm(
    prompts: List[str], 
    model_size: str = "S", 
    max_concurrent: int = 5,
    timeout: float = 30.0
) -> List[str]:
    """
    Process multiple LLM prompts concurrently with semaphore-based rate limiting.
    
    Args:
        prompts: List of prompt strings to process
        model_size: Model size tier ("S" = small/fast, "M" = medium, "L" = large/capable).
                   The actual model used depends on your configured provider.
        max_concurrent: Maximum number of concurrent requests (default: 5)
        timeout: Timeout in seconds for each individual request (default: 30.0)
    
    Returns:
        List of responses in the same order as input prompts.
        Failed requests will return error messages.
        
    Raises:
        ValueError: If environment variables are missing or model is invalid
    """
    # Validate inputs
    if not prompts:
        return []
    
    # Load runtime configuration
    runtime_config = load_runtime_config()
    
    # Validate inputs
    validate_model_size(model_size, CHAT_MODEL_SIZES)
    validate_llm_config(runtime_config)
    
    # Get provider config and model
    provider_config = runtime_config.get_llm_provider_config()
    model_name = runtime_config.get_chat_model(model_size)
    
    # Validate the model
    validate_chat_model(model_name, provider_config)
    
    # Create adapter
    adapter = get_adapter(provider_config, runtime_config.get_llm_api_key())
    
    # Create semaphore to limit concurrent requests
    sem = asyncio.Semaphore(max_concurrent)
    
    async def process_single_prompt(prompt: str) -> str:
        """Process a single prompt with semaphore control."""
        async with sem:
            try:
                return await adapter.chat_completion(model_name, prompt, timeout)
            except Exception as e:
                return f"Error processing prompt: {str(e)}"
    
    # Process all prompts concurrently
    tasks = [process_single_prompt(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert any remaining exceptions to error strings
    final_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            final_results.append(f"Error processing prompt {i}: {str(result)}")
        else:
            final_results.append(result)
    
    return final_results
