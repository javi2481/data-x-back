"""
Validation functions for the Sphinx library.

This module contains all validation logic for models, environment variables,
and other configuration that the library uses.
"""

from .config import RuntimeConfig, ProviderConfig


def validate_chat_model(model_name: str, provider_config: ProviderConfig) -> None:
    """
    Validate that the chat model is supported for the given provider.
    
    Args:
        model_name: The model name to validate
        provider_config: The provider configuration
        
    Raises:
        ValueError: If the model is not supported
    """
    # Only validate models for sphinx provider (others are user's responsibility)
    if provider_config.name != "sphinx":
        return
    
    if model_name not in provider_config.supported_chat_models:
        supported_list = ", ".join(sorted(provider_config.supported_chat_models))
        raise ValueError(
            f"Unsupported chat completion model for {provider_config.name}: '{model_name}'. "
            f"Supported models are: {supported_list}. "
            f"Please check your model configuration."
        )


def validate_embedding_model(model_name: str, provider_config: ProviderConfig) -> None:
    """
    Validate that the embedding model is supported for the given provider.
    
    Args:
        model_name: The model name to validate
        provider_config: The provider configuration
        
    Raises:
        ValueError: If the model is not supported
    """
    # Check if provider supports embeddings at all
    if not provider_config.supports_embedding:
        raise ValueError(
            f"Provider '{provider_config.name}' does not support text embeddings. "
            f"Please use a different provider for embeddings (e.g., openai or google)."
        )
    
    # Only validate models for sphinx provider (others are user's responsibility)
    if provider_config.name != "sphinx":
        return
    
    if model_name not in provider_config.supported_embedding_models:
        supported_list = ", ".join(sorted(provider_config.supported_embedding_models))
        raise ValueError(
            f"Unsupported embedding model for {provider_config.name}: '{model_name}'. "
            f"Supported models are: {supported_list}. "
            f"Please check your model configuration."
        )


def validate_llm_config(runtime_config: RuntimeConfig) -> None:
    """
    Validate that the LLM configuration is complete.
    
    Args:
        runtime_config: The runtime configuration
        
    Raises:
        ValueError: If required configuration is missing
    """
    provider = runtime_config.llm_provider
    
    # For Sphinx provider, need SPHINX_SERVER and SPHINX_LIBRARY_TOKEN
    if provider == "sphinx":
        if not runtime_config.sphinx_server:
            raise ValueError(
                "SPHINX_SERVER is required when using sphinx provider. "
                "Please set it via environment variable or use a different provider."
            )
        if not runtime_config.sphinx_token:
            raise ValueError(
                "SPHINX_LIBRARY_TOKEN is required when using sphinx provider. "
                "Please set it via environment variable or use a different provider."
            )
    else:
        # For other providers, need API key
        if not runtime_config.llm_api_key:
            raise ValueError(
                f"API key is required when using {provider} provider. "
                f"Please provide it via set_llm_config() or SPHINX_LLM_API_KEY environment variable."
            )


def validate_embedding_config(runtime_config: RuntimeConfig) -> None:
    """
    Validate that the embedding configuration is complete.
    
    Args:
        runtime_config: The runtime configuration
        
    Raises:
        ValueError: If required configuration is missing
    """
    provider = runtime_config.embedding_provider
    
    # For Sphinx provider, need SPHINX_SERVER and SPHINX_LIBRARY_TOKEN
    if provider == "sphinx":
        if not runtime_config.sphinx_server:
            raise ValueError(
                "SPHINX_SERVER is required when using sphinx provider. "
                "Please set it via environment variable or use a different provider."
            )
        if not runtime_config.sphinx_token:
            raise ValueError(
                "SPHINX_LIBRARY_TOKEN is required when using sphinx provider. "
                "Please set it via environment variable or use a different provider."
            )
    else:
        # For other providers, need API key
        if not runtime_config.embedding_api_key:
            raise ValueError(
                f"API key is required when using {provider} provider. "
                f"Please provide it via set_embedding_config() or SPHINX_EMBEDDING_API_KEY environment variable."
            )


def validate_model_size(model_size: str, supported_sizes: set) -> None:
    """
    Validate that the model size is supported.
    
    Args:
        model_size: The model size to validate
        supported_sizes: Set of supported model sizes
        
    Raises:
        ValueError: If the model size is not supported
    """
    if model_size not in supported_sizes:
        supported_list = ", ".join(sorted(supported_sizes))
        raise ValueError(
            f"Unsupported model size: '{model_size}'. "
            f"Supported sizes are: {supported_list}."
        )
