"""
Secrets functionality for the Sphinx library.

This module provides functions for retrieving secret values
stored in the Sphinx secrets store.
"""

import os

from .config import PROVIDER_CONFIGS, ProviderConfig
from .adapters import SphinxAdapter


async def get_user_secret_value(
    secret_name: str,
    timeout: float = 5.0
) -> str:
    """
    Get a user secret value from the Sphinx secrets store.

    Args:
        secret_name: The name of the secret to retrieve
        timeout: Timeout in seconds for the request (default: 5.0)

    Returns:
        The secret value as a string.

    Example:
        import sphinxai

        # Get a user secret value
        api_key = await sphinxai.get_user_secret_value("MY_API_KEY")

        # Use the secret
        client = SomeAPIClient(api_key=api_key)
    """
    # Get server configuration from environment
    sphinx_server = os.getenv("SPHINX_SERVER")
    token = os.getenv("SPHINX_LIBRARY_TOKEN")

    # Validate that required env vars are set
    if not sphinx_server:
        raise ValueError(
            "SPHINX_SERVER is required for retrieving secrets. "
            "Please set the SPHINX_SERVER environment variable."
        )
    if not token:
        raise ValueError(
            "SPHINX_LIBRARY_TOKEN is required for retrieving secrets. "
            "Please set the SPHINX_LIBRARY_TOKEN environment variable."
        )

    # Create a Sphinx provider config with the server URL
    sphinx_config = PROVIDER_CONFIGS["sphinx"]
    config = ProviderConfig(
        name=sphinx_config.name,
        base_url=sphinx_server,
        default_models=sphinx_config.default_models,
        supported_chat_models=sphinx_config.supported_chat_models,
        supported_embedding_models=sphinx_config.supported_embedding_models,
        supports_chat=sphinx_config.supports_chat,
        supports_embedding=sphinx_config.supports_embedding,
        api_format=sphinx_config.api_format
    )

    # Use SphinxAdapter to get the secret value
    adapter = SphinxAdapter(config, token)
    return await adapter.get_secret_value(secret_name, "user", timeout)

