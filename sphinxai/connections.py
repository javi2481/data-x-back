"""
Connection credentials functionality for the Sphinx library.

This module provides functions for retrieving connection credentials
for various integrations like Snowflake and Databricks.
"""

import os
from typing import Dict

from .config import PROVIDER_CONFIGS, ProviderConfig
from .adapters import SphinxAdapter


async def get_connection_credentials(integration_name: str, timeout: float = 5.0) -> Dict[str, str]:
    """
    Get connection credentials.

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
    # Get server configuration from environment
    sphinx_server = os.getenv("SPHINX_SERVER")
    token = os.getenv("SPHINX_LIBRARY_TOKEN")
    
    # Validate that required env vars are set
    if not sphinx_server:
        raise ValueError(
            "SPHINX_SERVER is required for connection credentials. "
            "Please set the SPHINX_SERVER environment variable."
        )
    if not token:
        raise ValueError(
            "SPHINX_LIBRARY_TOKEN is required for connection credentials. "
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
    
    # Use SphinxAdapter to get credentials
    adapter = SphinxAdapter(config, token)
    return await adapter.get_connection_credentials(integration_name, timeout)