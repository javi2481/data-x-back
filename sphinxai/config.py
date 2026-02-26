"""
Configuration system for the Sphinx library.

This module provides provider-specific configurations and runtime settings
for LLM and embedding services.
"""

import os
from dataclasses import dataclass
from typing import Dict, Set, Optional, Any


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    name: str
    base_url: str
    default_models: Dict[str, Dict[str, str]]  # {"chat": {"S": "model", ...}, "embedding": {...}}
    supported_chat_models: Set[str]
    supported_embedding_models: Set[str]
    supports_chat: bool = True
    supports_embedding: bool = True
    api_format: str = "openai"  # "openai", "anthropic", "google", "sphinx"


# Model size constants
# These abstract size tiers (S/M/L) allow code to be independent of specific model names,
# enabling seamless model upgrades without code changes. Particularly useful for agent-driven code.
CHAT_MODEL_SIZES = {"S", "M", "L"}
EMBEDDING_MODEL_SIZES = {"S", "L"}


# Provider configurations
PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "sphinx": ProviderConfig(
        name="sphinx",
        base_url="",  # Will be set from SPHINX_SERVER
        default_models={
            "chat": {
                "S": "gpt-4.1-nano",
                "M": "gpt-4.1-mini",
                "L": "gpt-4.1"
            },
            "embedding": {
                "S": "text-embedding-3-small",
                "L": "text-embedding-3-large"
            }
        },
        supported_chat_models={
            "gpt-4.1-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5",
            "claude-sonnet-4",
            "gemini-2.5-flash"
        },
        supported_embedding_models={
            "text-embedding-3-small",
            "text-embedding-3-large"
        },
        api_format="sphinx"
    ),
    "openai": ProviderConfig(
        name="openai",
        base_url="https://api.openai.com/v1",
        default_models={
            "chat": {
                "S": "gpt-4.1-nano",
                "M": "gpt-4.1-mini",
                "L": "gpt-4.1"
            },
            "embedding": {
                "S": "text-embedding-3-small",
                "L": "text-embedding-3-large"
            }
        },
        supported_chat_models=set(),  # User is responsible for providing correct model names
        supported_embedding_models=set(),  # User is responsible for providing correct model names
        api_format="openai"
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        base_url="https://api.anthropic.com/v1",
        default_models={
            "chat": {
                "S": "claude-haiku-4.5",
                "M": "claude-haiku-4.5",
                "L": "claude-sonnet-4.5"
            },
            "embedding": {}  # Anthropic doesn't support embeddings
        },
        supported_chat_models=set(),  # User is responsible for providing correct model names
        supported_embedding_models=set(),
        supports_embedding=False,
        api_format="anthropic"
    ),
    "google": ProviderConfig(
        name="google",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_models={
            "chat": {
                "S": "gemini-2.5-flash-lite",
                "M": "gemini-2.5-flash",
                "L": "gemini-2.5-pro"
            },
            "embedding": {
                "S": "gemini-embedding-001",
                "L": "gemini-embedding-001"
            }
        },
        supported_chat_models=set(),  # User is responsible for providing correct model names
        supported_embedding_models=set(),  # User is responsible for providing correct model names
        api_format="google"
    ),
}


@dataclass
class RuntimeConfig:
    """Runtime configuration for LLM and embedding services."""
    # LLM configuration
    llm_provider: str
    llm_api_key: Optional[str]
    llm_base_url: Optional[str]
    
    # Embedding configuration
    embedding_provider: str
    embedding_api_key: Optional[str]
    embedding_base_url: Optional[str]
    
    # Connection configuration (unchanged)
    sphinx_server: Optional[str]
    sphinx_token: Optional[str]
    
    # Model overrides
    chat_model_overrides: Dict[str, str]
    embedding_model_overrides: Dict[str, str]
    
    def _get_provider_config(
        self, 
        provider_name: str, 
        base_url_override: Optional[str], 
        service_type: str
    ) -> ProviderConfig:
        """Helper to get provider config with proper overrides.
        
        Args:
            provider_name: Name of the provider
            base_url_override: Optional base URL override
            service_type: "LLM" or "embedding" for error messages
        """
        if provider_name not in PROVIDER_CONFIGS:
            raise ValueError(
                f"Unknown {service_type} provider: '{provider_name}'. "
                f"Supported providers: {', '.join(PROVIDER_CONFIGS.keys())}"
            )
        config = PROVIDER_CONFIGS[provider_name]
        
        # Determine the actual base URL to use
        actual_base_url = None
        if base_url_override:
            actual_base_url = base_url_override
        elif config.name == "sphinx" and self.sphinx_server:
            actual_base_url = self.sphinx_server
        
        # If we need to override the base URL, create a new config
        if actual_base_url:
            config = ProviderConfig(
                name=config.name,
                base_url=actual_base_url,
                default_models=config.default_models,
                supported_chat_models=config.supported_chat_models,
                supported_embedding_models=config.supported_embedding_models,
                supports_chat=config.supports_chat,
                supports_embedding=config.supports_embedding,
                api_format=config.api_format
            )
        
        return config
    
    def get_llm_provider_config(self) -> ProviderConfig:
        """Get the provider config for LLM."""
        return self._get_provider_config(self.llm_provider, self.llm_base_url, "LLM")
    
    def get_embedding_provider_config(self) -> ProviderConfig:
        """Get the provider config for embeddings."""
        return self._get_provider_config(self.embedding_provider, self.embedding_base_url, "embedding")
    
    def get_chat_model(self, size: str) -> str:
        """Get the chat model for a given size, respecting overrides."""
        if size in self.chat_model_overrides:
            return self.chat_model_overrides[size]
        
        provider_config = self.get_llm_provider_config()
        return provider_config.default_models["chat"].get(size, provider_config.default_models["chat"]["L"])
    
    def get_embedding_model(self, size: str) -> str:
        """Get the embedding model for a given size, respecting overrides."""
        if size in self.embedding_model_overrides:
            return self.embedding_model_overrides[size]
        
        provider_config = self.get_embedding_provider_config()
        embedding_models = provider_config.default_models.get("embedding", {})
        
        # Check if provider has any embedding models
        if not embedding_models:
            raise ValueError(
                f"Provider '{provider_config.name}' does not support text embeddings. "
                f"Please use a different provider (openai, google, or sphinx)."
            )
        
        # Get the model for the specified size, fallback to L if not found
        if size not in embedding_models:
            if "L" in embedding_models:
                return embedding_models["L"]
            # If L doesn't exist either, just return any available model
            return next(iter(embedding_models.values()))
        
        return embedding_models[size]
    
    def get_llm_api_key(self) -> Optional[str]:
        """Get the appropriate API key for LLM requests."""
        return self.sphinx_token if self.llm_provider == "sphinx" else self.llm_api_key
    
    def get_embedding_api_key(self) -> Optional[str]:
        """Get the appropriate API key for embedding requests."""
        return self.sphinx_token if self.embedding_provider == "sphinx" else self.embedding_api_key


# Global runtime config override (for programmatic configuration)
_runtime_config_override: Optional[RuntimeConfig] = None


def _load_from_env() -> RuntimeConfig:
    """
    Internal helper to load configuration from environment variables only.
    Ignores any programmatic overrides.
    """
    # LLM configuration
    llm_provider = os.getenv("SPHINX_LLM_PROVIDER", "sphinx")
    llm_api_key = os.getenv("SPHINX_LLM_API_KEY")
    llm_base_url = os.getenv("SPHINX_LLM_BASE_URL")
    
    # Embedding configuration
    embedding_provider = os.getenv("SPHINX_EMBEDDING_PROVIDER", "sphinx")
    embedding_api_key = os.getenv("SPHINX_EMBEDDING_API_KEY")
    embedding_base_url = os.getenv("SPHINX_EMBEDDING_BASE_URL")
    
    # Connection configuration (for Sphinx-hosted services and connections)
    sphinx_server = os.getenv("SPHINX_SERVER")
    sphinx_token = os.getenv("SPHINX_LIBRARY_TOKEN")
    
    # Model overrides
    chat_model_overrides = {}
    if small_model := os.getenv("SPHINX_LIBRARY_SMALL_MODEL"):
        chat_model_overrides["S"] = small_model
    if medium_model := os.getenv("SPHINX_LIBRARY_MEDIUM_MODEL"):
        chat_model_overrides["M"] = medium_model
    if large_model := os.getenv("SPHINX_LIBRARY_LARGE_MODEL"):
        chat_model_overrides["L"] = large_model
    
    embedding_model_overrides = {}
    if small_emb := os.getenv("SPHINX_LIBRARY_EMBEDDING_SMALL_MODEL"):
        embedding_model_overrides["S"] = small_emb
    if large_emb := os.getenv("SPHINX_LIBRARY_EMBEDDING_LARGE_MODEL"):
        embedding_model_overrides["L"] = large_emb
    
    return RuntimeConfig(
        llm_provider=llm_provider,
        llm_api_key=llm_api_key,
        llm_base_url=llm_base_url,
        embedding_provider=embedding_provider,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        sphinx_server=sphinx_server,
        sphinx_token=sphinx_token,
        chat_model_overrides=chat_model_overrides,
        embedding_model_overrides=embedding_model_overrides,
    )


def load_runtime_config() -> RuntimeConfig:
    """Load runtime configuration from environment variables or programmatic override."""
    # If there's a programmatic override, use it
    if _runtime_config_override is not None:
        return _runtime_config_override
    
    # Otherwise load from environment variables
    return _load_from_env()


def set_llm_config(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    models: Optional[Dict[str, str]] = None
) -> None:
    """
    Set the LLM configuration programmatically.
    
    This function allows you to set up LLM configuration in code rather than
    using environment variables. This is useful for dynamic configuration or
    when you want to manage credentials programmatically.
    
    Args:
        provider: Provider name ("openai", "anthropic", "google")
        api_key: API key for the provider
        base_url: Optional custom base URL for the provider
        models: Optional dict mapping sizes to model names, e.g. {"S": "gpt-4.1-nano", "M": "gpt-4.1-mini", "L": "gpt-4.1"}.
               You can selectively override any size; unspecified sizes will use the provider's defaults.
        
    Example:
        import sphinxai
        
        # Configure OpenAI as LLM provider
        sphinxai.set_llm_config(
            provider="openai",
            api_key="sk-...",
            models={"S": "gpt-4.1-nano", "M": "gpt-4.1-mini", "L": "gpt-4.1"}
        )
        
        # Or use OpenAI format with a custom endpoint
        sphinxai.set_llm_config(
            provider="openai",
            api_key="your-key",
            base_url="https://your-openai-compatible-api.com/v1",
            models={"S": "your-small-model", "M": "your-medium-model", "L": "your-large-model"}
        )
    """
    global _runtime_config_override
    
    # Validate provider exists
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Supported providers: {', '.join(PROVIDER_CONFIGS.keys())}"
        )
    
    # Validate provider supports chat (all current providers do, but future-proofing)
    provider_config = PROVIDER_CONFIGS[provider]
    if not provider_config.supports_chat:
        raise ValueError(
            f"Provider '{provider}' does not support chat completions."
        )
    
    # Load current config (from env vars if no override exists)
    current_config = load_runtime_config()
    
    # Create new config with LLM settings updated
    _runtime_config_override = RuntimeConfig(
        llm_provider=provider,
        llm_api_key=api_key,
        llm_base_url=base_url,
        embedding_provider=current_config.embedding_provider,
        embedding_api_key=current_config.embedding_api_key,
        embedding_base_url=current_config.embedding_base_url,
        sphinx_server=current_config.sphinx_server,
        sphinx_token=current_config.sphinx_token,
        chat_model_overrides=models or {},
        embedding_model_overrides=current_config.embedding_model_overrides,
    )


def set_embedding_config(
    provider: str,
    api_key: str,
    base_url: Optional[str] = None,
    models: Optional[Dict[str, str]] = None
) -> None:
    """
    Set the embedding configuration programmatically.
    
    This function allows you to set up embedding configuration in code rather than
    using environment variables. This is useful for dynamic configuration or
    when you want to manage credentials programmatically.
    
    Args:
        provider: Provider name ("openai", "google")
        api_key: API key for the provider
        base_url: Optional custom base URL for the provider
        models: Optional dict mapping sizes to model names, e.g. {"S": "text-embedding-3-small", "L": "text-embedding-3-large"}.
               You can selectively override any size; unspecified sizes will use the provider's defaults.
        
    Example:
        import sphinxai
        
        # Configure OpenAI as embedding provider
        sphinxai.set_embedding_config(
            provider="openai",
            api_key="sk-...",
            models={"S": "text-embedding-3-small", "L": "text-embedding-3-large"}
        )
    """
    global _runtime_config_override
    
    # Validate provider exists
    if provider not in PROVIDER_CONFIGS:
        raise ValueError(
            f"Unknown provider: '{provider}'. "
            f"Supported providers: {', '.join(PROVIDER_CONFIGS.keys())}"
        )
    
    # Validate provider supports embeddings
    provider_config = PROVIDER_CONFIGS[provider]
    if not provider_config.supports_embedding:
        raise ValueError(
            f"Provider '{provider}' does not support text embeddings. "
            f"Supported embedding providers: openai, google, sphinx"
        )
    
    # Load current config (from env vars if no override exists)
    current_config = load_runtime_config()
    
    # Create new config with embedding settings updated
    _runtime_config_override = RuntimeConfig(
        llm_provider=current_config.llm_provider,
        llm_api_key=current_config.llm_api_key,
        llm_base_url=current_config.llm_base_url,
        embedding_provider=provider,
        embedding_api_key=api_key,
        embedding_base_url=base_url,
        sphinx_server=current_config.sphinx_server,
        sphinx_token=current_config.sphinx_token,
        chat_model_overrides=current_config.chat_model_overrides,
        embedding_model_overrides=models or {},
    )


def get_llm_config() -> Dict[str, Any]:
    """
    Get the current LLM configuration.
    
    Returns a dictionary showing the current LLM provider, base URL, model mappings,
    and whether an API key is configured (without exposing the actual key).
    
    Returns:
        Dict with LLM configuration details
        
    Example:
        import sphinxai
        
        config = sphinxai.get_llm_config()
        print(config)
        # {
        #     "provider": "openai",
        #     "base_url": "https://api.openai.com/v1",
        #     "models": {"S": "gpt-4.1-nano", "M": "gpt-4.1-mini", "L": "gpt-4.1"},
        #     "has_api_key": True,
        #     "config_source": "programmatic"  # or "environment"
        # }
    """
    runtime_config = load_runtime_config()
    
    # Get LLM provider config
    llm_provider_config = runtime_config.get_llm_provider_config()
    llm_models = {
        "S": runtime_config.get_chat_model("S"),
        "M": runtime_config.get_chat_model("M"),
        "L": runtime_config.get_chat_model("L"),
    }
    
    # Determine which API key to check for LLM
    if runtime_config.llm_provider == "sphinx":
        llm_has_key = runtime_config.sphinx_token is not None
    else:
        llm_has_key = runtime_config.llm_api_key is not None
    
    return {
        "provider": runtime_config.llm_provider,
        "base_url": llm_provider_config.base_url,
        "models": llm_models,
        "has_api_key": llm_has_key,
        "config_source": "programmatic" if _runtime_config_override is not None else "environment",
    }


def get_embedding_config() -> Dict[str, Any]:
    """
    Get the current embedding configuration.
    
    Returns a dictionary showing the current embedding provider, base URL, model mappings,
    whether an API key is configured, and whether the provider supports embeddings.
    
    Returns:
        Dict with embedding configuration details
        
    Example:
        import sphinxai
        
        config = sphinxai.get_embedding_config()
        print(config)
        # {
        #     "provider": "openai",
        #     "base_url": "https://api.openai.com/v1",
        #     "models": {"S": "text-embedding-3-small", "L": "text-embedding-3-large"},
        #     "has_api_key": True,
        #     "supports_embedding": True,
        #     "config_source": "programmatic"  # or "environment"
        # }
    """
    runtime_config = load_runtime_config()
    
    # Get embedding provider config
    embedding_provider_config = runtime_config.get_embedding_provider_config()
    embedding_models = {
        "S": runtime_config.get_embedding_model("S"),
        "L": runtime_config.get_embedding_model("L"),
    }
    
    # Determine which API key to check for embeddings
    if runtime_config.embedding_provider == "sphinx":
        embedding_has_key = runtime_config.sphinx_token is not None
    else:
        embedding_has_key = runtime_config.embedding_api_key is not None
    
    return {
        "provider": runtime_config.embedding_provider,
        "base_url": embedding_provider_config.base_url,
        "models": embedding_models,
        "has_api_key": embedding_has_key,
        "supports_embedding": embedding_provider_config.supports_embedding,
        "config_source": "programmatic" if _runtime_config_override is not None else "environment",
    }


def reset_llm_config() -> None:
    """
    Reset only LLM configuration to environment variable-based config.
    Keeps embedding configuration unchanged.
    
    Example:
        import sphinxai
        
        # Configure both
        sphinxai.set_llm_config("openai", "sk-...")
        sphinxai.set_embedding_config("google", "...")
        
        # Reset only LLM, keep embedding config
        sphinxai.reset_llm_config()
    """
    global _runtime_config_override
    
    if _runtime_config_override is None:
        return  # Nothing to reset
    
    # Load env vars for LLM config (bypassing override)
    env_config = _load_from_env()
    
    # Create new config with LLM from env, embedding unchanged
    _runtime_config_override = RuntimeConfig(
        llm_provider=env_config.llm_provider,
        llm_api_key=env_config.llm_api_key,
        llm_base_url=env_config.llm_base_url,
        embedding_provider=_runtime_config_override.embedding_provider,
        embedding_api_key=_runtime_config_override.embedding_api_key,
        embedding_base_url=_runtime_config_override.embedding_base_url,
        sphinx_server=_runtime_config_override.sphinx_server,
        sphinx_token=_runtime_config_override.sphinx_token,
        chat_model_overrides=env_config.chat_model_overrides,
        embedding_model_overrides=_runtime_config_override.embedding_model_overrides,
    )


def reset_embedding_config() -> None:
    """
    Reset only embedding configuration to environment variable-based config.
    Keeps LLM configuration unchanged.
    
    Example:
        import sphinxai
        
        # Configure both
        sphinxai.set_llm_config("openai", "sk-...")
        sphinxai.set_embedding_config("google", "...")
        
        # Reset only embedding, keep LLM config
        sphinxai.reset_embedding_config()
    """
    global _runtime_config_override
    
    if _runtime_config_override is None:
        return  # Nothing to reset
    
    # Load env vars for embedding config (bypassing override)
    env_config = _load_from_env()
    
    # Create new config with embedding from env, LLM unchanged
    _runtime_config_override = RuntimeConfig(
        llm_provider=_runtime_config_override.llm_provider,
        llm_api_key=_runtime_config_override.llm_api_key,
        llm_base_url=_runtime_config_override.llm_base_url,
        embedding_provider=env_config.embedding_provider,
        embedding_api_key=env_config.embedding_api_key,
        embedding_base_url=env_config.embedding_base_url,
        sphinx_server=_runtime_config_override.sphinx_server,
        sphinx_token=_runtime_config_override.sphinx_token,
        chat_model_overrides=_runtime_config_override.chat_model_overrides,
        embedding_model_overrides=env_config.embedding_model_overrides,
    )


def reset_config() -> None:
    """
    Reset all programmatic configuration and return to environment variable-based config.
    This resets both LLM and embedding configurations.
    
    Example:
        import sphinxai
        
        # Configure programmatically
        sphinxai.set_llm_config("openai", "sk-...")
        
        # Later, reset to use environment variables
        sphinxai.reset_config()
    """
    global _runtime_config_override
    _runtime_config_override = None
