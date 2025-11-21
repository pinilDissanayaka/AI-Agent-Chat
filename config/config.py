"""
Configuration file for AI Agent
Set your preferred model provider and model name here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Try to import streamlit for secrets support
try:
    import streamlit as st
    _has_streamlit = True
except ImportError:
    _has_streamlit = False

def _get_secret(key: str, default=None):
    """Get secret from Streamlit secrets or environment variables"""
    # First try Streamlit secrets
    if _has_streamlit:
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except (FileNotFoundError, KeyError):
            pass
    # Fall back to environment variables
    return os.getenv(key, default)

# Model Configuration
MODEL_PROVIDER = "openai"  # Options: "openai", "anthropic", "gemini", "local"
MODEL_NAME = "gpt-4o-mini"  # e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini-1.5-pro", etc.

# API Keys - supports both .env and Streamlit secrets
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
GEMINI_API_KEY = _get_secret("GEMINI_API_KEY")

# Ollama Configuration
OLLAMA_BASE_URL = _get_secret("OLLAMA_BASE_URL", "http://localhost:11434")  # Default Ollama URL

# Memory Configuration
MEMORY_FILE = os.path.join("data", "user_memory.json")

# Model Pricing (cost per 1M tokens)
MODEL_PRICING = {
    "openai": {
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet-20240229": {"input"  : 3.00, "output": 15.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    },
    "gemini": {
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.5-flash-lite": {"input": 0.015, "output": 0.06},
        "gemini-2.0-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.0-flash-lite": {"input": 0.015, "output": 0.06},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    },
    "local": {
        "llama3.3": {"input": 0.0, "output": 0.0},
        "llama3.2": {"input": 0.0, "output": 0.0},
        "llama3.1": {"input": 0.0, "output": 0.0},
        "llama2": {"input": 0.0, "output": 0.0},
        "mistral": {"input": 0.0, "output": 0.0},
        "mixtral": {"input": 0.0, "output": 0.0},
        "codellama": {"input": 0.0, "output": 0.0},
        "deepseek-coder": {"input": 0.0, "output": 0.0},
        "qwen2.5": {"input": 0.0, "output": 0.0},
        "phi4": {"input": 0.0, "output": 0.0},
        "gemma2": {"input": 0.0, "output": 0.0},
        "default": {"input": 0.0, "output": 0.0},
    }
}

def get_model_pricing(provider: str, model: str):
    """
    Retrieve the pricing information for the given model and provider.

    Args:
        provider (str): The model provider (e.g., "openai", "anthropic", "gemini", "local")
        model (str): The model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini-1.5-pro")

    Returns:
        dict: A dictionary with the pricing information for the given model and provider.
            The dictionary should contain the following keys:
                input (float): The cost per 1M input tokens
                output (float): The cost per 1M output tokens

    Raises:
        KeyError: If the given provider or model is not found in the pricing information.
    """
    provider_pricing = MODEL_PRICING.get(provider, {})
    model_pricing = provider_pricing.get(model, {"input": 0.0, "output": 0.0})
    return model_pricing

def calculate_cost(provider: str, model: str, input_tokens: int, output_tokens: int):
    """
    Calculate the total cost of a model call based on the provider, model, input tokens, and output tokens.

    Args:
        provider (str): The model provider (e.g., "openai", "anthropic", "gemini", "local")
        model (str): The model name (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20241022", "gemini-1.5-pro")
        input_tokens (int): The number of input tokens
        output_tokens (int): The number of output tokens

    Returns:
        float: The total cost of the model call in dollars
    """
    pricing = get_model_pricing(provider, model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return input_cost + output_cost
