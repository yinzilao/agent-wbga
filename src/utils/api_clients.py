"""API client utilities for LLM providers."""

import os
from typing import Optional

from openai import OpenAI
from anthropic import Anthropic
import httpx

from .env import get_required_env_var, load_env_variables

# Load environment variables
load_env_variables()

def get_openai_client() -> OpenAI:
    """Get a configured OpenAI client."""
    api_key = get_required_env_var("OPENAI_API_KEY")
    
    # You can customize the client with additional parameters
    return OpenAI(
        api_key=api_key,
        # Optional: timeout settings, base URL, etc.
    )

def get_anthropic_client() -> Anthropic:
    """Get a configured Anthropic client."""
    api_key = get_required_env_var("ANTHROPIC_API_KEY")
    
    return Anthropic(
        api_key=api_key,
    )

def get_huggingface_client() -> httpx.Client:
    """Get a configured client for Hugging Face Inference API."""
    api_key = get_required_env_var("HUGGINGFACE_API_KEY")
    
    return httpx.Client(
        base_url="https://api-inference.huggingface.co/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    ) 