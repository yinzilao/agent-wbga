# API Key Configuration

This guide walks you through obtaining and configuring API keys for major LLM providers. These keys will allow your agents to communicate with state-of-the-art language models.

## Overview of LLM API Providers

Here are the main LLM providers we'll set up:

1. **OpenAI**: Provider of GPT models (GPT-4, GPT-3.5)
2. **Anthropic**: Provider of Claude models (Claude 3 Opus, Sonnet, Haiku)
3. **Hugging Face**: Hub for open-source models

## 1. OpenAI API Setup

### Step 1: Create an OpenAI Account

1. Go to [OpenAI's website](https://platform.openai.com/signup)
2. Sign up for an account if you don't have one
3. Verify your email address

### Step 2: Get Your API Key

1. Log in to your OpenAI account
2. Navigate to [API keys](https://platform.openai.com/api-keys)
3. Click "Create new secret key"
4. Give your key a name (e.g., "Agent Development")
5. Copy your API key immediately (it won't be shown again)

### Step 3: Set Usage Limits (Optional but Recommended)

1. Go to [Usage limits](https://platform.openai.com/account/limits)
2. Set hard and soft limits to prevent unexpected charges

## 2. Anthropic API Setup

### Step 1: Create an Anthropic Account

1. Go to [Anthropic's website](https://www.anthropic.com/claude)
2. Click "Get API access" or navigate to [console.anthropic.com](https://console.anthropic.com/)
3. Sign up for an account
4. Verify your email address

### Step 2: Get Your API Key

1. Log in to the Anthropic Console
2. Navigate to the API Keys section
3. Create a new API key
4. Copy your API key

## 3. Hugging Face Setup

### Step 1: Create a Hugging Face Account

1. Go to [Hugging Face](https://huggingface.co/join)
2. Sign up for an account
3. Verify your email address

### Step 2: Get Your API Token

1. Log in to your Hugging Face account
2. Go to your profile settings
3. Click on "Access Tokens"
4. Create a new token with appropriate permissions
5. Copy your token

## Securely Storing API Keys

Now that you have your API keys, let's store them securely:

### Option 1: Using .env File (Development)

Create or update your `.env` file with your API keys:

```
# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Hugging Face
HUGGINGFACE_API_KEY=your_hugging_face_token_here
```

Remember to add `.env` to your `.gitignore` file to prevent committing these secrets to version control.

### Option 2: Using Environment Variables (Production)

For production environments, set environment variables directly:

```bash
# On Linux/macOS
export OPENAI_API_KEY=your_openai_api_key_here
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
export HUGGINGFACE_API_KEY=your_hugging_face_token_here

# On Windows
set OPENAI_API_KEY=your_openai_api_key_here
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
set HUGGINGFACE_API_KEY=your_hugging_face_token_here
```

### Option 3: Using a Secret Manager (Advanced)

For production applications, consider using cloud secret management services:

- AWS Secrets Manager
- Google Secret Manager
- Azure Key Vault
- HashiCorp Vault

## Creating API Client Configurations

Let's create a utility for loading and configuring API clients with the appropriate keys:

```python
# src/utils/api_clients.py

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
```

## Testing API Connectivity

Create a script to test your API connections:

```python
# test_api_connectivity.py

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from utils.api_clients import get_openai_client, get_anthropic_client, get_huggingface_client

def test_openai():
    """Test OpenAI API connectivity."""
    print("Testing OpenAI API connection...")
    client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, world!"}],
        )
        print(f"OpenAI API connection successful! Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"Error connecting to OpenAI API: {e}")
        return False

def test_anthropic():
    """Test Anthropic API connectivity."""
    print("Testing Anthropic API connection...")
    client = get_anthropic_client()
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hello, world!"}]
        )
        print(f"Anthropic API connection successful! Response: {response.content[0].text}")
        return True
    except Exception as e:
        print(f"Error connecting to Anthropic API: {e}")
        return False

def main():
    """Run API connection tests."""
    openai_success = test_openai()
    anthropic_success = test_anthropic()
    
    if openai_success and anthropic_success:
        print("\nAll API connections successful! ✅")
    else:
        print("\nSome API connections failed. Please check your configurations.")

if __name__ == "__main__":
    main()
```

## Best Practices for API Key Security

1. **Never commit API keys to version control**
2. **Rotate keys regularly** (every 30-90 days)
3. **Use different keys for development and production**
4. **Set up usage alerts** to detect unusual activity
5. **Use the principle of least privilege** when creating API keys
6. **Implement rate limiting** in your application

## API Cost Management

Be mindful of API costs when developing with LLMs:

1. **Monitor usage** regularly through provider dashboards
2. **Set hard spending limits** where possible
3. **Start with smaller, cheaper models** for development
4. **Cache responses** when appropriate
5. **Use streaming** for longer responses to reduce token usage

## Conclusion

You've now set up API keys for major LLM providers and created utility functions to securely access them in your agent applications. With these connections in place, you're ready to start building your first LLM agent.

In the next section, we'll create the basic framework for our first agent. 