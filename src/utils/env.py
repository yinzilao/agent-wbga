"""Environment variable utilities."""

import os
from pathlib import Path
from dotenv import load_dotenv

def load_env_variables():
    """Load environment variables from .env file."""
    # Find the .env file by looking in the current directory and parent directories
    path = Path.cwd()
    for _ in range(5):  # Look up to 5 directories up
        env_path = path / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            return True
        path = path.parent
    
    # If we get here, we didn't find a .env file, but that might be OK in production
    # where environment variables are set directly
    return False

def get_required_env_var(name: str) -> str:
    """Get an environment variable, raising an error if it's not set."""
    value = os.getenv(name)
    if value is None:
        raise ValueError(f"Environment variable {name} is required but not set")
    return value 