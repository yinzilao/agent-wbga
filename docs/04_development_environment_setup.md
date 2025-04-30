# Development Environment Setup

This guide will walk you through setting up a complete development environment for building LLM agents. We'll use Python as our primary language, as it has excellent support for LLM APIs and most related tools.

## Prerequisites

Before starting, make sure you have:

- Python 3.10+ installed
- A code editor (like VSCode, PyCharm, etc.)
- Basic knowledge of terminal/command line operations
- Internet access to download packages and access LLM APIs

## Step 1: Create a Virtual Environment

Virtual environments help keep your dependencies organized and project-specific.

```bash
# Create a new directory for your agent projects
mkdir llm-agent-projects
cd llm-agent-projects

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

## Step 2: Install Core Dependencies

Let's install the essential packages needed for our agent projects.

```bash
pip install -U pip  # Upgrade pip first

# Core dependencies
pip install -U openai anthropic httpx pydantic python-dotenv fastapi uvicorn

# Utility libraries
pip install -U numpy pandas matplotlib pyyaml rich typer

# Create requirements.txt
pip freeze > requirements.txt
```

### Understanding the Key Packages

- **openai**: Official Python client for OpenAI API
- **anthropic**: Official Python client for Anthropic's Claude models
- **httpx**: Modern, async-ready HTTP client
- **pydantic**: Data validation and settings management
- **python-dotenv**: Loads environment variables from .env files
- **fastapi & uvicorn**: For creating API endpoints around your agents

## Step 3: Set Up Your Project Structure

Create a standard project structure to keep your code organized:

```bash
mkdir -p src/agents src/tools src/memory src/utils examples tests
touch src/__init__.py src/agents/__init__.py src/tools/__init__.py src/memory/__init__.py src/utils/__init__.py
```

Your project structure should look like this:

```
llm-agent-projects/
├── venv/
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   └── __init__.py
│   ├── tools/
│   │   └── __init__.py
│   ├── memory/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── examples/
├── tests/
└── requirements.txt
```

## Step 4: Configure Environment Variables

Create a `.env` file to securely store your API keys:

```bash
touch .env
```

Add your API keys to the `.env` file:

```
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Create a utility script to load environment variables:

```bash
cat > src/utils/env.py << 'EOL'
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
EOL
```

## Step 5: Test Your Environment

Create a simple test script to verify that everything is working:

```bash
cat > test_environment.py << 'EOL'
"""Test the development environment setup."""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from utils.env import load_env_variables, get_required_env_var

def main():
    """Test the environment setup."""
    print("Testing development environment setup...")
    
    # Test environment variables
    loaded = load_env_variables()
    print(f"Environment variables loaded: {loaded}")
    
    try:
        openai_key = get_required_env_var("OPENAI_API_KEY")
        print("OpenAI API key found ✓")
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        anthropic_key = get_required_env_var("ANTHROPIC_API_KEY")
        print("Anthropic API key found ✓")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Test OpenAI import
    try:
        import openai
        print(f"OpenAI library version: {openai.__version__} ✓")
    except ImportError as e:
        print(f"Error importing OpenAI: {e}")
    
    # Test Anthropic import
    try:
        import anthropic
        print(f"Anthropic library version: {anthropic.__version__} ✓")
    except ImportError as e:
        print(f"Error importing Anthropic: {e}")
    
    print("\nEnvironment setup test complete!")

if __name__ == "__main__":
    main()
EOL
```

Run the test script:

```bash
python test_environment.py
```

## Step 6: Install Development Tools

Now let's install development tools for code quality:

```bash
# Install development tools
pip install -U pytest pytest-asyncio black isort mypy ruff

# Add them to requirements-dev.txt
pip freeze | grep -E 'pytest|black|isort|mypy|ruff' > requirements-dev.txt
```

Create a simple configuration for these tools:

```bash
cat > pyproject.toml << 'EOL'
[tool.black]
line-length = 88
target-version = ["py310"]

[tool.isort]
profile = "black"
line_length = 88

[tool.ruff]
line-length = 88
target-version = "py310"
select = ["E", "F", "I", "N", "W", "B"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
EOL
```

## Step 7: Set Up Version Control

```bash
# Initialize git repository
git init

# Create a .gitignore file
cat > .gitignore << 'EOL'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Environment variables
.env

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db
EOL

# Make initial commit
git add .
git commit -m "Initial project setup"
```

## Step 8: Install Jupyter Notebook (Optional)

If you prefer interactive development:

```bash
pip install jupyter
```

Start a Jupyter notebook:

```bash
jupyter notebook
```

## Conclusion

Your development environment is now set up and ready for building LLM agents! 

This setup provides you with:
- A clean, isolated Python environment
- Essential libraries for LLM integration
- A structured project layout
- Secure API key management
- Development tools for code quality
- Version control

In the next section, we'll configure the API keys needed to access various LLM services. 