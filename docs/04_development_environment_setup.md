# Development Environment Setup

This guide will walk you through setting up a complete development environment for building LLM agents. We'll use Python as our primary language, as it has excellent support for LLM APIs and most related tools.

## Beginner's Guide: Setting Up Essential Tools

If you're completely new to programming or LLM development, this section will guide you through installing the essential tools you need to get started.

### Installing Python

Python is the programming language we'll use for building LLM agents.

1. **Download Python**:
   - Visit [python.org/downloads](https://www.python.org/downloads/)
   - Download the latest version (3.10 or newer)
   - On Windows: Check "Add Python to PATH" during installation
   - On macOS: The installer will guide you through the process
   - On Linux: Use your package manager (e.g., `sudo apt install python3 python3-pip` on Ubuntu)

2. **Verify Installation**:
   Open a terminal/command prompt and type:
   ```bash
   python --version
   # or on some systems:
   python3 --version
   ```
   You should see the Python version number.

### Installing Git

Git is a version control system that helps you track changes to your code.

1. **Download Git**:
   - Windows: Download from [git-scm.com](https://git-scm.com/download/win)
   - macOS: Install via [Homebrew](https://brew.sh/) with `brew install git` or download from [git-scm.com](https://git-scm.com/download/mac)
   - Linux: Use your package manager (e.g., `sudo apt install git` on Ubuntu)

2. **Configure Git**:
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **Verify Installation**:
   ```bash
   git --version
   ```

### Installing Visual Studio Code (VSCode)

VSCode is a popular, free code editor with excellent Python support.

1. **Download VSCode**:
   - Visit [code.visualstudio.com](https://code.visualstudio.com/)
   - Download and install the version for your operating system

2. **Install Python Extensions**:
   - Open VSCode
   - Go to Extensions (icon on the left sidebar or press Ctrl+Shift+X)
   - Search for "Python" and install the official Python extension by Microsoft
   - Also recommended: "Pylance" for improved Python language support

### Getting Your API Keys

To use LLM services, you'll need API keys:

1. **OpenAI API Key**:
   - Create an account at [platform.openai.com](https://platform.openai.com/)
   - Navigate to API keys section
   - Click "Create new secret key"
   - Copy and save your key securely (you won't be able to see it again)
   - Note: OpenAI requires payment information even for free tier usage

2. **Anthropic API Key** (Optional):
   - Create an account at [console.anthropic.com](https://console.anthropic.com/)
   - Navigate to API keys section
   - Create a new key and save it securely

### Basic Terminal/Command Line Skills

Here are some essential commands to know:

- **Windows Command Prompt / PowerShell**:
  - `cd folder_name`: Change directory
  - `dir`: List files in current directory
  - `mkdir folder_name`: Create a new directory
  - `cd ..`: Move up one directory

- **macOS/Linux Terminal**:
  - `cd folder_name`: Change directory
  - `ls`: List files in current directory
  - `mkdir folder_name`: Create a new directory
  - `cd ..`: Move up one directory

Now you're ready to proceed with setting up your Python development environment!

## Prerequisites

Before continuing with the rest of the setup, make sure you have:

- Python 3.10+ installed
- Git installed
- A code editor (like VSCode, PyCharm, etc.)
- Basic knowledge of terminal/command line operations
- Internet access to download packages and access LLM APIs
- API keys for the LLM services you plan to use

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

Once activated, you'll notice your command prompt changes to indicate you're in the virtual environment. All Python packages you install will now be isolated to this environment.

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

**Note for Windows users**: The command above uses Unix-style syntax. You can either:
1. Use PowerShell and run the same command
2. Create these directories manually through File Explorer
3. Create each directory individually with `mkdir src`, `mkdir src\agents`, etc.

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
# On Windows, you can use:
echo. > .env

# On macOS/Linux:
touch .env
```

Add your API keys to the `.env` file. Open it in your code editor and add:

```
# .env file
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Create a utility script to load environment variables:

```bash
# First, ensure the utils directory exists
mkdir -p src/utils

# Create the env.py file
```

Now create a file named `env.py` in the `src/utils` directory with the following content:

```python
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
```

## Step 5: Test Your Environment

Create a simple test script to verify that everything is working.

Create a file named `test_environment.py` in your project root with the following content:

```python
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

# On Windows, if grep is not available, you can manually create the file
# or install grep via Chocolatey or use PowerShell's Select-String
```

Create a simple configuration for these tools by creating a file named `pyproject.toml` in your project root with the following content:

```toml
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
```

## Step 7: Set Up Version Control

```bash
# Initialize git repository
git init

# Create a .gitignore file
```

Create a file named `.gitignore` in your project root with the following content:

```
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
```

Now make your initial commit:

```bash
# Add all files to git
git add .

# Create the initial commit
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

## VSCode Setup Tips for Beginners

If you're using VSCode, here are some helpful setup tips:

1. **Open Your Project**:
   - File > Open Folder > Select your project directory

2. **Set Up Python Interpreter**:
   - Press Ctrl+Shift+P (or Cmd+Shift+P on macOS)
   - Type "Python: Select Interpreter"
   - Choose the one from your virtual environment (it should have "(venv)" in the name)

3. **Useful Keyboard Shortcuts**:
   - Ctrl+` (backtick): Open/close terminal
   - F5: Run current file
   - Shift+Alt+F: Format document
   - Ctrl+Shift+X: Open extensions panel

4. **Enable Autosave**:
   - File > Auto Save

5. **Install Helpful Extensions**:
   - "Python" (official Microsoft extension)
   - "Pylance" for better Python language support
   - "Python Docstring Generator" for easy documentation
   - "GitLens" for enhanced Git integration

## Conclusion

Your development environment is now set up and ready for building LLM agents! 

This setup provides you with:
- A complete Python development environment
- Essential libraries for LLM integration
- A structured project layout
- Secure API key management
- Development tools for code quality
- Version control with Git
- A powerful code editor (if using VSCode)

In the next section, we'll explore how to interact with different LLM APIs and build your first agent.

## Troubleshooting Common Issues

### "Python is not recognized as a command"
- Make sure Python is added to your PATH during installation
- On Windows, try using `py` instead of `python`

### "pip is not recognized as a command"
- Try using `python -m pip` instead

### Virtual Environment Not Activating
- On Windows PowerShell, you might need to adjust execution policy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`
- Make sure you're in the project directory when activating

### API Key Issues
- Double-check that your `.env` file is in the correct location
- Verify that API keys are copied correctly with no extra spaces
- Make sure your API keys are active and have not expired

### Package Installation Errors
- Make sure your virtual environment is activated (you should see "(venv)" in your command prompt)
- Try updating pip: `pip install --upgrade pip`
- If a specific package fails, try installing it separately 