# Question-Answering Agent

This example demonstrates a simple Question-Answering (QnA) agent built using our LLM Agent framework. The agent can:

1. Understand and respond to user questions
2. Maintain conversation context across multiple messages
3. Support different LLM providers (OpenAI and Anthropic)

## Prerequisites

Before running this example, make sure you have:

1. Set up your development environment as described in `docs/04_development_environment_setup.md`
2. Configured your API keys as described in `docs/05_api_key_configuration.md`

## Structure

This example includes:

- `agent_framework.py`: The core implementation of the QnA agent
- `cli_interface.py`: A command-line interface for interacting with the agent

## Running the Example

### Using the Command-Line Interface

The easiest way to interact with the agent is through the CLI:

```bash
# With default settings (OpenAI, gpt-3.5-turbo)
python cli_interface.py

# Using Anthropic's Claude
python cli_interface.py --provider anthropic --model claude-3-haiku-20240307

# With custom system prompt
python cli_interface.py --system-prompt "You are an expert in machine learning. Provide detailed technical explanations."
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--provider` | LLM provider (`openai` or `anthropic`) | `openai` |
| `--model` | Model name | `gpt-3.5-turbo` |
| `--temperature` | Randomness parameter (0-1) | `0.7` |
| `--system-prompt` | Instructions for the agent | Generic helpful assistant |

### Using the Agent in Code

You can also use the agent directly in your Python code:

```python
import asyncio
from agent_framework import QnAAgent, LLMConfig

async def example():
    # Configure the agent
    agent = QnAAgent(
        system_prompt="You are a helpful assistant specialized in coding.",
        llm_config=LLMConfig(provider="openai", model="gpt-4")
    )
    
    # Process user questions
    response1 = await agent.process_message("How do I sort a list in Python?")
    print(response1)
    
    # Ask a follow-up question (the agent maintains context)
    response2 = await agent.process_message("Can you explain the time complexity?")
    print(response2)
    
    # Reset conversation if needed
    agent.reset_conversation()

if __name__ == "__main__":
    asyncio.run(example())
```

## Extending the Agent

Here are some ways you can extend this simple agent:

1. Add persistent storage for conversations
2. Implement streaming responses
3. Add tool-calling capabilities (see next examples)
4. Improve error handling and retry logic
5. Add response templating for consistent formatting

## Next Steps

After exploring this simple QnA agent, you can move on to the Data Analysis Agent example, which builds on these concepts and adds tool usage capabilities. 