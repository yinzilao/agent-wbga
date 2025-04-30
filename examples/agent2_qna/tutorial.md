# QnA Agent Tutorial

This tutorial will walk you through using and customizing the complete QnA agent we've built. This agent combines:

1. Memory management (in-memory, file-based, and summary-based)
2. Tool integration (calculator, datetime, web search)
3. Support for multiple LLM providers (OpenAI and Anthropic)

## Prerequisites

Before running the agent, make sure you have:

1. Python 3.8 or higher installed
2. Required Python packages installed (see requirements.txt)
3. API keys configured in your .env file:
   - For OpenAI: `OPENAI_API_KEY`
   - For Anthropic: `ANTHROPIC_API_KEY`
   - For Google Search (optional): `GOOGLE_API_KEY` and `GOOGLE_CSE_ID`

## Running the Agent

The simplest way to run the agent is:

```bash
cd examples/agent1_qna
python complete_agent.py
```

This will start an interactive session with the default settings:
- OpenAI's gpt-3.5-turbo model
- In-memory conversation management
- All available tools enabled

## Command Line Options

The agent supports several command-line options:

```bash
python complete_agent.py --help
```

Key options include:

- `--provider`: Choose between "openai" and "anthropic" (default: openai)
- `--model`: Specify the model to use (default: gpt-3.5-turbo)
- `--temperature`: Set the temperature for sampling (default: 0.7)
- `--memory`: Choose the memory type ("in_memory", "file", or "summary")
- `--file`: Specify a file path for file-based memory
- `--system-prompt`: Customize the system prompt
- `--verbose`: Enable verbose output for debugging

## Examples

### Using OpenAI with GPT-4

```bash
python complete_agent.py --provider openai --model gpt-4
```

### Using Anthropic Claude

```bash
python complete_agent.py --provider anthropic --model claude-3-sonnet-20240229
```

### Using File-Based Memory

```bash
python complete_agent.py --memory file --file ./conversations/my_conversation.json
```

### Using Summary-Based Memory

```bash
python complete_agent.py --memory summary
```

### Custom System Prompt

```bash
python complete_agent.py --system-prompt "You are a friendly AI assistant focused on educational answers. Explain concepts simply."
```

## Interactive Commands

During an interactive session, you can use the following commands:

- `exit` or `quit`: End the session
- `reset`: Reset the conversation history
- `save [filename]`: Save the current conversation to a file (doesn't apply to file-based memory which is saved automatically)

## Programming with the Agent

You can also integrate the agent into your own applications:

```python
import asyncio
from examples.agent1_qna.complete_agent import CompleteQnAAgent
from src.agent_framework import LLMConfig

async def example():
    # Configure the agent
    agent = CompleteQnAAgent(
        memory_type="in_memory",
        system_prompt="You are a helpful assistant.",
        llm_config=LLMConfig(
            provider="openai",
            model="gpt-3.5-turbo",
            temperature=0.7
        ),
        verbose=True
    )
    
    # Process a message
    response = await agent.process_message("What is the capital of France?")
    print(f"Response: {response}")
    
    # Process another message in the same conversation
    response = await agent.process_message("What is its population?")
    print(f"Response: {response}")
    
    # Save the conversation
    agent.save_conversation("./my_conversation.json")

# Run the example
asyncio.run(example())
```

## Customizing the Agent

The agent is designed to be extended. Here are some ways to customize it:

### Adding New Tools

You can create custom tools by extending the `BaseTool` class from `src.tools.basic_tools`:

```python
from src.tools.basic_tools import BaseTool

class MyCustomTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="my_tool",
            description="Description of what my tool does"
        )
    
    async def _execute(self, param: str) -> str:
        # Implement your tool's functionality here
        return f"Result of processing: {param}"

# Add your tool to the agent
from src.tools.basic_tools import ToolRegistry
registry = ToolRegistry()
registry.register_tool(MyCustomTool())

agent = CompleteQnAAgent(
    tool_registry=registry,
    # other parameters...
)
```

### Custom Memory Systems

You can create custom memory systems by implementing the `BaseConversation` abstract class from `src.memory.conversation_memory`.

## Troubleshooting

### API Key Issues

If you encounter "API key not found" or authentication errors:
1. Check that your `.env` file contains the necessary API keys
2. Make sure the `.env` file is in the correct location (project root)
3. Verify API key validity in the respective service dashboards

### Tool Execution Errors

If tools fail to execute properly:
1. Check the error message using the `--verbose` flag
2. Verify API keys for services like Google Search
3. Check network connectivity for tools that require internet access

### Memory Issues

If you have issues with file-based memory:
1. Ensure the specified directory exists and is writable
2. Check file permissions

## Next Steps

Here are some suggestions for extending the agent:
1. Add more sophisticated tools
2. Implement streaming responses
3. Add a web interface
4. Implement parallel tool execution
5. Add user authentication 