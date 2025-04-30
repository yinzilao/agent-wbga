"""
Complete Question-Answering Agent

This module brings together all the components we've developed to create a
complete QnA agent with memory management and tool integration capabilities.
"""

import os
import argparse
import asyncio
from pathlib import Path
import sys
import json
from typing import Dict, List, Optional, Any, Union

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from src.memory.conversation_memory import (
    Message, 
    InMemoryConversation, 
    FileConversation, 
    SummaryConversation
)
from src.tools.basic_tools import (
    ToolRegistry, 
    CalculatorTool, 
    DateTimeTool, 
    WebSearchTool,
    ToolError
)
from src.utils.env import load_env_variables, get_required_env_var
from src.utils.api_clients import get_openai_client, get_anthropic_client
from agent_framework import LLMConfig

# Load environment variables
load_env_variables()


class CompleteQnAAgent:
    """Complete QnA Agent with memory and tool capabilities."""
    
    def __init__(
        self, 
        memory_type: str = "in_memory",
        file_path: Optional[str] = None,
        system_prompt: str = "You are a helpful AI assistant with access to tools. Answer the user's questions accurately and concisely.",
        llm_config: Optional[LLMConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
        verbose: bool = False,
    ):
        """Initialize the Complete QnA Agent.
        
        Args:
            memory_type: Type of memory to use ("in_memory", "file", or "summary")
            file_path: Path to the conversation file (required for file memory)
            system_prompt: Instructions for the agent
            llm_config: Configuration for LLM calls
            tool_registry: Registry of tools the agent can use
            verbose: Whether to print verbose output
        """
        self.system_prompt = system_prompt
        self.llm_config = llm_config or LLMConfig()
        self.verbose = verbose
        
        # Set up tool registry
        self.tool_registry = tool_registry or ToolRegistry()
        if not self.tool_registry.tools:
            # Add default tools if registry is empty
            self.tool_registry.register_tool(CalculatorTool())
            self.tool_registry.register_tool(DateTimeTool())
            self.tool_registry.register_tool(WebSearchTool())
        
        # Set up memory
        if memory_type == "in_memory":
            self.memory = InMemoryConversation()
        elif memory_type == "file":
            if not file_path:
                raise ValueError("file_path is required for file memory")
            self.memory = FileConversation(file_path)
        elif memory_type == "summary":
            self.memory = SummaryConversation(
                summarize_func=self._summarize_conversation,
                summary_threshold=8,
                retain_messages=4
            )
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        
        # Add system message with tool instructions
        tool_instructions = self._generate_tool_instructions()
        full_system_prompt = f"{system_prompt}\n\n{tool_instructions}"
        
        if isinstance(self.memory, SummaryConversation):
            # For summary memory, we need to use the async add_message method
            asyncio.run(self.memory.add_message(Message("system", full_system_prompt)))
        else:
            self.memory.add_message(Message("system", full_system_prompt))
    
    def _generate_tool_instructions(self) -> str:
        """Generate instructions for using tools.
        
        Returns:
            Instructions as a string
        """
        tools_description = self.tool_registry.get_tool_descriptions()
        
        return f"""
        You have access to the following tools:
        
        {tools_description}
        
        To use a tool, use the following format in your response:
        
        ```tool
        tool_name: parameter
        ```
        
        For example:
        
        ```tool
        calculator: 2 + 2 * 3
        ```
        
        ```tool
        datetime: now
        ```
        
        You can use multiple tools in a single response. After using a tool, explain the result to the user.
        """
    
    async def _summarize_conversation(self, messages) -> str:
        """Summarize a list of messages using the LLM.
        
        Args:
            messages: List of messages to summarize
        
        Returns:
            Summary text
        """
        messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        summarization_prompt = f"""
        Please summarize the key points from the following conversation.
        Focus on important information, user preferences, and any decisions made.
        Keep the summary concise but comprehensive.
        
        Conversation:
        {messages_text}
        
        Summary:
        """
        
        provider = self.llm_config.provider
        params = self.llm_config.to_params()
        
        if provider == "openai":
            client = get_openai_client()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                    {"role": "user", "content": summarization_prompt}
                ],
                **params
            )
            return response.choices[0].message.content
        
        elif provider == "anthropic":
            client = get_anthropic_client()
            response = await asyncio.to_thread(
                client.messages.create,
                messages=[
                    {"role": "user", "content": summarization_prompt}
                ],
                system="You are a helpful assistant that summarizes conversations.",
                **params
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            Response content from LLM
        """
        provider = self.llm_config.provider
        params = self.llm_config.to_params()
        
        if self.verbose:
            print(f"Sending request to {provider} with {len(messages)} messages...")
        
        if provider == "openai":
            client = get_openai_client()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=messages,
                **params
            )
            return response.choices[0].message.content
        
        elif provider == "anthropic":
            client = get_anthropic_client()
            # Convert messages format for Anthropic
            anthropic_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    # Anthropic handles system messages differently
                    continue
                anthropic_messages.append(msg)
            
            # Add system prompt if present
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            system_prompt = system_messages[0]["content"] if system_messages else None
            
            response = await asyncio.to_thread(
                client.messages.create,
                messages=anthropic_messages,
                system=system_prompt,
                **params
            )
            return response.content[0].text
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _extract_and_execute_tools(self, response: str) -> str:
        """Extract tool calls from a response and execute them.
        
        Args:
            response: LLM response containing tool calls
        
        Returns:
            Response with tool calls replaced by results
        """
        import re
        # Extract tool calls using regex
        pattern = r"```tool\s*\n(.*?): (.*?)\n```"
        tool_calls = re.findall(pattern, response, re.DOTALL)
        
        if not tool_calls:
            return response
        
        if self.verbose:
            print(f"Found {len(tool_calls)} tool calls in response.")
        
        # Execute each tool and collect results
        results = []
        for tool_name, tool_param in tool_calls:
            tool_name = tool_name.strip()
            tool_param = tool_param.strip()
            
            if self.verbose:
                print(f"Executing tool: {tool_name} with parameter: {tool_param}")
            
            try:
                result = await self.tool_registry.execute_tool(tool_name, tool_param)
                results.append((tool_name, tool_param, result, None))
                
                if self.verbose:
                    print(f"Tool result: {result}")
            except ToolError as e:
                results.append((tool_name, tool_param, None, str(e)))
                
                if self.verbose:
                    print(f"Tool error: {e}")
        
        # Replace tool calls with results
        modified_response = response
        for tool_name, tool_param, result, error in results:
            tool_call = f"```tool\n{tool_name}: {tool_param}\n```"
            if error:
                replacement = f"```tool\n{tool_name}: {tool_param}\n```\n\nTool Error: {error}"
            else:
                replacement = f"```tool\n{tool_name}: {tool_param}\n```\n\nTool Result: {result}"
            
            modified_response = modified_response.replace(tool_call, replacement)
        
        return modified_response
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message, potentially using tools, and generate a response.
        
        Args:
            user_message: The user's message content
        
        Returns:
            Agent's response
        """
        # Add user message to memory
        user_msg = Message("user", user_message)
        
        if self.verbose:
            print(f"Processing user message: {user_message}")
        
        # Different memory types have different add_message interfaces
        if isinstance(self.memory, SummaryConversation):
            await self.memory.add_message(user_msg)
        else:
            self.memory.add_message(user_msg)
        
        # Get initial LLM response
        if self.verbose:
            print("Getting initial LLM response...")
        
        initial_response = await self.get_llm_response(self.memory.to_api_messages())
        
        # Extract and execute any tool calls in the response
        if self.verbose:
            print("Extracting and executing tool calls...")
        
        final_response = await self._extract_and_execute_tools(initial_response)
        
        # Add assistant message to memory
        assistant_msg = Message("assistant", final_response)
        
        if isinstance(self.memory, SummaryConversation):
            await self.memory.add_message(assistant_msg)
        else:
            self.memory.add_message(assistant_msg)
        
        return final_response
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        if self.verbose:
            print("Resetting conversation...")
        
        self.memory.clear()
        
        # Re-add system message with tool instructions
        tool_instructions = self._generate_tool_instructions()
        full_system_prompt = f"{self.system_prompt}\n\n{tool_instructions}"
        
        if isinstance(self.memory, SummaryConversation):
            # For summary memory, we need to use the async add_message method
            asyncio.run(self.memory.add_message(Message("system", full_system_prompt)))
        else:
            self.memory.add_message(Message("system", full_system_prompt))
    
    def save_conversation(self, file_path: str) -> None:
        """Save the current conversation to a file.
        
        Args:
            file_path: Path to save the conversation to
        """
        if isinstance(self.memory, FileConversation):
            # For FileConversation, the conversation is already saved
            if self.verbose:
                print(f"Conversation is already saved to {self.memory.file_path}")
            return
        
        try:
            # Get all messages
            messages = self.memory.get_messages()
            
            # Convert to serializable format
            data = [msg.to_dict() for msg in messages]
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            if self.verbose:
                print(f"Conversation saved to {file_path}")
        
        except Exception as e:
            print(f"Error saving conversation: {e}")


async def run_interactive_session(agent: CompleteQnAAgent):
    """Run an interactive session with the agent.
    
    Args:
        agent: The QnA Agent
    """
    print("\n===== Complete QnA Agent Interactive Session =====")
    print("Type 'exit', 'quit', or use Ctrl+C to end the session.")
    print("Type 'reset' to reset the conversation.")
    print("Type 'save [filename]' to save the conversation.")
    print("=================================================\n")
    
    try:
        while True:
            # Get user input
            user_input = input("\nUser: ")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            # Check for reset command
            if user_input.lower() == "reset":
                agent.reset_conversation()
                print("Conversation reset.")
                continue
            
            # Check for save command
            if user_input.lower().startswith("save "):
                filename = user_input[5:].strip()
                if not filename:
                    print("Please specify a filename.")
                    continue
                
                if not filename.endswith(".json"):
                    filename += ".json"
                
                conversations_dir = Path(__file__).parent / "conversations"
                conversations_dir.mkdir(exist_ok=True)
                file_path = conversations_dir / filename
                
                agent.save_conversation(file_path)
                print(f"Conversation saved to {file_path}")
                continue
            
            # Process the message
            print("Processing...")
            response = await agent.process_message(user_input)
            
            # Print the response
            print(f"\nAssistant: {response}")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Complete QnA Agent")
    
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic"],
        help="LLM provider (default: openai)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-3.5-turbo",
        help="Model name (default: gpt-3.5-turbo)",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)",
    )
    
    parser.add_argument(
        "--memory",
        type=str,
        default="in_memory",
        choices=["in_memory", "file", "summary"],
        help="Memory type (default: in_memory)",
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Path to conversation file (for file memory)",
    )
    
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant with access to tools. Answer the user's questions accurately and concisely.",
        help="System prompt for the agent",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


async def main():
    """Run the Complete QnA Agent."""
    args = parse_args()
    
    # Create the LLM config
    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    
    # Create tool registry
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(DateTimeTool())
    
    # Add web search if API keys are available
    try:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        google_cse_id = os.getenv("GOOGLE_CSE_ID")
        
        if google_api_key and google_cse_id:
            registry.register_tool(WebSearchTool(
                api_key=google_api_key,
                custom_search_id=google_cse_id
            ))
        else:
            # Fallback to DuckDuckGo
            registry.register_tool(WebSearchTool())
    except Exception as e:
        print(f"Warning: Could not set up web search tool: {e}")
    
    # Set up file path for file memory
    file_path = None
    if args.memory == "file":
        if args.file:
            file_path = args.file
        else:
            conversations_dir = Path(__file__).parent / "conversations"
            conversations_dir.mkdir(exist_ok=True)
            file_path = conversations_dir / "conversation.json"
    
    # Create the agent
    agent = CompleteQnAAgent(
        memory_type=args.memory,
        file_path=file_path,
        system_prompt=args.system_prompt,
        llm_config=llm_config,
        tool_registry=registry,
        verbose=args.verbose,
    )
    
    # Run the interactive session
    await run_interactive_session(agent)


if __name__ == "__main__":
    asyncio.run(main()) 