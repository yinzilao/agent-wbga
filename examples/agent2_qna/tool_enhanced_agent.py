"""
Tool-Enhanced Q&A Agent

This module builds on the basic agent framework to add tool usage capabilities,
enabling the agent to use tools like web search, calculator, and datetime functions.
"""

import os
import json
import asyncio
import re
from pathlib import Path
import sys
from typing import Dict, List, Optional, Any, Union

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from src.memory.conversation_memory import Message, InMemoryConversation
from src.tools.basic_tools import (
    ToolRegistry, 
    CalculatorTool, 
    DateTimeTool, 
    WebSearchTool,
    ToolError
)
from src.utils.env import load_env_variables
from src.utils.api_clients import get_openai_client, get_anthropic_client
from agent_framework import LLMConfig

# Load environment variables
load_env_variables()


class ToolEnhancedAgent:
    """QnA Agent with tool-using capabilities."""
    
    def __init__(
        self, 
        system_prompt: str = "You are a helpful AI assistant with access to tools. Answer the user's questions accurately and concisely.",
        llm_config: Optional[LLMConfig] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """Initialize the Tool-Enhanced Agent.
        
        Args:
            system_prompt: Instructions for the agent
            llm_config: Configuration for LLM calls
            tool_registry: Registry of tools the agent can use
        """
        self.conversation = InMemoryConversation()
        self.system_prompt = system_prompt
        self.llm_config = llm_config or LLMConfig()
        
        # Set up tool registry
        self.tool_registry = tool_registry or ToolRegistry()
        if not self.tool_registry.tools:
            # Add default tools if registry is empty
            self.tool_registry.register_tool(CalculatorTool())
            self.tool_registry.register_tool(DateTimeTool())
        
        # Add system message with tool instructions
        tool_instructions = self._generate_tool_instructions()
        full_system_prompt = f"{system_prompt}\n\n{tool_instructions}"
        self.conversation.add_message(Message("system", full_system_prompt))
    
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
    
    async def get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            Response content from LLM
        """
        provider = self.llm_config.provider
        params = self.llm_config.to_params()
        
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
        # Extract tool calls using regex
        pattern = r"```tool\s*\n(.*?): (.*?)\n```"
        tool_calls = re.findall(pattern, response, re.DOTALL)
        
        if not tool_calls:
            return response
        
        # Execute each tool and collect results
        results = []
        for tool_name, tool_param in tool_calls:
            tool_name = tool_name.strip()
            tool_param = tool_param.strip()
            
            try:
                result = await self.tool_registry.execute_tool(tool_name, tool_param)
                results.append((tool_name, tool_param, result, None))
            except ToolError as e:
                results.append((tool_name, tool_param, None, str(e)))
        
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
        # Add user message to conversation
        self.conversation.add_message(Message("user", user_message))
        
        # Get initial LLM response
        initial_response = await self.get_llm_response(self.conversation.to_api_messages())
        
        # Extract and execute any tool calls in the response
        final_response = await self._extract_and_execute_tools(initial_response)
        
        # Add assistant message to conversation
        self.conversation.add_message(Message("assistant", final_response))
        
        return final_response
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation.clear()
        # Re-add system message with tool instructions
        tool_instructions = self._generate_tool_instructions()
        full_system_prompt = f"{self.system_prompt}\n\n{tool_instructions}"
        self.conversation.add_message(Message("system", full_system_prompt))


async def demo_calculator():
    """Demonstrate the agent using the calculator tool."""
    print("\n===== Calculator Tool Demo =====")
    
    # Create registry with calculator tool
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    
    # Create agent
    agent = ToolEnhancedAgent(
        system_prompt="You are a math assistant who helps solve mathematical problems.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.2),
        tool_registry=registry
    )
    
    # Process questions
    questions = [
        "What is 123 + 456?",
        "Calculate the square root of 169.",
        "What is the value of sin(pi/4)?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = await agent.process_message(question)
        print(f"Assistant: {response}")


async def demo_datetime():
    """Demonstrate the agent using the datetime tool."""
    print("\n===== DateTime Tool Demo =====")
    
    # Create registry with datetime tool
    registry = ToolRegistry()
    registry.register_tool(DateTimeTool())
    
    # Create agent
    agent = ToolEnhancedAgent(
        system_prompt="You are a helpful assistant who can provide date and time information.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.2),
        tool_registry=registry
    )
    
    # Process questions
    questions = [
        "What is today's date?",
        "How many days are there between January 1, 2023 and December 31, 2023?"
    ]
    
    for question in questions:
        print(f"\nUser: {question}")
        response = await agent.process_message(question)
        print(f"Assistant: {response}")


async def demo_multi_tool():
    """Demonstrate the agent using multiple tools."""
    print("\n===== Multi-Tool Demo =====")
    
    # Create registry with multiple tools
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(DateTimeTool())
    registry.register_tool(WebSearchTool())
    
    # Create agent
    agent = ToolEnhancedAgent(
        system_prompt="You are a helpful assistant who can use various tools to answer questions accurately.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.2),
        tool_registry=registry
    )
    
    # Process a complex question that might use multiple tools
    question = "What is the current year plus 25? Also, what can you tell me about the Python programming language?"
    
    print(f"\nUser: {question}")
    response = await agent.process_message(question)
    print(f"Assistant: {response}")


async def interactive_demo():
    """Run an interactive demo of the tool-enhanced agent."""
    print("\n===== Tool-Enhanced Agent Interactive Demo =====")
    print("Type 'exit', 'quit', or use Ctrl+C to end the session.")
    print("Type 'reset' to reset the conversation.")
    print("================================================\n")
    
    # Create registry with multiple tools
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(DateTimeTool())
    
    # Create agent
    agent = ToolEnhancedAgent(
        system_prompt="You are a helpful assistant who can use tools to answer questions.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.7),
        tool_registry=registry
    )
    
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
            
            # Process the message
            print("Processing...")
            response = await agent.process_message(user_input)
            
            # Print the response
            print(f"\nAssistant: {response}")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run demonstrations of the tool-enhanced agent."""
    await demo_calculator()
    await demo_datetime()
    await demo_multi_tool()
    
    # Uncomment to run interactive demo
    # await interactive_demo()


if __name__ == "__main__":
    asyncio.run(main()) 