"""
Basic Agent Framework for Q&A Agent

This module provides the foundational structure for a simple question-answering agent,
including prompt management and response handling.
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime

# Add the src directory to the path
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / "src"))

from utils.env import load_env_variables
from utils.api_clients import get_openai_client, get_anthropic_client

# Load environment variables
load_env_variables()

class Message:
    """Represents a message in a conversation."""
    
    def __init__(self, role: str, content: str, timestamp: Optional[datetime] = None):
        """Initialize a message.
        
        Args:
            role: The role of the message sender (e.g., "user", "assistant")
            content: The content of the message
            timestamp: When the message was created (default: current time)
        """
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, str]:
        """Convert message to a dictionary for API requests."""
        return {"role": self.role, "content": self.content}
    
    def __str__(self) -> str:
        """String representation of the message."""
        return f"{self.role.capitalize()}: {self.content}"


class Conversation:
    """Manages a conversation history."""
    
    def __init__(self, max_messages: int = 100):
        """Initialize a conversation.
        
        Args:
            max_messages: Maximum number of messages to store
        """
        self.messages: List[Message] = []
        self.max_messages = max_messages
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        # Trim if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """Get the last N messages.
        
        Args:
            last_n: Number of messages to retrieve (default: all)
        
        Returns:
            List of messages
        """
        if last_n is None:
            return self.messages
        return self.messages[-last_n:]
    
    def to_api_messages(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Convert messages to format expected by API.
        
        Args:
            last_n: Number of messages to retrieve (default: all)
        
        Returns:
            List of message dictionaries
        """
        messages = self.get_messages(last_n)
        return [msg.to_dict() for msg in messages]
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []


class PromptTemplate:
    """Template for constructing prompts."""
    
    def __init__(self, template: str):
        """Initialize a prompt template.
        
        Args:
            template: String template with {placeholders}
        """
        self.template = template
    
    def format(self, **kwargs) -> str:
        """Format the template with provided values.
        
        Args:
            **kwargs: Values for placeholders
        
        Returns:
            Formatted prompt
        """
        return self.template.format(**kwargs)


class LLMConfig:
    """Configuration for LLM calls."""
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
    ):
        """Initialize LLM configuration.
        
        Args:
            provider: The LLM provider (e.g., "openai", "anthropic")
            model: The model to use
            temperature: Randomness parameter (0-1)
            max_tokens: Maximum tokens in response
            top_p: Nucleus sampling parameter
        """
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
    
    def to_params(self) -> Dict[str, Any]:
        """Convert to provider-specific parameters.
        
        Returns:
            Dictionary of parameters
        """
        if self.provider == "openai":
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.max_tokens:
                params["max_tokens"] = self.max_tokens
            return params
        
        elif self.provider == "anthropic":
            params = {
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.max_tokens:
                params["max_tokens"] = self.max_tokens
            return params
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


class QnAAgent:
    """Basic Question-Answering Agent."""
    
    def __init__(
        self, 
        system_prompt: str = "You are a helpful AI assistant. Answer the user's questions accurately and concisely.",
        llm_config: Optional[LLMConfig] = None,
    ):
        """Initialize the QnA Agent.
        
        Args:
            system_prompt: Instructions for the agent
            llm_config: Configuration for LLM calls
        """
        self.conversation = Conversation()
        self.system_prompt = system_prompt
        self.llm_config = llm_config or LLMConfig()
        
        # Add system message
        self.conversation.add_message(Message("system", system_prompt))
    
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
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and generate a response.
        
        Args:
            user_message: The user's message content
        
        Returns:
            Agent's response
        """
        # Add user message to conversation
        self.conversation.add_message(Message("user", user_message))
        
        # Get LLM response
        response_content = await self.get_llm_response(self.conversation.to_api_messages())
        
        # Add assistant message to conversation
        self.conversation.add_message(Message("assistant", response_content))
        
        return response_content
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation.clear()
        # Re-add system message
        self.conversation.add_message(Message("system", self.system_prompt))


async def main():
    """Demo the QnA Agent."""
    # Create agent
    agent = QnAAgent(
        system_prompt="You are a knowledgeable AI assistant. Provide concise, helpful answers to questions.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.5)
    )
    
    # Process a question
    response = await agent.process_message("What is an LLM Agent?")
    print(f"Response: {response}")
    
    # Follow-up question (testing conversation memory)
    response = await agent.process_message("What are the key components of one?")
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main()) 