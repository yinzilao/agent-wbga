"""
Memory-Enhanced Q&A Agent

This module builds on the basic agent framework to demonstrate more advanced
memory capabilities, including file-backed persistence and conversation summarization.
"""

import os
import asyncio
from pathlib import Path
import sys

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from src.memory.conversation_memory import (
    Message, 
    InMemoryConversation, 
    FileConversation, 
    SummaryConversation
)
from src.utils.env import load_env_variables
from src.utils.api_clients import get_openai_client, get_anthropic_client
from agent_framework import LLMConfig

# Load environment variables
load_env_variables()


class MemoryEnhancedAgent:
    """QnA Agent with enhanced memory capabilities."""
    
    def __init__(
        self, 
        memory_type: str = "in_memory",
        file_path: str = None,
        system_prompt: str = "You are a helpful AI assistant. Answer the user's questions accurately and concisely.",
        llm_config: LLMConfig = None,
    ):
        """Initialize the Memory-Enhanced Agent.
        
        Args:
            memory_type: Type of memory to use ("in_memory", "file", or "summary")
            file_path: Path to the conversation file (required for file memory)
            system_prompt: Instructions for the agent
            llm_config: Configuration for LLM calls
        """
        self.system_prompt = system_prompt
        self.llm_config = llm_config or LLMConfig()
        
        # Initialize memory based on type
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
        
        # Add system message
        self.memory.add_message(Message("system", system_prompt))
    
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
    
    async def get_llm_response(self, messages) -> str:
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
        # Add user message to memory
        user_msg = Message("user", user_message)
        
        # Different memory types have different add_message interfaces
        if isinstance(self.memory, SummaryConversation):
            await self.memory.add_message(user_msg)
        else:
            self.memory.add_message(user_msg)
        
        # Get LLM response
        response_content = await self.get_llm_response(self.memory.to_api_messages())
        
        # Add assistant message to memory
        assistant_msg = Message("assistant", response_content)
        
        if isinstance(self.memory, SummaryConversation):
            await self.memory.add_message(assistant_msg)
        else:
            self.memory.add_message(assistant_msg)
        
        return response_content
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.memory.clear()
        # Re-add system message
        self.memory.add_message(Message("system", self.system_prompt))


async def demo_in_memory():
    """Demonstrate the in-memory conversation agent."""
    print("\n===== In-Memory Conversation Demo =====")
    agent = MemoryEnhancedAgent(
        memory_type="in_memory",
        system_prompt="You are a helpful assistant. Be concise.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.5)
    )
    
    response = await agent.process_message("What is an LLM Agent?")
    print(f"User: What is an LLM Agent?\nAssistant: {response}\n")
    
    response = await agent.process_message("What are its main components?")
    print(f"User: What are its main components?\nAssistant: {response}\n")


async def demo_file_memory():
    """Demonstrate the file-backed conversation agent."""
    print("\n===== File-Backed Conversation Demo =====")
    conversations_dir = Path(__file__).parent / "conversations"
    conversations_dir.mkdir(exist_ok=True)
    
    agent = MemoryEnhancedAgent(
        memory_type="file",
        file_path=conversations_dir / "demo_conversation.json",
        system_prompt="You are a helpful assistant. Be concise.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.5)
    )
    
    response = await agent.process_message("Tell me about conversation memory.")
    print(f"User: Tell me about conversation memory.\nAssistant: {response}\n")
    
    response = await agent.process_message("Why is it important for agents?")
    print(f"User: Why is it important for agents?\nAssistant: {response}\n")
    
    print(f"Conversation saved to: {conversations_dir / 'demo_conversation.json'}")


async def demo_summary_memory():
    """Demonstrate the summarizing conversation agent."""
    print("\n===== Summarizing Conversation Demo =====")
    agent = MemoryEnhancedAgent(
        memory_type="summary",
        system_prompt="You are a helpful assistant. Be concise.",
        llm_config=LLMConfig(provider="openai", model="gpt-3.5-turbo", temperature=0.5)
    )
    
    # Generate a longer conversation to trigger summarization
    questions = [
        "What is Python used for?",
        "What are the key differences between Python 2 and Python 3?",
        "What are some popular Python libraries?",
        "How does memory management work in Python?",
        "What is the GIL in Python?",
        "How does Python handle concurrency?",
        "What are Python decorators?",
        "Can you explain Python's list comprehensions?",
        "What is the meaning of life?"
    ]
    
    for i, question in enumerate(questions):
        print(f"Question {i+1}: {question}")
        response = await agent.process_message(question)
        print(f"Response {i+1}: {response}\n")
        
        # Show if summarization has occurred
        if i >= 7:  # After 8 messages, summarization should have occurred
            system_messages = [
                msg for msg in agent.memory.messages 
                if msg.role == "system" and msg.content.startswith("Summary of previous conversation:")
            ]
            if system_messages:
                print("\n===== Conversation Summary =====")
                print(system_messages[0].content)
                print("================================\n")


async def main():
    """Run demos of the different memory types."""
    await demo_in_memory()
    await demo_file_memory()
    await demo_summary_memory()


if __name__ == "__main__":
    asyncio.run(main()) 