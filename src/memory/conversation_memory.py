"""
Conversation Memory Module

This module provides classes for managing conversation history and context
for LLM agents.
"""

from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import os
from pathlib import Path


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to a dictionary for API requests or serialization."""
        return {
            "role": self.role, 
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else None
        return cls(data["role"], data["content"], timestamp)
    
    def __str__(self) -> str:
        """String representation of the message."""
        return f"{self.role.capitalize()}: {self.content}"


class ConversationMemory:
    """Base class for conversation memory."""
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation memory.
        
        Args:
            message: The message to add
        """
        raise NotImplementedError("Subclasses must implement add_message")
    
    def get_messages(self, last_n: Optional[int] = None) -> List[Message]:
        """Get the messages from memory.
        
        Args:
            last_n: Number of messages to retrieve (default: all)
        
        Returns:
            List of messages
        """
        raise NotImplementedError("Subclasses must implement get_messages")
    
    def to_api_messages(self, last_n: Optional[int] = None) -> List[Dict[str, str]]:
        """Convert messages to format expected by API.
        
        Args:
            last_n: Number of messages to retrieve (default: all)
        
        Returns:
            List of message dictionaries
        """
        messages = self.get_messages(last_n)
        return [{"role": msg.role, "content": msg.content} for msg in messages]
    
    def clear(self) -> None:
        """Clear all messages from memory."""
        raise NotImplementedError("Subclasses must implement clear")


class InMemoryConversation(ConversationMemory):
    """Manages conversation history in memory."""
    
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
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []


class FileConversation(ConversationMemory):
    """Manages conversation history persisted to a file."""
    
    def __init__(self, file_path: Union[str, Path], max_messages: int = 100):
        """Initialize a file-backed conversation.
        
        Args:
            file_path: Path to the conversation file
            max_messages: Maximum number of messages to store
        """
        self.file_path = Path(file_path)
        self.max_messages = max_messages
        self.messages: List[Message] = []
        
        # Load existing messages if file exists
        self._load_messages()
    
    def _load_messages(self) -> None:
        """Load messages from the file."""
        if not self.file_path.exists():
            self.messages = []
            return
        
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.messages = [Message.from_dict(msg) for msg in data]
                
                # Trim if needed
                if len(self.messages) > self.max_messages:
                    self.messages = self.messages[-self.max_messages:]
        except Exception as e:
            print(f"Error loading conversation: {e}")
            self.messages = []
    
    def _save_messages(self) -> None:
        """Save messages to the file."""
        try:
            # Create directory if it doesn't exist
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.file_path, "w", encoding="utf-8") as f:
                data = [msg.to_dict() for msg in self.messages]
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving conversation: {e}")
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        
        # Trim if needed
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
        
        # Save to file
        self._save_messages()
    
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
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self._save_messages()


class SummaryConversation(ConversationMemory):
    """Conversation memory that summarizes older messages to manage context length."""
    
    def __init__(
        self, 
        summarize_func, 
        summary_threshold: int = 10, 
        retain_messages: int = 5,
        max_messages: int = 100
    ):
        """Initialize a summarizing conversation.
        
        Args:
            summarize_func: Function to summarize messages
            summary_threshold: Number of messages that triggers summarization
            retain_messages: Number of recent messages to retain unchanged
            max_messages: Maximum total messages (including summary)
        """
        self.messages: List[Message] = []
        self.summarize_func = summarize_func
        self.summary_threshold = summary_threshold
        self.retain_messages = retain_messages
        self.max_messages = max_messages
        self.summary: Optional[Message] = None
    
    async def _maybe_summarize(self) -> None:
        """Check if summarization is needed and perform it."""
        user_assistant_messages = [
            msg for msg in self.messages 
            if msg.role in ["user", "assistant"]
        ]
        
        # Check if we have enough messages to summarize
        if len(user_assistant_messages) >= self.summary_threshold:
            # Keep the most recent messages
            to_retain = user_assistant_messages[-self.retain_messages:]
            to_summarize = user_assistant_messages[:-self.retain_messages]
            
            # Skip if nothing to summarize
            if not to_summarize:
                return
            
            # Generate summary
            summary_text = await self.summarize_func(to_summarize)
            
            # Create a summary message
            self.summary = Message("system", f"Summary of previous conversation: {summary_text}")
            
            # Update messages list: keep system messages + summary + retained messages
            system_messages = [msg for msg in self.messages if msg.role == "system" and msg != self.summary]
            
            # Remove old summary if exists
            system_messages = [msg for msg in system_messages if not msg.content.startswith("Summary of previous conversation:")]
            
            self.messages = system_messages + [self.summary] + to_retain
    
    async def add_message(self, message: Message) -> None:
        """Add a message to the conversation.
        
        Args:
            message: The message to add
        """
        self.messages.append(message)
        
        # Check if we need to summarize
        await self._maybe_summarize()
        
        # Trim if still needed
        if len(self.messages) > self.max_messages:
            # Keep system messages and trim the rest
            system_messages = [msg for msg in self.messages if msg.role == "system"]
            other_messages = [msg for msg in self.messages if msg.role != "system"]
            
            # Keep the most recent non-system messages
            other_messages = other_messages[-(self.max_messages - len(system_messages)):]
            
            self.messages = system_messages + other_messages
    
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
    
    def clear(self) -> None:
        """Clear all messages."""
        self.messages = []
        self.summary = None 