#!/usr/bin/env python3
"""
Command-line interface for the QnA Agent.

This script provides a simple CLI for interacting with the QnA Agent.
"""

import asyncio
import argparse
from pathlib import Path
import sys

# Add the parent directory to sys.path to import the agent_framework
sys.path.append(str(Path(__file__).parent))

from agent_framework import QnAAgent, LLMConfig

async def interactive_session(agent: QnAAgent):
    """Run an interactive session with the agent.
    
    Args:
        agent: The QnA Agent
    """
    print("\n===== QnA Agent Interactive Session =====")
    print("Type 'exit', 'quit', or use Ctrl+C to end the session.")
    print("Type 'reset' to reset the conversation.")
    print("==========================================\n")
    
    try:
        while True:
            # Get user input
            user_input = input("\n> ")
            
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
            print(f"\n{response}")
    
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error: {e}")

def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="QnA Agent CLI")
    
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
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant. Answer the user's questions accurately and concisely.",
        help="System prompt for the agent",
    )
    
    return parser.parse_args()

async def main():
    """Run the QnA Agent CLI."""
    args = parse_args()
    
    # Create the LLM config
    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
    )
    
    # Create the agent
    agent = QnAAgent(
        system_prompt=args.system_prompt,
        llm_config=llm_config,
    )
    
    # Run the interactive session
    await interactive_session(agent)

if __name__ == "__main__":
    asyncio.run(main()) 