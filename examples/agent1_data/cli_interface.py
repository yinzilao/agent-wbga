"""
Command-Line Interface for Data Analysis Agent

This module provides a command-line interface for the Data Analysis Agent,
allowing it to be run from the terminal with various configuration options.
"""

import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent.parent))

from examples.agent2_data.data_analysis_agent import DataAnalysisAgent
from src.tools.data_tools import create_data_tool_registry
from src.tools.basic_tools import CalculatorTool, DateTimeTool
from src.utils.env import load_env_variables

# Load environment variables
load_env_variables()


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Data Analysis Agent CLI")
    
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
        default=0.2,
        help="Temperature for sampling (default: 0.2)",
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
        "--data-dir",
        type=str,
        help="Directory containing data files",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


async def run_cli():
    """Run the Data Analysis Agent CLI."""
    args = parse_args()
    
    # Create the LLM config
    llm_config = {
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
    }
    
    # Set up file path for file memory
    file_path = None
    if args.memory == "file":
        if args.file:
            file_path = args.file
        else:
            conversations_dir = Path(__file__).parent / "conversations"
            conversations_dir.mkdir(exist_ok=True)
            file_path = conversations_dir / "conversation.json"
    
    # Create the data tool registry
    tool_registry = create_data_tool_registry()
    
    # Add basic tools
    tool_registry.register_tool(CalculatorTool())
    tool_registry.register_tool(DateTimeTool())
    
    # Set up data directory
    data_dir = args.data_dir
    if not data_dir:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
    
    # Create the agent
    agent = DataAnalysisAgent(
        memory_type=args.memory,
        file_path=file_path,
        data_dir=data_dir,
        llm_config=llm_config,
        tool_registry=tool_registry,
        verbose=args.verbose,
    )
    
    print("\n===== Data Analysis Agent CLI =====")
    print(f"Provider: {args.provider}, Model: {args.model}")
    print(f"Memory: {args.memory}")
    print(f"Data directory: {data_dir}")
    print("==================================\n")
    
    print("Welcome to the Data Analysis Agent! I can help you analyze and visualize data.")
    print("You can start by loading a dataset with the data_load tool.")
    print("Example datasets in the data directory:")
    
    # List example datasets
    try:
        data_files = [f for f in os.listdir(data_dir) if f.endswith(('.csv', '.json', '.xlsx'))]
        if data_files:
            print(", ".join(data_files))
        else:
            print("No datasets found. You can place CSV, JSON, or Excel files in the data directory.")
    except Exception as e:
        print(f"Could not list data files: {e}")
    
    print("\nType 'exit', 'quit', or Ctrl+C to end the session.")
    print("Type 'reset' to reset the conversation.")
    print("Type 'save [filename]' to save the conversation.")
    
    # Interactive session
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


def main():
    """Entry point for the CLI."""
    asyncio.run(run_cli())


if __name__ == "__main__":
    main() 