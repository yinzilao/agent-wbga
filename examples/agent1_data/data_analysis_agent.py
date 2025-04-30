"""
Data Analysis Agent

This module implements a specialized agent for data analysis tasks, including:
- Loading and exploring datasets
- Performing statistical analysis
- Creating visualizations
- Answering questions about data
"""

import os
import argparse
import asyncio
from pathlib import Path
import sys
import json
import time
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
    ToolError
)
from src.tools.data_tools import (
    DataLoadTool,
    DataQueryTool,
    DataStatsTool,
    DataVisualizeTool,
    create_data_tool_registry
)
from src.utils.env import load_env_variables
from src.utils.api_clients import get_openai_client, get_anthropic_client
from src.utils.logger import setup_logger, get_logger

# Load environment variables
load_env_variables()


class DataAnalysisAgent:
    """Agent specialized for data analysis tasks."""
    
    def __init__(
        self, 
        memory_type: str = "in_memory",
        file_path: Optional[str] = None,
        data_dir: Optional[str] = None,
        vis_dir: Optional[str] = None,
        system_prompt: str = "You are a data analysis assistant that helps analyze and visualize data. You have access to tools for data manipulation, statistical analysis, and visualization.",
        llm_config: Optional[Dict[str, Any]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        verbose: bool = False,
        log_level: str = "INFO",
    ):
        """Initialize the data analysis agent.
        
        Args:
            memory_type: Type of memory to use ('in_memory', 'file', or 'summary')
            file_path: Path to the file for file memory (only used if memory_type is 'file')
            data_dir: Directory containing data files
            vis_dir: Directory for saving visualization images
            system_prompt: System prompt for the agent
            llm_config: LLM configuration
            tool_registry: Tool registry to use (if not provided, a default one will be created)
            verbose: Whether to print verbose output
            log_level: Level of logging detail (DEBUG, INFO, WARNING, ERROR)
        """
        self.verbose = verbose
        self.system_prompt = system_prompt
        
        # Setup logging
        log_dir = os.path.join(os.path.dirname(__file__), "logs")
        self.logger = setup_logger(log_dir=log_dir, log_to_console=verbose, log_level=log_level)
        self.logger.info("========== INITIALIZING DATA ANALYSIS AGENT ==========")
        
        # Set up data_dir
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), "data")
        
        # Set up visualizations directory
        self.visualizations_dir = vis_dir or os.path.join(os.path.dirname(__file__), "visualizations")
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Set up LLM config
        self.llm_config = llm_config or {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
        }
        
        self.logger.info(f"Initializing agent with data_dir: {self.data_dir}")
        self.logger.info(f"Visualizations will be saved to: {self.visualizations_dir}")
        self.logger.info(f"Using LLM provider: {self.llm_config['provider']}, model: {self.llm_config['model']}")
        
        # Set up tool registry
        if tool_registry:
            self.tool_registry = tool_registry
        else:
            self.logger.info("Creating default tool registry")
            self.tool_registry = ToolRegistry()
            
            # Add data analysis tools with the correct data_dir
            self.tool_registry.register_tool(DataLoadTool(data_dir=self.data_dir))
            self.tool_registry.register_tool(DataQueryTool())
            self.tool_registry.register_tool(DataStatsTool())
            self.tool_registry.register_tool(DataVisualizeTool(save_dir=self.visualizations_dir))
            
            # Add basic tools
            self.tool_registry.register_tool(CalculatorTool())
            self.tool_registry.register_tool(DateTimeTool())
        
        tool_names = list(self.tool_registry.tools.keys())
        self.logger.info(f"Registered tools: {', '.join(tool_names)}")
        
        # Update DataLoadTool with the correct data_dir if it was provided
        if data_dir and "data_load" in self.tool_registry.tools:
            data_load_tool = self.tool_registry.tools["data_load"]
            if isinstance(data_load_tool, DataLoadTool):
                data_load_tool.data_dir = data_dir
                self.logger.info(f"Updated DataLoadTool with data_dir: {self.data_dir}")
        
        # Set up memory
        self.logger.info(f"Setting up memory type: {memory_type}")
        if memory_type == "in_memory":
            self.memory = InMemoryConversation()
            self.logger.info("Using InMemoryConversation")
        elif memory_type == "file":
            if not file_path:
                raise ValueError("file_path is required for file memory")
            self.memory = FileConversation(file_path)
            self.logger.info(f"Using FileConversation with path: {file_path}")
        elif memory_type == "summary":
            self.memory = SummaryConversation(
                summarize_func=self._summarize_conversation,
                summary_threshold=8,
                retain_messages=4
            )
            self.logger.info("Using SummaryConversation with threshold 8 and retain 4")
        else:
            raise ValueError(f"Unsupported memory type: {memory_type}")
        
        # Add system message with tool instructions
        tool_instructions = self._generate_tool_instructions()
        full_system_prompt = f"{system_prompt}\n\n{tool_instructions}"
        
        self.logger.debug("Adding system message with tool instructions")
        if isinstance(self.memory, SummaryConversation):
            # For summary memory, we need to use the async add_message method
            asyncio.run(self.memory.add_message(Message("system", full_system_prompt)))
        else:
            self.memory.add_message(Message("system", full_system_prompt))
        
        self.logger.info("Data Analysis Agent initialization complete")
    
    def _generate_tool_instructions(self) -> str:
        """Generate instructions for using tools.
        
        Returns:
            Instructions as a string
        """
        self.logger.debug("Generating tool instructions")
        tools_description = self.tool_registry.get_tool_descriptions()
        self.logger.debug(f"Tool descriptions: {len(tools_description)} chars")
        
        instructions = f"""
        You are a data analysis assistant that helps users explore, analyze, and visualize datasets.
        You can load datasets, query them, perform statistical analysis, and create visualizations.
        
        You have access to the following tools:
        
        {tools_description}
        
        To use a tool, use the following format in your response:
        
        ```tool
        tool_name: parameter
        ```
        
        For example:
        
        ```tool
        data_load: sales_data.csv
        ```
        
        ```tool
        data_query: sales_data | groupby('region').sum()
        ```
        
        ```tool
        data_viz: sales_data | bar | x=region,y=sales,title=Sales by Region
        ```
        
        You can use multiple tools in a single response. After using a tool, explain the result to the user.
        When analyzing data, be methodical and explain your approach. Suggest additional analyses that might be insightful.
        """
        
        self.logger.debug("Tool instructions generated successfully")
        return instructions
    
    async def _summarize_conversation(self, messages) -> str:
        """Summarize a list of messages using the LLM.
        
        Args:
            messages: List of messages to summarize
        
        Returns:
            Summary text
        """
        self.logger.info(f"Summarizing conversation with {len(messages)} messages")
        messages_text = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        
        summarization_prompt = f"""
        Please summarize the key points from the following data analysis conversation.
        Focus on the datasets that were analyzed, main findings, and any visualizations created.
        Keep the summary concise but comprehensive.
        
        Conversation:
        {messages_text}
        
        Summary:
        """
        
        provider = self.llm_config["provider"]
        
        self.logger.debug(f"Calling {provider} to generate summary")
        self.logger.data("Summarization Prompt", summarization_prompt[:500] + "..." if len(summarization_prompt) > 500 else summarization_prompt)
        
        start_time = time.time()
        if provider == "openai":
            client = get_openai_client()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes data analysis conversations."},
                    {"role": "user", "content": summarization_prompt}
                ],
                model=self.llm_config["model"],
                temperature=self.llm_config["temperature"],
            )
            result = response.choices[0].message.content
        
        elif provider == "anthropic":
            client = get_anthropic_client()
            response = await asyncio.to_thread(
                client.messages.create,
                messages=[
                    {"role": "user", "content": summarization_prompt}
                ],
                system="You are a helpful assistant that summarizes data analysis conversations.",
                model=self.llm_config["model"],
                temperature=self.llm_config["temperature"],
            )
            result = response.content[0].text
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Summary generated in {elapsed_time:.2f} seconds")
        self.logger.data("Summary Result", result[:500] + "..." if len(result) > 500 else result)
        
        return result
    
    async def get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            Response content from LLM
        """
        provider = self.llm_config["provider"]
        
        self.logger.info(f"Sending request to {provider} with {len(messages)} messages")
        
        # Log full input messages sent to LLM
        self.logger.data("LLM Input Messages", messages)
        
        # Rest of the existing code for getting the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"]
                break
        
        if last_user_msg:
            self.logger.data("Last User Message", last_user_msg)
        
        # Enhanced debugging: Log message structure
        if self.logger._should_log("DEBUG"):
            roles = [msg["role"] for msg in messages]
            role_counts = {role: roles.count(role) for role in set(roles)}
            self.logger.debug(f"Message structure: {role_counts}")
            
            # Log approximate token count (rough estimate)
            total_chars = sum(len(msg.get("content", "")) for msg in messages)
            est_tokens = total_chars / 4  # Very rough estimate: ~4 chars per token
            self.logger.debug(f"Approximate token count: ~{int(est_tokens)} tokens")
        
        start_time = time.time()
        if provider == "openai":
            client = get_openai_client()
            try:
                model = self.llm_config["model"]
                temperature = self.llm_config["temperature"]
                self.logger.debug(f"Calling OpenAI API with model={model}, temperature={temperature}")
                
                # Log full parameters in debug mode
                if self.logger._should_log("DEBUG"):
                    self.logger.debug(f"OpenAI request parameters: model={model}, temperature={temperature}, messages_count={len(messages)}")
                    
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    messages=messages,
                    model=self.llm_config["model"],
                    temperature=self.llm_config["temperature"],
                )
                content = response.choices[0].message.content
                elapsed_time = time.time() - start_time
                self.logger.info(f"Received response from OpenAI in {elapsed_time:.2f} seconds ({len(content)} chars)")
                
                # Log full response object, converting non-serializable objects to strings
                usage_dict = None
                if hasattr(response, 'usage'):
                    try:
                        # Try to convert to dict if _asdict() is available
                        usage_dict = response.usage._asdict() if hasattr(response.usage, '_asdict') else str(response.usage)
                    except Exception:
                        # Fallback to string representation
                        usage_dict = str(response.usage)
                
                self.logger.data("LLM Full Response Object", {
                    "id": response.id,
                    "model": response.model,
                    "content": content,
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": usage_dict
                })
                
                # Log response metadata if available
                if hasattr(response, 'usage') and response.usage:
                    try:
                        self.logger.debug(f"Token usage - prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}, total: {response.usage.total_tokens}")
                    except AttributeError:
                        self.logger.debug("Token usage information not available")
                
                self.logger.data("LLM Response", content[:1000] + "..." if len(content) > 1000 else content)
                return content
            except Exception as e:
                self.logger.error(f"Error getting response from OpenAI: {str(e)}")
                # More detailed error logging
                self.logger.error(f"Error type: {type(e).__name__}")
                import traceback
                self.logger.debug(f"OpenAI call traceback: {traceback.format_exc()}")
                raise
        
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
            
            # Log Anthropic-specific message format
            self.logger.data("Anthropic Input Format", {
                "messages": anthropic_messages,
                "system_prompt": system_prompt
            })
            
            try:
                model = self.llm_config["model"]
                temperature = self.llm_config["temperature"]
                self.logger.debug(f"Calling Anthropic API with model={model}, temperature={temperature}")
                
                # Log full parameters in debug mode
                if self.logger._should_log("DEBUG"):
                    self.logger.debug(f"Anthropic request parameters: model={model}, temperature={temperature}, system_prompt_length={len(system_prompt) if system_prompt else 0}")
                    self.logger.debug(f"Anthropic messages: {len(anthropic_messages)} messages after system prompt extraction")
                
                response = await asyncio.to_thread(
                    client.messages.create,
                    messages=anthropic_messages,
                    system=system_prompt,
                    model=self.llm_config["model"],
                    temperature=self.llm_config["temperature"],
                    max_tokens=1024,
                )
                content = response.content[0].text
                elapsed_time = time.time() - start_time
                self.logger.info(f"Received response from Anthropic in {elapsed_time:.2f} seconds ({len(content)} chars)")
                
                # Log full response object with safe serialization
                stop_reason = response.stop_reason if hasattr(response, 'stop_reason') else None
                
                # Safely convert usage to string to avoid serialization issues
                usage_str = None
                if hasattr(response, 'usage'):
                    usage_str = str(response.usage)
                
                self.logger.data("LLM Full Response Object", {
                    "id": response.id,
                    "model": response.model,
                    "content": content,
                    "stop_reason": stop_reason,
                    "usage": usage_str
                })
                
                # Log response metadata if available
                if hasattr(response, 'usage') and response.usage:
                    self.logger.debug(f"Token usage information: {str(response.usage)}")
                
                self.logger.data("LLM Response", content[:1000] + "..." if len(content) > 1000 else content)
                return content
            except Exception as e:
                self.logger.error(f"Error getting response from Anthropic: {str(e)}")
                # More detailed error logging
                self.logger.error(f"Error type: {type(e).__name__}")
                import traceback
                self.logger.debug(f"Anthropic call traceback: {traceback.format_exc()}")
                raise
        
        else:
            self.logger.error(f"Unsupported provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _extract_and_execute_tools(self, response: str) -> str:
        """Extract tool calls from a response and execute them.
        
        Args:
            response: LLM response containing tool calls
        
        Returns:
            Response with tool calls replaced by results
        """
        import re
        self.logger.info(f"Extracting tool calls from response ({len(response)} chars)")
        
        # Debug: Log a sample of the response for context
        if self.logger._should_log("DEBUG"):
            response_sample = response[:200] + "..." if len(response) > 200 else response
            self.logger.debug(f"Response sample for tool extraction: {response_sample}")
        
        # Extract tool calls using regex
        pattern = r"```tool\s*\n(.*?): (.*?)\n```"
        tool_calls = re.findall(pattern, response, re.DOTALL)
        
        if not tool_calls:
            self.logger.info("No tool calls found in response")
            return response
        
        self.logger.info(f"Found {len(tool_calls)} tool calls in response")
        
        # Debug: Log all extracted tool calls
        if self.logger._should_log("DEBUG"):
            for i, (tool_name, tool_param) in enumerate(tool_calls):
                self.logger.debug(f"Tool call {i+1} raw extraction - name: '{tool_name.strip()}', param: '{tool_param.strip()}'")
        
        # Execute each tool and collect results
        results = []
        for i, (tool_name, tool_param) in enumerate(tool_calls):
            tool_name = tool_name.strip()
            tool_param = tool_param.strip()
            
            self.logger.info(f"Tool call {i+1}: {tool_name} with param: {tool_param}")
            
            if tool_name not in self.tool_registry.tools:
                available_tools = list(self.tool_registry.tools.keys())
                error_msg = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
                self.logger.error(f"Tool not found: {tool_name}")
                
                # Debug: Check for similar tool names (typos)
                if self.logger._should_log("DEBUG"):
                    possible_matches = [t for t in available_tools if tool_name.lower() in t.lower() or t.lower() in tool_name.lower()]
                    if possible_matches:
                        self.logger.debug(f"Possible tool name matches: {', '.join(possible_matches)}")
                
                results.append((tool_name, tool_param, None, error_msg))
                continue
            
            try:
                self.logger.info(f"Executing tool: {tool_name}")
                
                # Debug: Log tool type and detailed parameters
                if self.logger._should_log("DEBUG"):
                    tool_obj = self.tool_registry.tools[tool_name]
                    tool_type = type(tool_obj).__name__
                    self.logger.debug(f"Tool type: {tool_type}, Parameter length: {len(tool_param)} chars")
                    
                    # Parse and log more details for specific tool types
                    if "query" in tool_name.lower() or "stats" in tool_name.lower():
                        parts = tool_param.split('|', 1)
                        if len(parts) == 2:
                            dataset, operation = parts[0].strip(), parts[1].strip()
                            self.logger.debug(f"Data operation - Dataset: '{dataset}', Operation: '{operation}'")
                
                start_time = time.time()
                result = await self.tool_registry.execute_tool(tool_name, tool_param)
                elapsed_time = time.time() - start_time
                self.logger.info(f"Tool execution completed in {elapsed_time:.2f} seconds")
                
                # Log the result (truncated if too long)
                result_str = str(result)
                if len(result_str) > 500:
                    self.logger.data(f"Tool Result ({tool_name})", result_str[:500] + "... [truncated]")
                else:
                    self.logger.data(f"Tool Result ({tool_name})", result_str)
                
                # Debug: Log result type and size
                if self.logger._should_log("DEBUG"):
                    result_type = type(result).__name__
                    result_size = len(result_str)
                    self.logger.debug(f"Tool result type: {result_type}, size: {result_size} chars")
                
                results.append((tool_name, tool_param, result, None))
                
            except ToolError as e:
                self.logger.error(f"Tool error: {str(e)}")
                results.append((tool_name, tool_param, None, str(e)))
            except Exception as e:
                self.logger.error(f"Unexpected error executing tool: {type(e).__name__}: {str(e)}")
                
                # Debug: More detailed error information
                if self.logger._should_log("DEBUG"):
                    import traceback
                    tb = traceback.format_exc()
                    self.logger.debug(f"Tool execution traceback: {tb}")
                
                results.append((tool_name, tool_param, None, f"Unexpected error: {type(e).__name__}: {e}"))
        
        # Replace tool calls with results
        self.logger.info("Replacing tool calls with results in response")
        modified_response = response
        
        # Debug: Track replacements
        replacement_count = 0
        
        for tool_name, tool_param, result, error in results:
            tool_call = f"```tool\n{tool_name}: {tool_param}\n```"
            if error:
                replacement = f"```tool\n{tool_name}: {tool_param}\n```\n\nTool Error: {error}"
                self.logger.debug(f"Replacing tool call {tool_name} with error message")
            else:
                replacement = f"```tool\n{tool_name}: {tool_param}\n```\n\nTool Result: {result}"
                self.logger.debug(f"Replacing tool call {tool_name} with result")
            
            # Check if the exact tool call text exists in the response
            if tool_call in modified_response:
                modified_response = modified_response.replace(tool_call, replacement)
                replacement_count += 1
            else:
                self.logger.warning(f"Exact tool call text not found for replacement: {tool_name}")
                
                # Debug: Try to diagnose the issue
                if self.logger._should_log("DEBUG"):
                    self.logger.debug(f"Original tool call: '{tool_call}'")
                    # Look for similar patterns
                    pattern = r"```tool\s*\n\s*" + re.escape(tool_name) + r"\s*:.*?\n\s*```"
                    alt_matches = re.findall(pattern, modified_response, re.DOTALL)
                    if alt_matches:
                        self.logger.debug(f"Found similar pattern: '{alt_matches[0]}'")
                        # Use the found pattern instead
                        modified_response = modified_response.replace(alt_matches[0], replacement)
                        replacement_count += 1
                        self.logger.debug("Replaced using alternative pattern match")
        
        # Debug: Log replacement results
        if self.logger._should_log("DEBUG"):
            self.logger.debug(f"Made {replacement_count} replacements out of {len(results)} tool calls")
        
        self.logger.info(f"Response after tool execution: {len(modified_response)} chars")
        return modified_response
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message, potentially using tools, and generate a response.
        
        Args:
            user_message: The user's message content
        
        Returns:
            Agent's response
        """
        self.logger.info("========== PROCESSING NEW USER MESSAGE ==========")
        self.logger.info(f"User message: {user_message}")
        
        # Debug: Log message context
        if self.logger._should_log("DEBUG"):
            message_length = len(user_message)
            word_count = len(user_message.split())
            self.logger.debug(f"Message stats: {message_length} chars, ~{word_count} words")
            
            # Add context about what we're about to do
            self.logger.debug("Process flow: 1) Add user message to memory → 2) Get LLM response → 3) Process tool calls → 4) Add assistant response to memory")
        
        # Add user message to memory
        user_msg = Message("user", user_message)
        
        try:
            # Different memory types have different add_message interfaces
            self.logger.info("Adding user message to memory")
            start_time = time.time()
            
            if isinstance(self.memory, SummaryConversation):
                self.logger.debug("Using SummaryConversation memory")
                await self.memory.add_message(user_msg)
            else:
                self.logger.debug(f"Using {type(self.memory).__name__} memory")
                self.memory.add_message(user_msg)
            
            memory_update_time = time.time() - start_time
            if self.logger._should_log("DEBUG"):
                self.logger.debug(f"Memory update completed in {memory_update_time:.4f}s")
            
            # Get all messages for debugging
            all_messages = self.memory.get_messages()
            self.logger.info(f"Conversation history has {len(all_messages)} messages")
            
            # Debug: Log conversation state
            if self.logger._should_log("DEBUG"):
                message_roles = [msg.role for msg in all_messages]
                role_counts = {role: message_roles.count(role) for role in set(message_roles)}
                self.logger.debug(f"Current conversation state: {role_counts}")
                
                # Log conversation flow
                if len(all_messages) > 1:
                    # Show the recent conversation flow (last 3 turns)
                    recent_flow = [msg.role for msg in all_messages[-min(6, len(all_messages)):]]
                    self.logger.debug(f"Recent conversation flow: {' → '.join(recent_flow)}")
            
            # Get initial LLM response
            self.logger.info("Getting initial LLM response...")
            
            api_messages = self.memory.to_api_messages()
            self.logger.info(f"Sending {len(api_messages)} messages to API")
            
            # Debug: Compare message counts
            if self.logger._should_log("DEBUG"):
                memory_msg_count = len(all_messages)
                api_msg_count = len(api_messages)
                if memory_msg_count != api_msg_count:
                    self.logger.debug(f"Message count difference: {memory_msg_count} in memory vs {api_msg_count} in API format")
                    # Log transformation details if there's a difference
                    if isinstance(self.memory, SummaryConversation):
                        self.logger.debug("SummaryConversation may have summarized some messages")
            
            # Measure LLM response time
            start_time = time.time()
            initial_response = await self.get_llm_response(api_messages)
            llm_time = time.time() - start_time
            self.logger.info(f"Got initial response in {llm_time:.2f} seconds ({len(initial_response)} chars)")
            
            # Extract and execute any tool calls in the response
            self.logger.info("Processing tool calls...")
            
            # Check response for potential tool calls before processing
            if self.logger._should_log("DEBUG"):
                if "```tool" in initial_response:
                    potential_tool_count = initial_response.count("```tool")
                    self.logger.debug(f"Detected {potential_tool_count} potential tool calls")
                else:
                    self.logger.debug("No potential tool calls detected in response")
            
            # Measure tool execution time
            start_time = time.time()
            final_response = await self._extract_and_execute_tools(initial_response)
            tool_time = time.time() - start_time
            self.logger.info(f"Tool execution completed in {tool_time:.2f} seconds")
            
            # Debug: Analyze if response changed after tool execution
            if self.logger._should_log("DEBUG"):
                if initial_response == final_response:
                    self.logger.debug("Response unchanged after tool execution (no tools executed)")
                else:
                    length_diff = len(final_response) - len(initial_response)
                    self.logger.debug(f"Response changed after tool execution: {length_diff:+d} chars")
            
            # Add assistant message to memory
            self.logger.info("Adding assistant response to memory")
            assistant_msg = Message("assistant", final_response)
            
            start_time = time.time()
            if isinstance(self.memory, SummaryConversation):
                await self.memory.add_message(assistant_msg)
            else:
                self.memory.add_message(assistant_msg)
            memory_update_time = time.time() - start_time
            
            # Debug: log performance metrics
            if self.logger._should_log("DEBUG"):
                self.logger.debug(f"Memory update (assistant message) completed in {memory_update_time:.4f}s")
                
                # Log overall performance metrics
                total_time = llm_time + tool_time + memory_update_time
                self.logger.debug(f"Performance breakdown: LLM: {llm_time:.2f}s ({llm_time/total_time*100:.1f}%), " +
                                 f"Tools: {tool_time:.2f}s ({tool_time/total_time*100:.1f}%), " +
                                 f"Memory: {memory_update_time:.4f}s ({memory_update_time/total_time*100:.1f}%)")
            
            self.logger.info("Message processing complete")
            
            return final_response
            
        except Exception as e:
            error_msg = f"Error in process_message: {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg)
            
            # Enhanced error logging
            import traceback
            tb = traceback.format_exc()
            self.logger.error(f"Traceback: {tb}")
            
            # Debug: log context about the error
            if self.logger._should_log("DEBUG"):
                # Try to determine where in the process the error occurred
                if 'initial_response' not in locals():
                    self.logger.debug("Error occurred before getting LLM response")
                elif 'final_response' not in locals():
                    self.logger.debug("Error occurred during tool execution")
                else:
                    self.logger.debug("Error occurred after tool execution")
                
                # Log memory state if relevant to the error
                if "memory" in str(e) or "SummaryConversation" in str(tb):
                    if hasattr(self, 'memory'):
                        memory_type = type(self.memory).__name__
                        msg_count = len(self.memory.get_messages()) if hasattr(self.memory, 'get_messages') else "unknown"
                        self.logger.debug(f"Memory context: type={memory_type}, message_count={msg_count}")
            
            return f"Error processing message: {str(e)}"
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.logger.info("Resetting conversation...")
        
        self.memory.clear()
        
        # Re-add system message with tool instructions
        tool_instructions = self._generate_tool_instructions()
        full_system_prompt = f"{self.system_prompt}\n\n{tool_instructions}"
        
        self.logger.info("Adding system message with tool instructions")
        if isinstance(self.memory, SummaryConversation):
            # For summary memory, we need to use the async add_message method
            asyncio.run(self.memory.add_message(Message("system", full_system_prompt)))
        else:
            self.memory.add_message(Message("system", full_system_prompt))
        
        self.logger.info("Conversation reset complete")
    
    def save_conversation(self, file_path: str) -> None:
        """Save the current conversation to a file.
        
        Args:
            file_path: Path to save the conversation to
        """
        self.logger.info(f"Saving conversation to {file_path}")
        
        if isinstance(self.memory, FileConversation):
            # For FileConversation, the conversation is already saved
            self.logger.info(f"Conversation is already saved to {self.memory.file_path}")
            return
        
        try:
            # Get all messages
            messages = self.memory.get_messages()
            
            self.logger.info(f"Saving conversation with {len(messages)} messages")
            
            # Convert to serializable format
            data = [msg.to_dict() for msg in messages]
            
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Conversation saved successfully to {file_path}")
            
        except Exception as e:
            error_msg = f"Error saving conversation: {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg)
            print(error_msg)

    def debug_state(self) -> str:
        """Generate a comprehensive debug report of the agent's current state.
        
        Returns:
            A multi-line string with detailed debugging information
        """
        try:
            debug_info = []
            debug_info.append("===== DATA ANALYSIS AGENT DEBUG STATE =====")
            
            # Agent configuration
            debug_info.append("\n----- CONFIGURATION -----")
            debug_info.append(f"Data Directory: {self.data_dir}")
            debug_info.append(f"Visualizations Directory: {self.visualizations_dir}")
            debug_info.append(f"LLM Provider: {self.llm_config['provider']}")
            debug_info.append(f"LLM Model: {self.llm_config['model']}")
            debug_info.append(f"Temperature: {self.llm_config['temperature']}")
            
            # Memory state
            debug_info.append("\n----- MEMORY STATE -----")
            memory_type = type(self.memory).__name__
            debug_info.append(f"Memory Type: {memory_type}")
            
            messages = self.memory.get_messages()
            debug_info.append(f"Total Messages: {len(messages)}")
            
            # Count messages by role
            role_counts = {}
            for msg in messages:
                role = msg.role
                role_counts[role] = role_counts.get(role, 0) + 1
            
            debug_info.append(f"Message Counts by Role: {role_counts}")
            
            # Recent message summary
            if messages:
                debug_info.append("\n----- RECENT MESSAGES -----")
                recent_msgs = messages[-min(3, len(messages)):]
                for i, msg in enumerate(recent_msgs):
                    content = msg.content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    debug_info.append(f"Message {len(messages) - len(recent_msgs) + i + 1} ({msg.role}): {content}")
            
            # Tool registry state
            debug_info.append("\n----- TOOL REGISTRY -----")
            tools = list(self.tool_registry.tools.keys())
            debug_info.append(f"Registered Tools: {', '.join(tools)}")
            
            # Dataset info
            debug_info.append("\n----- LOADED DATASETS -----")
            if hasattr(DataLoadTool, '_dataframes'):
                datasets = DataLoadTool.list_dataframes()
                if datasets:
                    for ds_name in datasets:
                        df = DataLoadTool.get_dataframe(ds_name)
                        if df is not None:
                            debug_info.append(f"Dataset '{ds_name}': {df.shape[0]} rows × {df.shape[1]} columns")
                else:
                    debug_info.append("No datasets loaded")
            else:
                debug_info.append("No datasets loaded")
            
            # Log file information
            debug_info.append("\n----- LOGGING -----")
            if hasattr(self.logger, 'log_path') and self.logger.log_path:
                debug_info.append(f"Log file: {self.logger.log_path}")
            debug_info.append(f"Log level: {self.logger.log_level}")
            
            # System information
            debug_info.append("\n----- SYSTEM INFO -----")
            import platform
            debug_info.append(f"Python version: {platform.python_version()}")
            debug_info.append(f"OS: {platform.system()} {platform.release()}")
            
            # Pandas/NumPy versions
            import pandas as pd
            import numpy as np
            debug_info.append(f"Pandas version: {pd.__version__}")
            debug_info.append(f"NumPy version: {np.__version__}")
            
            debug_info.append("\n=======================================")
            
            # Log the debug state
            full_debug_report = "\n".join(debug_info)
            self.logger.data("Agent Debug State", full_debug_report)
            
            return full_debug_report
            
        except Exception as e:
            error_msg = f"Error generating debug state: {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg)
            return f"Error generating debug state: {str(e)}"

    def debug_data_summary(self, dataset_name: Optional[str] = None) -> str:
        """Generate a detailed summary of loaded datasets.
        
        Args:
            dataset_name: Optional name of a specific dataset to analyze. If None, summarize all datasets.
            
        Returns:
            A multi-line string with detailed dataset information
        """
        try:
            summary = []
            summary.append("===== DATASET ANALYSIS SUMMARY =====")
            
            # Check if any datasets are loaded
            if not hasattr(DataLoadTool, '_dataframes') or not DataLoadTool._dataframes:
                return "No datasets currently loaded."
            
            # Get list of datasets
            available_datasets = DataLoadTool.list_dataframes()
            
            # If a specific dataset was requested
            if dataset_name:
                # Look for the dataset (case-insensitive)
                found = False
                for name in available_datasets:
                    if name.lower() == dataset_name.lower():
                        datasets_to_analyze = [name]
                        found = True
                        break
                
                if not found:
                    return f"Dataset '{dataset_name}' not found. Available datasets: {', '.join(available_datasets)}"
            else:
                datasets_to_analyze = available_datasets
            
            # Analyze each dataset
            for ds_name in datasets_to_analyze:
                df = DataLoadTool.get_dataframe(ds_name)
                if df is None:
                    summary.append(f"\nDataset '{ds_name}': Unable to access dataset")
                    continue
                
                summary.append(f"\n----- DATASET: {ds_name} -----")
                
                # Basic info
                summary.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                summary.append(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB")
                
                # Column information
                summary.append("\nColumns:")
                for col in df.columns:
                    col_type = str(df[col].dtype)
                    non_null = df[col].count()
                    null_count = df[col].isna().sum()
                    null_pct = null_count / len(df) * 100 if len(df) > 0 else 0
                    
                    # Get unique values info
                    unique_count = df[col].nunique()
                    unique_pct = unique_count / non_null * 100 if non_null > 0 else 0
                    
                    summary.append(f"  - {col} ({col_type}): {non_null} non-null, {null_count} null ({null_pct:.1f}%), {unique_count} unique values ({unique_pct:.1f}%)")
                
                # Numeric column statistics
                num_cols = df.select_dtypes(include=['number']).columns
                if not num_cols.empty:
                    summary.append("\nNumeric Column Statistics:")
                    stats = df[num_cols].describe().transpose()
                    for col in stats.index:
                        stat_info = f"  - {col}: min={stats.loc[col, 'min']:.2f}, max={stats.loc[col, 'max']:.2f}, mean={stats.loc[col, 'mean']:.2f}, std={stats.loc[col, 'std']:.2f}"
                        summary.append(stat_info)
                
                # Categorical column values
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                if not cat_cols.empty:
                    summary.append("\nCategorical Column Top Values:")
                    for col in cat_cols[:min(5, len(cat_cols))]:  # Limit to first 5 categorical columns
                        val_counts = df[col].value_counts().head(3)  # Top 3 values
                        if not val_counts.empty:
                            values_str = ", ".join([f"'{val}': {count}" for val, count in val_counts.items()])
                            summary.append(f"  - {col}: {values_str}")
                
                # Sample data
                summary.append("\nSample Data (first 3 rows):")
                sample = df.head(3).to_string()
                summary.append(sample)
            
            full_summary = "\n".join(summary)
            self.logger.data("Dataset Analysis Summary", full_summary)
            
            return full_summary
            
        except Exception as e:
            error_msg = f"Error generating dataset summary: {type(e).__name__}: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            tb = traceback.format_exc()
            self.logger.error(f"Traceback: {tb}")
            return f"Error generating dataset summary: {str(e)}"


async def run_interactive_session(agent: DataAnalysisAgent):
    """Run an interactive session with the data analysis agent.
    
    Args:
        agent: The data analysis agent
    """
    logger = get_logger()
    logger.info("Starting interactive session")
    
    print("\n===== Data Analysis Agent Interactive Session =====")
    print("Type 'exit', 'quit', or use Ctrl+C to end the session.")
    print("Type 'reset' to reset the conversation.")
    print("Type 'save [filename]' to save the conversation.")
    print("Type 'debug' to print detailed debug information.")
    print("Type 'data-debug [dataset_name]' to analyze loaded datasets.")
    print("===================================================\n")
    
    print("Welcome to the Data Analysis Agent! I can help you analyze and visualize data.")
    print("You can start by loading a dataset with the data_load tool.")
    print("Example datasets in the data directory:")
    
    # List example datasets
    try:
        data_files = [f for f in os.listdir(agent.data_dir) if f.endswith(('.csv', '.json', '.xlsx'))]
        if data_files:
            file_list = ", ".join(data_files)
            logger.info(f"Available datasets: {file_list}")
            print(file_list)
            
            # Add a note about case sensitivity for column names
            print("\nNOTE: When accessing columns in datasets, make sure to use the exact column names")
            print("as they appear in the file (case-sensitive, including underscores).")
            print("For example, use 'marketing_spend' rather than 'Marketing Spend'.")
        else:
            logger.info("No datasets found")
            print("No datasets found. You can place CSV, JSON, or Excel files in the data directory.")
    except Exception as e:
        logger.error(f"Could not list data files: {str(e)}")
        print(f"Could not list data files: {e}")
    
    print("")
    
    try:
        while True:
            # Get user input
            user_input = input("\nUser: ")
            logger.info(f"User input: {user_input}")
            
            # Check for exit commands
            if user_input.lower() in ["exit", "quit"]:
                logger.info("User requested to exit")
                print("Goodbye!")
                break
            
            # Check for reset command
            if user_input.lower() == "reset":
                logger.info("User requested to reset conversation")
                agent.reset_conversation()
                print("Conversation reset.")
                continue
            
            # Check for debug command
            if user_input.lower() == "debug":
                logger.info("User requested debug information")
                debug_info = agent.debug_state()
                print("\n" + debug_info)
                continue
            
            # Check for data-debug command
            if user_input.lower().startswith("data-debug"):
                logger.info("User requested dataset debug information")
                parts = user_input.split(maxsplit=1)
                dataset_name = parts[1].strip() if len(parts) > 1 else None
                
                if dataset_name:
                    logger.debug(f"Analyzing specific dataset: {dataset_name}")
                else:
                    logger.debug("Analyzing all loaded datasets")
                    
                data_summary = agent.debug_data_summary(dataset_name)
                print("\n" + data_summary)
                continue
            
            # Check for save command
            if user_input.lower().startswith("save "):
                filename = user_input[5:].strip()
                if not filename:
                    logger.warning("Save command with no filename")
                    print("Please specify a filename.")
                    continue
                
                if not filename.endswith(".json"):
                    filename += ".json"
                
                logger.info(f"User requested to save conversation as {filename}")
                conversations_dir = Path(__file__).parent / "conversations"
                conversations_dir.mkdir(exist_ok=True)
                file_path = conversations_dir / filename
                
                agent.save_conversation(file_path)
                print(f"Conversation saved to {file_path}")
                continue
            
            # Process the message
            logger.info("Processing user message")
            print("Processing...")
            response = await agent.process_message(user_input)
            
            # Print the response
            logger.info("Displaying response to user")
            print(f"\nAssistant: {response}")
    
    except KeyboardInterrupt:
        logger.info("Session interrupted by user (KeyboardInterrupt)")
        print("\nGoodbye!")
    except Exception as e:
        error_msg = f"Error in interactive session: {type(e).__name__}: {str(e)}"
        logger.error(error_msg)
        print(f"Error: {e}")
        import traceback
        tb = traceback.format_exc()
        logger.error(f"Traceback: {tb}")


def parse_args():
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run a data analysis agent")
    
    # Model and provider options
    parser.add_argument(
        "--provider", "-p", 
        choices=["openai", "anthropic"], 
        default="openai",
        help="LLM provider (openai or anthropic)"
    )
    parser.add_argument(
        "--model", "-m", 
        default="gpt-3.5-turbo",
        help="Model name (e.g. gpt-3.5-turbo, gpt-4, claude-3-opus-20240229)"
    )
    parser.add_argument(
        "--temperature", "-t", 
        type=float,
        default=0.7,
        help="Temperature for LLM sampling"
    )
    
    # Memory options
    parser.add_argument(
        "--memory", 
        choices=["in_memory", "file", "summary"], 
        default="in_memory",
        help="Memory type to use for conversation history"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="File path for file memory (only used if memory is 'file')"
    )
    
    # Data options
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        help="Directory containing data files"
    )
    
    # Visualization options
    parser.add_argument(
        "--vis-dir", "-v",
        type=str,
        help="Directory for saving visualization images"
    )
    
    # Debug and logging options
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug output"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level"
    )
    
    return parser.parse_args()


async def main():
    """Run the data analysis agent."""
    args = parse_args()
    
    # Setup logger
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    log_level = "DEBUG" if args.debug else args.log_level
    logger = setup_logger(log_dir=log_dir, log_to_console=True, log_level=log_level)
    
    logger.info("========== STARTING DATA ANALYSIS AGENT ==========")
    logger.info(f"Command-line arguments: {vars(args)}")
    
    # Create the LLM config
    llm_config = {
        "provider": args.provider,
        "model": args.model,
        "temperature": args.temperature,
    }
    logger.info(f"LLM configuration: {llm_config}")
    
    # Set up file path for file memory
    file_path = None
    if args.memory == "file":
        if args.file:
            file_path = args.file
        else:
            conversations_dir = Path(__file__).parent / "conversations"
            conversations_dir.mkdir(exist_ok=True)
            file_path = conversations_dir / "conversation.json"
        
        logger.info(f"Using file memory with path: {file_path}")
    
    # Set up data_dir
    data_dir = args.data_dir or os.path.join(os.path.dirname(__file__), "data")
    logger.info(f"Using data directory: {data_dir}")
    
    # Set up visualizations directory
    visualizations_dir = args.vis_dir or os.path.join(os.path.dirname(__file__), "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    logger.info(f"Visualizations will be saved to {visualizations_dir}")
    
    # Create the data tool registry
    logger.info("Creating tool registry")
    tool_registry = ToolRegistry()
    
    # Add data analysis tools with the correct data_dir
    tool_registry.register_tool(DataLoadTool(data_dir=data_dir))
    tool_registry.register_tool(DataQueryTool())
    tool_registry.register_tool(DataStatsTool())
    tool_registry.register_tool(DataVisualizeTool(save_dir=visualizations_dir))
    
    # Add basic tools
    tool_registry.register_tool(CalculatorTool())
    tool_registry.register_tool(DateTimeTool())
    
    logger.info(f"Registered tools: {', '.join(tool_registry.tools.keys())}")
    
    # Create the agent
    logger.info("Creating Data Analysis Agent")
    agent = DataAnalysisAgent(
        memory_type=args.memory,
        file_path=file_path,
        data_dir=data_dir,
        vis_dir=visualizations_dir,
        system_prompt="You are a data analysis assistant that helps analyze and visualize data. You have access to tools for data manipulation, statistical analysis, and visualization.",
        llm_config=llm_config,
        tool_registry=tool_registry,
        verbose=args.debug,
        log_level=log_level,
    )
    
    # Run the interactive session
    logger.info("Starting interactive session")
    await run_interactive_session(agent)
    
    logger.info("Session ended")


if __name__ == "__main__":
    asyncio.run(main()) 