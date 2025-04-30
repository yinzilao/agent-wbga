"""
Basic Tools Module

This module provides simple tools that can be used by LLM agents, including:
- Web search
- Calculator
- Date and time utilities
"""

import datetime
import math
import json
import re
import time
from typing import Dict, List, Any, Optional, Union, Callable
import asyncio
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# Import logger if available
try:
    from src.utils.logger import get_logger
    logger = get_logger()
    has_logger = True
except ImportError:
    has_logger = False
    logger = None


class ToolError(Exception):
    """Exception raised when a tool encounters an error."""
    pass


class Tool:
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str):
        """Initialize a tool.
        
        Args:
            name: The name of the tool
            description: A description of what the tool does
        """
        self.name = name
        self.description = description
        if has_logger:
            logger.debug(f"Initialized tool: {name}")
    
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the tool.
        
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Tool subclasses must implement __call__")
    
    def to_dict(self) -> Dict[str, str]:
        """Convert tool to a dictionary for API tool definitions."""
        return {
            "name": self.name,
            "description": self.description
        }


class CalculatorTool(Tool):
    """A tool for performing mathematical calculations."""
    
    def __init__(self):
        """Initialize the calculator tool."""
        super().__init__(
            name="calculator",
            description="Performs mathematical calculations. Input should be a valid mathematical expression."
        )
    
    async def __call__(self, expression: str) -> Union[float, str]:
        """Evaluate a mathematical expression.
        
        Args:
            expression: A string containing a mathematical expression
        
        Returns:
            The result of the calculation
        
        Raises:
            ToolError: If the expression is invalid or unsafe
        """
        if has_logger:
            logger.info(f"CalculatorTool executing with expression: {expression}")
        
        # Clean the expression
        expression = expression.replace('^', '**')
        
        # Safety check
        if any(keyword in expression for keyword in 
               ['import', 'eval', 'exec', 'compile', 'globals', 'locals', 
                'getattr', 'setattr', 'os.', 'sys.', 'open', '__']):
            if has_logger:
                logger.error(f"Calculator safety check failed for expression: {expression}")
            raise ToolError("Expression contains unsafe operations.")
        
        try:
            start_time = time.time()
            # Restrict to safe operations
            allowed_names = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'len': len,
                'int': int, 'float': float, 'str': str,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'asin': math.asin, 'acos': math.acos, 'atan': math.atan,
                'sqrt': math.sqrt, 'log': math.log, 'log10': math.log10,
                'exp': math.exp, 'pi': math.pi, 'e': math.e
            }
            
            # Use restricted eval
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            
            # Format result nicely
            if isinstance(result, (int, float)):
                if result.is_integer():
                    final_result = int(result)
                else:
                    final_result = round(result, 10)  # Avoid floating point precision issues
                    
            else:
                final_result = str(result)
                
            elapsed_time = time.time() - start_time
            if has_logger:
                logger.info(f"CalculatorTool computed {expression} = {final_result} in {elapsed_time:.4f}s")
                
            return final_result
        
        except Exception as e:
            if has_logger:
                logger.error(f"CalculatorTool error: {str(e)}")
            raise ToolError(f"Error evaluating expression: {str(e)}")


class DateTimeTool(Tool):
    """A tool for working with dates and times."""
    
    def __init__(self):
        """Initialize the date/time tool."""
        super().__init__(
            name="datetime",
            description="Provides current date and time information or performs date calculations. Commands: 'now', 'today', 'date [YYYY-MM-DD]', 'time_diff [date1] [date2]'"
        )
    
    async def __call__(self, command: str) -> str:
        """Execute a date/time related command.
        
        Args:
            command: A command string like 'now', 'today', 'date 2023-12-31', or 'time_diff 2023-01-01 2023-12-31'
        
        Returns:
            Date/time information as a string
        
        Raises:
            ToolError: If the command is invalid
        """
        command = command.strip().lower()
        if has_logger:
            logger.info(f"DateTimeTool executing with command: {command}")
        
        try:
            start_time = time.time()
            
            if command == 'now':
                # Current date and time
                now = datetime.datetime.now()
                result = now.strftime('%Y-%m-%d %H:%M:%S')
            
            elif command == 'today':
                # Current date
                today = datetime.date.today()
                result = today.strftime('%Y-%m-%d')
            
            elif command.startswith('date '):
                # Parse date
                date_str = command[5:].strip()
                try:
                    # Parse ISO format (YYYY-MM-DD)
                    date = datetime.date.fromisoformat(date_str)
                    
                    # Return formatted date with weekday
                    result = f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')})"
                except ValueError:
                    if has_logger:
                        logger.error(f"DateTimeTool invalid date format: {date_str}")
                    raise ToolError(f"Invalid date format. Use YYYY-MM-DD.")
            
            elif command.startswith('time_diff '):
                # Calculate difference between two dates
                params = command[10:].strip().split()
                if len(params) != 2:
                    if has_logger:
                        logger.error(f"DateTimeTool invalid time_diff params: {command[10:]}")
                    raise ToolError("time_diff requires two dates in YYYY-MM-DD format")
                
                try:
                    date1 = datetime.date.fromisoformat(params[0])
                    date2 = datetime.date.fromisoformat(params[1])
                    
                    diff = date2 - date1
                    
                    result = f"Difference: {abs(diff.days)} days"
                except ValueError:
                    if has_logger:
                        logger.error(f"DateTimeTool invalid date format in time_diff: {params}")
                    raise ToolError("Invalid date format. Use YYYY-MM-DD for both dates.")
            
            else:
                if has_logger:
                    logger.error(f"DateTimeTool unknown command: {command}")
                raise ToolError(f"Unknown datetime command: {command}")
            
            elapsed_time = time.time() - start_time
            if has_logger:
                logger.info(f"DateTimeTool completed command '{command}' with result '{result}' in {elapsed_time:.4f}s")
            
            return result
        
        except Exception as e:
            if isinstance(e, ToolError):
                raise
            if has_logger:
                logger.error(f"DateTimeTool error: {str(e)}")
            raise ToolError(f"Error processing datetime command: {str(e)}")


class WebSearchTool(Tool):
    """A tool for searching the web."""
    
    def __init__(self, api_key: str = None, custom_search_id: str = None, http_client: httpx.AsyncClient = None):
        """Initialize the web search tool.
        
        Args:
            api_key: Google Custom Search API key (if using Google)
            custom_search_id: Google Custom Search Engine ID (if using Google)
            http_client: Optional HTTP client to use for requests
        """
        super().__init__(
            name="web_search",
            description="Searches the web for information. Input should be a search query."
        )
        self.api_key = api_key
        self.custom_search_id = custom_search_id
        self.http_client = http_client or httpx.AsyncClient(timeout=30.0)
        if has_logger:
            logger.debug(f"WebSearchTool initialized with API key: {'set' if api_key else 'not set'}")
    
    async def __call__(self, query: str, num_results: int = 3) -> str:
        """Perform a web search.
        
        Args:
            query: The search query
            num_results: Number of results to return
        
        Returns:
            Search results as a string
        
        Raises:
            ToolError: If the search fails
        """
        if has_logger:
            logger.info(f"WebSearchTool executing with query: {query}, num_results: {num_results}")
        
        try:
            start_time = time.time()
            if self.api_key and self.custom_search_id:
                if has_logger:
                    logger.debug(f"Using Google Custom Search API for query: {query}")
                result = await self._google_search(query, num_results)
            else:
                if has_logger:
                    logger.debug(f"Using DuckDuckGo search fallback for query: {query}")
                result = await self._ddg_search(query, num_results)
                
            elapsed_time = time.time() - start_time
            if has_logger:
                logger.info(f"WebSearchTool completed query in {elapsed_time:.2f}s, returned {len(result.split('Title:')) - 1} results")
                
            return result
        
        except Exception as e:
            if has_logger:
                logger.error(f"WebSearchTool error: {str(e)}")
            raise ToolError(f"Error performing web search: {str(e)}")
    
    async def _google_search(self, query: str, num_results: int) -> str:
        """Perform a search using Google Custom Search API.
        
        Args:
            query: The search query
            num_results: Number of results to return
        
        Returns:
            Search results as a string
        """
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": self.api_key,
            "cx": self.custom_search_id,
            "q": query,
            "num": min(num_results, 10)  # API limit is 10
        }
        
        if has_logger:
            logger.debug(f"Sending request to Google API with params: {params}")
            
        response = await self.http_client.get(url, params=params)
        if response.status_code != 200:
            if has_logger:
                logger.error(f"Google search API error: status code {response.status_code}, response: {response.text[:200]}")
            raise ToolError(f"Google search API returned status code {response.status_code}")
        
        data = response.json()
        if "items" not in data:
            if has_logger:
                logger.warning(f"No results found for query: {query}")
            return "No results found."
        
        results = []
        for item in data["items"][:num_results]:
            title = item.get("title", "No title")
            link = item.get("link", "No link")
            snippet = item.get("snippet", "No description")
            
            results.append(f"Title: {title}\nURL: {link}\nDescription: {snippet}\n")
        
        return "\n".join(results)
    
    async def _ddg_search(self, query: str, num_results: int) -> str:
        """Perform a search using DuckDuckGo (HTML parsing).
        
        This is a fallback method that doesn't require API keys.
        
        Args:
            query: The search query
            num_results: Number of results to return
        
        Returns:
            Search results as a string
        """
        # Use DuckDuckGo HTML search as a fallback
        # Note: This is not an official API and may break
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        }
        escaped_query = query.replace(" ", "+")
        url = f"https://html.duckduckgo.com/html/?q={escaped_query}"
        
        if has_logger:
            logger.debug(f"Sending request to DuckDuckGo with URL: {url}")
            
        response = await self.http_client.get(url, headers=headers, follow_redirects=True)
        if response.status_code != 200:
            if has_logger:
                logger.error(f"DuckDuckGo search error: status code {response.status_code}")
            raise ToolError(f"Search request failed with status code {response.status_code}")
        
        # Parse HTML
        soup = BeautifulSoup(response.text, "html.parser")
        results_elements = soup.select(".result")
        
        if not results_elements:
            if has_logger:
                logger.warning(f"No results found for query: {query}")
            return "No results found."
        
        results = []
        for i, result in enumerate(results_elements[:num_results]):
            title_element = result.select_one(".result__title")
            link_element = result.select_one(".result__url")
            snippet_element = result.select_one(".result__snippet")
            
            title = title_element.get_text().strip() if title_element else "No title"
            
            if link_element:
                link = link_element.get("href", "")
                if link.startswith("/"):
                    link = "https://duckduckgo.com" + link
            else:
                link = "No link"
                
            snippet = snippet_element.get_text().strip() if snippet_element else "No description"
            
            results.append(f"Title: {title}\nURL: {link}\nDescription: {snippet}\n")
        
        return "\n".join(results)


class ToolRegistry:
    """Registry for managing and accessing tools."""
    
    def __init__(self):
        """Initialize an empty tool registry."""
        self.tools: Dict[str, Tool] = {}
        if has_logger:
            logger.debug("Initialized ToolRegistry")
    
    def register_tool(self, tool: Tool) -> None:
        """Register a tool.
        
        Args:
            tool: The tool to register
        """
        self.tools[tool.name] = tool
        if has_logger:
            logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name.
        
        Args:
            name: The name of the tool
        
        Returns:
            The tool, or None if not found
        """
        tool = self.tools.get(name)
        if has_logger:
            if tool:
                logger.debug(f"Tool found: {name}")
            else:
                logger.warning(f"Tool not found: {name}")
        return tool
    
    def list_tools(self) -> List[Dict[str, str]]:
        """Get a list of all registered tools.
        
        Returns:
            List of tool dictionaries with name and description
        """
        tools_list = [tool.to_dict() for tool in self.tools.values()]
        if has_logger:
            logger.debug(f"Listed {len(tools_list)} tools")
        return tools_list
    
    def get_tool_descriptions(self) -> str:
        """Get a formatted string of tool descriptions.
        
        Returns:
            Newline-separated list of tool descriptions
        """
        descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools.values()])
        if has_logger:
            logger.debug(f"Generated tool descriptions ({len(descriptions)} chars)")
        return descriptions
    
    async def execute_tool(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a tool by name.
        
        Args:
            name: The name of the tool
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool
        
        Returns:
            The result of the tool execution
        
        Raises:
            ToolError: If the tool is not found or execution fails
        """
        tool = self.get_tool(name)
        if not tool:
            if has_logger:
                logger.error(f"Tool not found for execution: {name}")
            raise ToolError(f"Tool not found: {name}")
        
        try:
            if has_logger:
                arg_str = str(args[0]) if args else ""
                logger.info(f"Executing tool {name} with args: {arg_str}")
                
            start_time = time.time()
            result = await tool(*args, **kwargs)
            elapsed_time = time.time() - start_time
            
            if has_logger:
                result_str = str(result)
                if len(result_str) > 100:
                    result_sample = result_str[:100] + "..."
                else:
                    result_sample = result_str
                logger.info(f"Tool {name} executed successfully in {elapsed_time:.2f}s with result: {result_sample}")
            
            return result
        except Exception as e:
            if has_logger:
                logger.error(f"Error executing tool {name}: {type(e).__name__}: {str(e)}")
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"Error executing tool {name}: {str(e)}")


# Convenience function to create a registry with basic tools
def create_basic_tool_registry() -> ToolRegistry:
    """Create a registry with basic tools.
    
    Returns:
        A tool registry with calculator and datetime tools
    """
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    registry.register_tool(DateTimeTool())
    if has_logger:
        logger.info("Created basic tool registry with calculator and datetime tools")
    return registry


# Example of usage
async def main():
    """Demo the basic tools."""
    # Create tools
    calculator = CalculatorTool()
    datetime_tool = DateTimeTool()
    web_search = WebSearchTool()
    
    # Create registry
    registry = ToolRegistry()
    registry.register_tool(calculator)
    registry.register_tool(datetime_tool)
    registry.register_tool(web_search)
    
    # Test calculator
    try:
        result = await calculator("2 + 2 * 3")
        print(f"Calculator: 2 + 2 * 3 = {result}")
        
        result = await calculator("sin(pi/2)")
        print(f"Calculator: sin(pi/2) = {result}")
    except ToolError as e:
        print(f"Calculator error: {e}")
    
    # Test datetime
    try:
        result = await datetime_tool("now")
        print(f"DateTime: now = {result}")
        
        result = await datetime_tool("time_diff 2023-01-01 2023-12-31")
        print(f"DateTime: time_diff = {result}")
    except ToolError as e:
        print(f"DateTime error: {e}")
    
    # Test web search
    try:
        result = await web_search("Python programming language")
        print(f"Web Search results:\n{result}")
    except ToolError as e:
        print(f"Web Search error: {e}")
    
    # Test registry
    try:
        result = await registry.execute_tool("calculator", "1 + 1")
        print(f"Registry - Calculator: 1 + 1 = {result}")
        
        result = await registry.execute_tool("datetime", "today")
        print(f"Registry - DateTime: today = {result}")
    except ToolError as e:
        print(f"Registry error: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 