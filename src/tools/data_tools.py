"""
Data Analysis Tools Module

This module provides tools for data analysis tasks, including:
- Data loading and manipulation
- Statistical analysis
- Data visualization
"""

import os
import json
import csv
import asyncio
import time
from typing import Dict, List, Any, Optional, Union, Callable
from pathlib import Path
import base64
from io import BytesIO
import datetime

# Import data analysis libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from .basic_tools import Tool, ToolError, ToolRegistry

# Import logger if available
try:
    from src.utils.logger import get_logger
    logger = get_logger()
    has_logger = True
except ImportError:
    has_logger = False
    logger = None


class DataLoadTool(Tool):
    """A tool for loading data from various file formats."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the data loading tool.
        
        Args:
            data_dir: Optional directory where data files are stored
        """
        super().__init__(
            name="data_load",
            description="Load data from CSV, JSON, or other file formats. Format: 'data_load: filename.ext' or 'data_load: path/to/file.ext'"
        )
        self.data_dir = data_dir or os.getcwd()
        if has_logger:
            logger.info(f"DataLoadTool initialized with data_dir: {self.data_dir}")
    
    async def __call__(self, file_path: str) -> str:
        """Load data from the specified file.
        
        Args:
            file_path: Path to the data file, relative to data_dir if not absolute
        
        Returns:
            Summary of the loaded data
        
        Raises:
            ToolError: If the file cannot be loaded
        """
        if has_logger:
            logger.info(f"DataLoadTool called with file_path: {file_path}")
        
        try:
            # Resolve the file path
            if os.path.isabs(file_path):
                full_path = file_path
            else:
                full_path = os.path.join(self.data_dir, file_path)
            
            if has_logger:
                logger.debug(f"Resolved file path: {full_path}")
            
            # Check if file exists
            if not os.path.exists(full_path):
                if has_logger:
                    logger.error(f"File not found: {full_path}")
                raise ToolError(f"File not found: {file_path}")
            
            # Load based on file extension
            ext = os.path.splitext(full_path)[1].lower()
            
            if has_logger:
                logger.info(f"Loading file with extension: {ext}")
                start_time = time.time()
            
            if ext == '.csv':
                if has_logger:
                    logger.debug(f"Loading CSV file: {full_path}")
                df = pd.read_csv(full_path)
                # Store in memory for later use
                self._store_dataframe(df, file_path)
                summary = self._summarize_dataframe(df)
            
            elif ext == '.json':
                if has_logger:
                    logger.debug(f"Loading JSON file: {full_path}")
                df = pd.read_json(full_path)
                self._store_dataframe(df, file_path)
                summary = self._summarize_dataframe(df)
            
            elif ext == '.xlsx' or ext == '.xls':
                if has_logger:
                    logger.debug(f"Loading Excel file: {full_path}")
                df = pd.read_excel(full_path)
                self._store_dataframe(df, file_path)
                summary = self._summarize_dataframe(df)
            
            elif ext == '.txt':
                # Try to determine format and load
                if has_logger:
                    logger.debug(f"Loading TXT file, attempting to determine format: {full_path}")
                try:
                    df = pd.read_csv(full_path, sep='\t')
                    self._store_dataframe(df, file_path)
                    summary = self._summarize_dataframe(df)
                except:
                    with open(full_path, 'r') as f:
                        data = f.read()
                    summary = f"Loaded text file: {len(data)} characters"
                    if has_logger:
                        logger.debug(f"Loaded text file with {len(data)} characters")
            
            else:
                if has_logger:
                    logger.error(f"Unsupported file format: {ext}")
                raise ToolError(f"Unsupported file format: {ext}")
            
            if has_logger:
                elapsed_time = time.time() - start_time
                logger.info(f"File loaded successfully in {elapsed_time:.2f}s")
                logger.debug(f"DataFrame shape: {df.shape if 'df' in locals() else 'N/A'}")
            
            return summary
        
        except Exception as e:
            if has_logger:
                logger.error(f"Error loading data: {type(e).__name__}: {str(e)}")
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"Error loading data: {str(e)}")
    
    def _store_dataframe(self, df: pd.DataFrame, identifier: str):
        """Store the DataFrame in a global location for other tools to access.
        
        Args:
            df: The DataFrame to store
            identifier: A name to identify this DataFrame
        """
        # Create data store if it doesn't exist
        if not hasattr(DataLoadTool, '_dataframes'):
            DataLoadTool._dataframes = {}
        
        # Store with a clean identifier
        key = os.path.basename(identifier).split('.')[0]
        DataLoadTool._dataframes[key] = df
        
        if has_logger:
            logger.info(f"Stored DataFrame '{key}' with shape {df.shape}")
    
    @staticmethod
    def get_dataframe(identifier: str) -> Optional[pd.DataFrame]:
        """Get a stored DataFrame by identifier.
        
        Args:
            identifier: The name of the DataFrame
        
        Returns:
            The DataFrame, or None if not found
        """
        if not hasattr(DataLoadTool, '_dataframes'):
            if has_logger:
                logger.warning("No DataFrames have been loaded yet")
            return None
        
        # Try to get by exact key
        if identifier in DataLoadTool._dataframes:
            if has_logger:
                logger.debug(f"Retrieved DataFrame '{identifier}' by exact match")
            return DataLoadTool._dataframes[identifier]
        
        # Try to match partial key
        for key in DataLoadTool._dataframes:
            if identifier in key or key in identifier:
                if has_logger:
                    logger.debug(f"Retrieved DataFrame '{key}' by partial match with '{identifier}'")
                return DataLoadTool._dataframes[key]
        
        if has_logger:
            logger.warning(f"DataFrame '{identifier}' not found in loaded DataFrames")
        return None
    
    @staticmethod
    def list_dataframes() -> List[str]:
        """List all available DataFrames.
        
        Returns:
            List of DataFrame identifiers
        """
        if not hasattr(DataLoadTool, '_dataframes'):
            if has_logger:
                logger.debug("No DataFrames available to list")
            return []
        
        dataframes = list(DataLoadTool._dataframes.keys())
        if has_logger:
            logger.debug(f"Listed {len(dataframes)} available DataFrames: {', '.join(dataframes)}")
        return dataframes
    
    def _summarize_dataframe(self, df: pd.DataFrame) -> str:
        """Create a summary of the DataFrame.
        
        Args:
            df: The DataFrame to summarize
        
        Returns:
            Summary text
        """
        if has_logger:
            logger.debug(f"Generating summary for DataFrame with shape {df.shape}")
        
        summary = []
        summary.append(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")
        summary.append(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Data types
        dtype_counts = df.dtypes.value_counts().to_dict()
        dtype_str = ", ".join([f"{count} {dtype}" for dtype, count in dtype_counts.items()])
        summary.append(f"Data types: {dtype_str}")
        
        # Missing values
        missing = df.isna().sum()
        if missing.sum() > 0:
            missing_cols = [f"{col}: {count}" for col, count in missing[missing > 0].items()]
            summary.append(f"Missing values: {', '.join(missing_cols)}")
        else:
            summary.append("Missing values: None")
        
        # Sample data (first 5 rows)
        sample = df.head(5).to_string()
        summary.append(f"\nSample data (first 5 rows):\n{sample}")
        
        summary_text = "\n".join(summary)
        return summary_text


class DataQueryTool(Tool):
    """A tool for querying and manipulating data with pandas."""
    
    def __init__(self):
        """Initialize the data query tool."""
        super().__init__(
            name="data_query",
            description="Query and manipulate a loaded DataFrame using pandas operations. Format: 'data_query: dataset_name | operation'."
        )
    
    async def __call__(self, query: str) -> str:
        """Execute a data query operation.
        
        Args:
            query: Query in the format "dataset_name | operation"
        
        Returns:
            Result of the query
        
        Raises:
            ToolError: If the query cannot be executed
        """
        if has_logger:
            logger.info(f"DataQueryTool called with query: {query}")
        
        try:
            # Parse the query
            parts = query.split('|', 1)
            if len(parts) != 2:
                if has_logger:
                    logger.error("Invalid query format: Missing separator '|'")
                raise ToolError("Query format should be 'dataset_name | operation'")
            
            dataset_name = parts[0].strip()
            operation = parts[1].strip()
            
            if has_logger:
                logger.debug(f"Parsed query - dataset: '{dataset_name}', operation: '{operation}'")
            
            # Get the DataFrame
            start_time = time.time()
            df = DataLoadTool.get_dataframe(dataset_name)
            if df is None:
                # List available datasets
                available = DataLoadTool.list_dataframes()
                if available:
                    available_str = ", ".join(available)
                    if has_logger:
                        logger.error(f"Dataset '{dataset_name}' not found. Available: {available_str}")
                    raise ToolError(f"Dataset '{dataset_name}' not found. Available datasets: {available_str}")
                else:
                    if has_logger:
                        logger.error(f"Dataset '{dataset_name}' not found. No datasets loaded.")
                    raise ToolError(f"Dataset '{dataset_name}' not found. No datasets loaded yet.")
            
            # Execute the operation
            # Use restricted eval to execute pandas operations
            local_vars = {"df": df, "pd": pd, "np": np}
            
            # Safety check
            unsafe_patterns = ['import', 'eval', 'exec', 'compile', 'globals', 'locals', 
                              'getattr', 'setattr', 'os.', 'sys.', 'open', '__']
            if any(keyword in operation for keyword in unsafe_patterns):
                if has_logger:
                    logger.error(f"Operation contains unsafe patterns: {operation}")
                raise ToolError("Operation contains unsafe functions.")
            
            # Execute the operation
            if has_logger:
                logger.debug(f"Executing pandas operation: df.{operation}")
            
            result_df = eval(f"df.{operation}", {"__builtins__": {}}, local_vars)
            
            # Format the result
            if isinstance(result_df, pd.DataFrame):
                if len(result_df) > 10:
                    result_str = result_df.head(10).to_string()
                    result_text = f"{result_str}\n\n(Showing 10 of {len(result_df)} rows)"
                    if has_logger:
                        logger.debug(f"Query returned DataFrame with {len(result_df)} rows, showing first 10")
                else:
                    result_text = result_df.to_string()
                    if has_logger:
                        logger.debug(f"Query returned DataFrame with {len(result_df)} rows")
            elif isinstance(result_df, pd.Series):
                result_text = result_df.to_string()
                if has_logger:
                    logger.debug(f"Query returned Series with {len(result_df)} values")
            else:
                result_text = str(result_df)
                if has_logger:
                    logger.debug(f"Query returned {type(result_df).__name__}")
            
            elapsed_time = time.time() - start_time
            if has_logger:
                logger.info(f"Query executed successfully in {elapsed_time:.2f}s")
            
            return result_text
        
        except Exception as e:
            if has_logger:
                logger.error(f"Error executing query: {type(e).__name__}: {str(e)}")
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"Error executing query: {str(e)}")


class DataStatsTool(Tool):
    """A tool for statistical analysis of data."""
    
    def __init__(self):
        """Initialize the data statistics tool."""
        super().__init__(
            name="data_stats",
            description="Perform statistical analysis on a dataset. Format: 'data_stats: dataset_name | operation', where operation can be 'summary', 'correlation', 'describe', etc."
        )
    
    async def __call__(self, query: str) -> str:
        """Perform statistical analysis.
        
        Args:
            query: Query in the format "dataset_name | operation"
        
        Returns:
            Statistical results
        
        Raises:
            ToolError: If the analysis cannot be performed
        """
        if has_logger:
            logger.info(f"DataStatsTool called with query: {query}")
        
        try:
            # Parse the query
            parts = query.split('|', 1)
            if len(parts) != 2:
                if has_logger:
                    logger.error("Invalid query format: Missing separator '|'")
                raise ToolError("Query format should be 'dataset_name | operation'")
            
            dataset_name = parts[0].strip()
            operation = parts[1].strip().lower()
            
            if has_logger:
                logger.debug(f"Parsed stats query - dataset: '{dataset_name}', operation: '{operation}'")
            
            # Get the DataFrame
            start_time = time.time()
            df = DataLoadTool.get_dataframe(dataset_name)
            if df is None:
                available = DataLoadTool.list_dataframes()
                if available:
                    available_str = ", ".join(available)
                    if has_logger:
                        logger.error(f"Dataset '{dataset_name}' not found. Available: {available_str}")
                    raise ToolError(f"Dataset '{dataset_name}' not found. Available datasets: {available_str}")
                else:
                    if has_logger:
                        logger.error(f"Dataset '{dataset_name}' not found. No datasets loaded.")
                    raise ToolError(f"Dataset '{dataset_name}' not found. No datasets loaded yet.")
            
            # Execute the statistical operation
            if has_logger:
                logger.debug(f"Executing statistical operation: {operation}")
                
            if operation == 'summary' or operation == 'describe':
                result = df.describe().to_string()
                if has_logger:
                    logger.debug("Generated statistical summary (describe)")
            
            elif operation == 'correlation' or operation == 'corr':
                # Get only numeric columns
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.empty:
                    if has_logger:
                        logger.warning("No numeric columns found for correlation analysis")
                    return "No numeric columns found for correlation analysis."
                result = numeric_df.corr().to_string()
                if has_logger:
                    logger.debug(f"Generated correlation matrix for {len(numeric_df.columns)} numeric columns")
            
            elif operation == 'info':
                buffer = BytesIO()
                df.info(buf=buffer)
                buffer.seek(0)
                result = buffer.read().decode()
                if has_logger:
                    logger.debug("Generated DataFrame info")
            
            elif operation == 'nunique':
                result = df.nunique().to_string()
                if has_logger:
                    logger.debug("Generated unique value counts")
            
            elif operation == 'missing' or operation == 'na':
                missing = df.isna().sum()
                missing_pct = df.isna().mean() * 100
                missing_df = pd.DataFrame({
                    'Count': missing,
                    'Percentage': missing_pct
                })
                result = missing_df.to_string()
                if has_logger:
                    logger.debug("Generated missing value analysis")
            
            else:
                if has_logger:
                    logger.error(f"Unknown statistical operation: {operation}")
                raise ToolError(f"Unknown statistical operation: {operation}")
            
            elapsed_time = time.time() - start_time
            if has_logger:
                logger.info(f"Statistical operation completed in {elapsed_time:.2f}s")
            
            return result
        
        except Exception as e:
            if has_logger:
                logger.error(f"Error performing statistical analysis: {type(e).__name__}: {str(e)}")
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"Error performing statistical analysis: {str(e)}")


class DataVisualizeTool(Tool):
    """A tool for data visualization."""
    
    def __init__(self, save_dir: Optional[str] = None):
        """Initialize the data visualization tool.
        
        Args:
            save_dir: Directory to save visualization images. If None, images won't be saved locally.
        """
        super().__init__(
            name="data_viz",
            description="Create data visualizations. Format: 'data_viz: dataset_name | plot_type | options', where plot_type can be 'bar', 'line', 'scatter', 'hist', etc."
        )
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            if has_logger:
                logger.info(f"DataVisualizeTool initialized with save_dir: {save_dir}")
    
    def _get_column_case_insensitive(self, df: pd.DataFrame, column_name: str) -> Optional[str]:
        """Find the column name in the DataFrame regardless of case sensitivity.
        
        Args:
            df: The DataFrame to search in
            column_name: The column name to find (case insensitive)
            
        Returns:
            The actual column name if found, None otherwise
        """
        if column_name in df.columns:
            return column_name
            
        # Try case-insensitive match
        for col in df.columns:
            if col.lower() == column_name.lower():
                if has_logger:
                    logger.debug(f"Found column '{col}' by case-insensitive match with '{column_name}'")
                return col
        
        if has_logger:
            logger.warning(f"Column '{column_name}' not found in DataFrame (columns: {', '.join(df.columns)})")
        return None
    
    def _generate_filename(self, dataset_name: str, plot_type: str, options: Dict[str, str]) -> str:
        """Generate a filename for the visualization image.
        
        Args:
            dataset_name: Name of the dataset
            plot_type: Type of plot
            options: Visualization options
            
        Returns:
            A filename string
        """
        # Create a timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get plot info
        title = options.get('title', '').replace(' ', '_')
        x_col = options.get('x', '')
        y_col = options.get('y', '')
        
        # Construct filename components
        components = [timestamp, dataset_name, plot_type]
        
        # Add optional components if they exist
        if title:
            components.append(f"title_{title}")
        if x_col:
            components.append(f"x_{x_col}")
        if y_col:
            components.append(f"y_{y_col}")
            
        # Join with underscores and add extension
        filename = f"{'_'.join(components)}.png"
        if has_logger:
            logger.debug(f"Generated visualization filename: {filename}")
        return filename
    
    async def __call__(self, query: str) -> str:
        """Create a data visualization.
        
        Args:
            query: Query in the format "dataset_name | plot_type | options"
        
        Returns:
            Base64-encoded image data
        
        Raises:
            ToolError: If the visualization cannot be created
        """
        if has_logger:
            logger.info(f"DataVisualizeTool called with query: {query}")
        
        try:
            # Parse the query
            parts = query.split('|')
            if len(parts) < 2:
                if has_logger:
                    logger.error("Invalid visualization query format")
                raise ToolError("Query format should be 'dataset_name | plot_type | options'")
            
            dataset_name = parts[0].strip()
            plot_type = parts[1].strip().lower()
            
            # Get optional parameters
            options = {}
            if len(parts) > 2:
                options_str = parts[2].strip()
                # Parse options in key=value format
                for option in options_str.split(','):
                    if '=' in option:
                        key, value = option.split('=', 1)
                        options[key.strip()] = value.strip()
            
            if has_logger:
                logger.debug(f"Parsed viz query - dataset: '{dataset_name}', plot_type: '{plot_type}', options: {options}")
            
            # Get the DataFrame
            start_time = time.time()
            df = DataLoadTool.get_dataframe(dataset_name)
            if df is None:
                available = DataLoadTool.list_dataframes()
                if available:
                    available_str = ", ".join(available)
                    if has_logger:
                        logger.error(f"Dataset '{dataset_name}' not found. Available: {available_str}")
                    raise ToolError(f"Dataset '{dataset_name}' not found. Available datasets: {available_str}")
                else:
                    if has_logger:
                        logger.error(f"Dataset '{dataset_name}' not found. No datasets loaded.")
                    raise ToolError(f"Dataset '{dataset_name}' not found. No datasets loaded yet.")
            
            # Create the plot
            if has_logger:
                logger.debug(f"Creating {plot_type} plot for dataset '{dataset_name}'")
                
            plt.figure(figsize=(10, 6))
            
            # Extract x and y columns if specified (with case-insensitive matching)
            x_col_name = options.get('x', None)
            y_col_name = options.get('y', None)
            
            # Find actual column names (case-insensitive)
            x_col = self._get_column_case_insensitive(df, x_col_name) if x_col_name else None
            y_col = self._get_column_case_insensitive(df, y_col_name) if y_col_name else None
            
            # Validate columns exist
            if x_col_name and not x_col:
                available_cols = ", ".join(df.columns)
                if has_logger:
                    logger.error(f"Column '{x_col_name}' not found. Available columns: {available_cols}")
                raise ToolError(f"Column '{x_col_name}' not found. Available columns: {available_cols}")
            if y_col_name and not y_col:
                available_cols = ", ".join(df.columns)
                if has_logger:
                    logger.error(f"Column '{y_col_name}' not found. Available columns: {available_cols}")
                raise ToolError(f"Column '{y_col_name}' not found. Available columns: {available_cols}")
            
            # Handle different plot types
            if plot_type in ['bar', 'barplot']:
                if has_logger:
                    logger.debug(f"Creating bar plot with x={x_col}, y={y_col}")
                if x_col and y_col:
                    df.plot(kind='bar', x=x_col, y=y_col)
                else:
                    # If no specific columns, use value_counts for a categorical column
                    if x_col:
                        df[x_col].value_counts().plot(kind='bar')
                    else:
                        # Use the first categorical column
                        cat_cols = df.select_dtypes(include=['object', 'category']).columns
                        if not cat_cols.empty:
                            df[cat_cols[0]].value_counts().plot(kind='bar')
                        else:
                            # Or the first column if no categorical columns
                            df.iloc[:, 0].value_counts().plot(kind='bar')
            
            elif plot_type in ['line', 'lineplot']:
                if has_logger:
                    logger.debug(f"Creating line plot with x={x_col}, y={y_col}")
                if x_col and y_col:
                    df.plot(kind='line', x=x_col, y=y_col)
                else:
                    # If no specific columns, use the first suitable columns
                    df.plot(kind='line')
            
            elif plot_type in ['scatter', 'scatterplot']:
                if has_logger:
                    logger.debug(f"Creating scatter plot with x={x_col}, y={y_col}")
                if x_col and y_col:
                    df.plot(kind='scatter', x=x_col, y=y_col)
                else:
                    # Need at least two numeric columns
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) >= 2:
                        df.plot(kind='scatter', x=num_cols[0], y=num_cols[1])
                    else:
                        if has_logger:
                            logger.error("Need at least two numeric columns for scatter plot")
                        raise ToolError("Need at least two numeric columns for scatter plot.")
            
            elif plot_type in ['hist', 'histogram']:
                if has_logger:
                    logger.debug(f"Creating histogram with column={x_col}")
                if x_col:
                    df[x_col].plot(kind='hist')
                else:
                    # Use all numeric columns
                    df.select_dtypes(include=[np.number]).plot(kind='hist')
            
            elif plot_type in ['box', 'boxplot']:
                if has_logger:
                    logger.debug(f"Creating boxplot with column={x_col}")
                if x_col:
                    df[x_col].plot(kind='box')
                else:
                    # Use all numeric columns
                    df.select_dtypes(include=[np.number]).plot(kind='box')
            
            elif plot_type in ['pie', 'pieplot']:
                if has_logger:
                    logger.debug(f"Creating pie chart with column={x_col}")
                if x_col:
                    df[x_col].value_counts().plot(kind='pie')
                else:
                    # Use the first categorical column
                    cat_cols = df.select_dtypes(include=['object', 'category']).columns
                    if not cat_cols.empty:
                        df[cat_cols[0]].value_counts().plot(kind='pie')
                    else:
                        # Or the first column if no categorical columns
                        df.iloc[:, 0].value_counts().plot(kind='pie')
            
            else:
                if has_logger:
                    logger.error(f"Unknown plot type: {plot_type}")
                raise ToolError(f"Unknown plot type: {plot_type}")
            
            # Set title if provided
            if 'title' in options:
                plt.title(options['title'])
                if has_logger:
                    logger.debug(f"Set plot title: {options['title']}")
            
            # Set axis labels if provided
            if 'xlabel' in options:
                plt.xlabel(options['xlabel'])
            if 'ylabel' in options:
                plt.ylabel(options['ylabel'])
            
            # Save the plot to a byte buffer
            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png')
            
            # Save locally if save_dir is specified
            filepath = None
            if self.save_dir is not None:
                filename = self._generate_filename(dataset_name, plot_type, options)
                filepath = os.path.join(self.save_dir, filename)
                plt.savefig(filepath)
                if has_logger:
                    logger.info(f"Saved visualization to {filepath}")
            
            plt.close()
            
            # Simplified output that doesn't include the full base64 data
            if self.save_dir is not None and filepath is not None:
                markdown_output = f"Image created and saved as: {os.path.basename(filepath)}"
            else:
                # If no save directory is specified, we still need to return something
                # Convert to base64 for display but don't include it in the output
                buffer.seek(0)
                image_data = base64.b64encode(buffer.read()).decode()
                markdown_output = "Image created successfully (not saved to disk)"
            
            elapsed_time = time.time() - start_time
            if has_logger:
                logger.info(f"Visualization created in {elapsed_time:.2f}s")
                
            return markdown_output
        
        except Exception as e:
            if has_logger:
                logger.error(f"Error creating visualization: {type(e).__name__}: {str(e)}")
            if isinstance(e, ToolError):
                raise
            raise ToolError(f"Error creating visualization: {str(e)}")


def create_data_tool_registry() -> ToolRegistry:
    """Create a registry with data analysis tools.
    
    Returns:
        A tool registry with data analysis tools
    """
    registry = ToolRegistry()
    registry.register_tool(DataLoadTool())
    registry.register_tool(DataQueryTool())
    registry.register_tool(DataStatsTool())
    registry.register_tool(DataVisualizeTool())
    if has_logger:
        logger.info("Created data tool registry with load, query, stats, and visualization tools")
    return registry


# Example usage
async def main():
    """Demo the data tools."""
    # Create the data tools
    data_load = DataLoadTool()
    data_query = DataQueryTool()
    data_stats = DataStatsTool()
    data_viz = DataVisualizeTool()
    
    # Create sample data for testing
    sample_data = {
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 45],
        'city': ['New York', 'Boston', 'Chicago', 'Denver', 'Seattle'],
        'salary': [50000, 60000, 70000, 80000, 90000]
    }
    
    df = pd.DataFrame(sample_data)
    temp_csv = 'temp_sample.csv'
    df.to_csv(temp_csv, index=False)
    
    try:
        # Test data loading
        result = await data_load(temp_csv)
        print(f"Data Load Result:\n{result}\n")
        
        # Test data query
        result = await data_query("temp_sample | head(3)")
        print(f"Data Query Result:\n{result}\n")
        
        # Test data stats
        result = await data_stats("temp_sample | summary")
        print(f"Data Stats Result:\n{result}\n")
        
        # Test data visualization
        result = await data_viz("temp_sample | bar | x=city,y=salary,title=Salary by City")
        print(f"Data Visualization Result: {result}")
        
    finally:
        # Clean up
        if os.path.exists(temp_csv):
            os.remove(temp_csv)


if __name__ == "__main__":
    asyncio.run(main()) 