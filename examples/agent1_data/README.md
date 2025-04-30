# Data Analysis Agent

This example demonstrates a specialized agent for data analysis tasks, built on the foundations of the QnA agent. The Data Analysis Agent can load datasets, perform statistical analysis, create visualizations, and answer questions about data.

## Features

- **Data Loading**: Load data from CSV, JSON, Excel, and other file formats
- **Data Querying**: Analyze data using pandas operations
- **Statistical Analysis**: Compute summary statistics, correlations, and more
- **Data Visualization**: Create bar charts, line plots, scatter plots, and other visualizations
- **Memory Management**: Maintain conversation context for better follow-up questions
- **Flexible Configuration**: Support for different LLM providers (OpenAI, Anthropic)

## Prerequisites

Before running this example, make sure you have:

1. Set up the development environment as described in the main README
2. Configured your API keys for the LLM provider (OpenAI or Anthropic) in your `.env` file
3. Installed the required dependencies:
   - pandas
   - numpy
   - matplotlib
   - openpyxl (for Excel file support)

You can install these with:

```bash
pip install pandas numpy matplotlib openpyxl
```

## Structure

This example includes the following files:

- `data_analysis_agent.py`: The main implementation of the Data Analysis Agent
- `cli_interface.py`: Command-line interface for interacting with the agent
- `data/`: Directory to store datasets for analysis
- `conversations/`: Directory where conversation histories can be saved

## Running the Agent

You can run the Data Analysis Agent via the command-line interface:

```bash
# Using OpenAI (default)
python examples/agent2_data/cli_interface.py

# Using Anthropic
python examples/agent2_data/cli_interface.py --provider anthropic --model claude-3-sonnet-20240229

# With custom configuration
python examples/agent2_data/cli_interface.py --temperature 0.3 --memory summary --verbose
```

### Command-line Options

- `--provider`: LLM provider (`openai` or `anthropic`)
- `--model`: Model name (e.g., `gpt-3.5-turbo`, `claude-3-sonnet-20240229`)
- `--temperature`: Temperature for sampling (default: 0.2)
- `--memory`: Type of memory to use (`in_memory`, `file`, or `summary`)
- `--file`: Path to conversation file (for file memory)
- `--data-dir`: Directory containing data files
- `--verbose`: Enable verbose output

## Using the Agent

Once the agent is running, you can interact with it to analyze data:

1. **Load a dataset**:
   ```
   Please load the sales_data.csv file
   ```

2. **Explore the data**:
   ```
   What columns are in this dataset? Can you show me some summary statistics?
   ```

3. **Perform analysis**:
   ```
   What's the correlation between sales and marketing spend?
   ```

4. **Create visualizations**:
   ```
   Create a bar chart of sales by region
   ```

The agent will use the appropriate tools to fulfill your requests and explain the results.

## Creating Your Own Data Analysis Agent

You can use the Data Analysis Agent as a building block for your own applications:

```python
from examples.agent2_data.data_analysis_agent import DataAnalysisAgent
from src.tools.data_tools import create_data_tool_registry

# Create the agent
agent = DataAnalysisAgent(
    memory_type="in_memory",
    data_dir="path/to/data/directory",
    llm_config={
        "provider": "openai",
        "model": "gpt-4",
        "temperature": 0.2,
    },
    tool_registry=create_data_tool_registry(),
    verbose=True,
)

# Process a user message
response = await agent.process_message("Load the customer_data.csv file and show me the top 5 rows")
print(response)
```

## Extending the Agent

You can extend the Data Analysis Agent with additional capabilities:

- Add more specialized statistical analysis tools
- Integrate with more data sources (databases, APIs)
- Add support for different visualization types
- Implement machine learning capabilities for predictive analytics
- Add custom data processing pipelines

## Next Steps

After exploring the Data Analysis Agent, you can move on to the Task Automation Agent example, which builds on these concepts to create an agent that can automate various tasks on your system. 