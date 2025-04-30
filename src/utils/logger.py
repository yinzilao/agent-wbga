"""
Logging Utility for AI Agent

This module provides a centralized logging system for AI agents to track workflow steps,
inputs, and outputs for educational purposes.
"""

import os
import sys
import time
import datetime
from pathlib import Path
from typing import Optional, TextIO, Dict, Any, Union

class AgentLogger:
    """Logger for AI agents that writes to both console and file."""
    
    def __init__(
        self, 
        log_dir: str = "logs",
        log_to_console: bool = True,
        log_level: str = "INFO"
    ):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to save log files
            log_to_console: Whether to print logs to console
            log_level: Minimum log level to record (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = Path(log_dir)
        self.log_to_console = log_to_console
        self.log_level = log_level.upper()
        self.log_file: Optional[TextIO] = None
        self.log_path: Optional[Path] = None
        
        # Log levels
        self.levels = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3
        }
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a new log file
        self._create_log_file()
    
    def _create_log_file(self):
        """Create a new log file with timestamp in the name."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"agent_log_{timestamp}.log"
        
        if self.log_file:
            self.log_file.close()
        
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.info(f"Log file created at {self.log_path}")
    
    def _should_log(self, level: str) -> bool:
        """Check if a message at the given level should be logged."""
        return self.levels.get(level.upper(), 0) >= self.levels.get(self.log_level, 0)
    
    def _write(self, level: str, message: str):
        """Write a message to the log file and optionally to console."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_line = f"[{timestamp}] [{level}] {message}"
        
        # Write to file
        if self.log_file:
            self.log_file.write(log_line + "\n")
            self.log_file.flush()
        
        # Write to console
        if self.log_to_console:
            print(log_line)
    
    def debug(self, message: str):
        """Log a debug message."""
        if self._should_log("DEBUG"):
            self._write("DEBUG", message)
    
    def info(self, message: str):
        """Log an info message."""
        if self._should_log("INFO"):
            self._write("INFO", message)
    
    def warning(self, message: str):
        """Log a warning message."""
        if self._should_log("WARNING"):
            self._write("WARNING", message)
    
    def error(self, message: str):
        """Log an error message."""
        if self._should_log("ERROR"):
            self._write("ERROR", message)
    
    def data(self, title: str, data: Union[str, Dict, list]):
        """Log data with a title for clear structure."""
        if self._should_log("DEBUG"):
            if isinstance(data, (dict, list)):
                import json
                data_str = json.dumps(data, indent=2)
            else:
                data_str = str(data)
            
            # Format with clear separators
            separator = "-" * 40
            message = f"\n{separator}\n{title}\n{separator}\n{data_str}\n{separator}"
            self._write("DATA", message)
    
    def close(self):
        """Close the log file."""
        if self.log_file:
            self.log_file.close()
            self.log_file = None

# Global logger instance
_logger: Optional[AgentLogger] = None

def get_logger() -> AgentLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = AgentLogger()
    return _logger

def setup_logger(
    log_dir: str = "logs",
    log_to_console: bool = True,
    log_level: str = "INFO"
) -> AgentLogger:
    """Setup the global logger with custom settings.
    
    Args:
        log_dir: Directory to save log files
        log_to_console: Whether to print logs to console
        log_level: Minimum log level to record
        
    Returns:
        Configured logger instance
    """
    global _logger
    if _logger:
        _logger.close()
    
    _logger = AgentLogger(
        log_dir=log_dir,
        log_to_console=log_to_console,
        log_level=log_level
    )
    
    return _logger 