"""
Utility functions for ArXiv search and analysis.

This module contains helper functions for various tasks.
"""

import os
import time
from typing import Any, Dict, List, Optional

def get_api_key() -> str:
    """
    Get the Google API key from environment variable or .env file.
    
    Returns:
        The Google API key
    """
    # Check environment variable first
    api_key = os.environ.get("GOOGLE_API_KEY")
    
    # If not found, try loading from .env file
    if not api_key:
        try:
            # Try to load from .env file
            from dotenv import load_dotenv
            
            # Look for .env file in the current directory and one level up
            if os.path.exists(".env"):
                load_dotenv(".env")
            elif os.path.exists("../.env"):
                load_dotenv("../.env")
            
            # Try getting the API key again after loading .env
            api_key = os.environ.get("GOOGLE_API_KEY")
        except ImportError:
            # python-dotenv not installed
            pass
    
    # If still not found, check for a config.ini file
    if not api_key:
        try:
            import configparser
            config = configparser.ConfigParser()
            
            # Look for config.ini in current directory and one level up
            if os.path.exists("config.ini"):
                config.read("config.ini")
                if "DEFAULT" in config and "GOOGLE_API_KEY" in config["DEFAULT"]:
                    api_key = config["DEFAULT"]["GOOGLE_API_KEY"]
            elif os.path.exists("../config.ini"):
                config.read("../config.ini")
                if "DEFAULT" in config and "GOOGLE_API_KEY" in config["DEFAULT"]:
                    api_key = config["DEFAULT"]["GOOGLE_API_KEY"]
        except:
            # Ignore any errors with config parsing
            pass
    
    # If still not found, raise error
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please provide it by one of these methods:\n"
            "1. Set the GOOGLE_API_KEY environment variable\n"
            "2. Create a .env file with GOOGLE_API_KEY=your_key\n"
            "3. Create a config.ini file with [DEFAULT] section and GOOGLE_API_KEY=your_key\n"
            "\nTo use .env files, install python-dotenv: pip install python-dotenv"
        )
    
    return api_key

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (Exception,),
):
    """
    Retry a function with exponential backoff.
    
    Args:
        func: The function to retry
        initial_delay: Initial delay in seconds
        exponential_base: Base of the exponential backoff
        jitter: Whether to add jitter to the delay
        max_retries: Maximum number of retries
        errors: Tuple of errors to catch and retry
        
    Returns:
        The result of the function
    """
    import random
    import time
    
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        
        # Loop until a successful response or max_retries is hit
        while True:
            try:
                return func(*args, **kwargs)
            
            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1
                
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    ) from e
                
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                
                # Sleep for the delay
                time.sleep(delay)
    
    return wrapper

def format_timestamp() -> str:
    """
    Generate a formatted timestamp.
    
    Returns:
        A formatted timestamp string
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: The text to truncate
        max_length: Maximum length of the text
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
        
    # Find the last space before max_length
    last_space = text[:max_length].rfind(" ")
    if last_space == -1:
        return text[:max_length] + "..."
        
    return text[:last_space] + "..."

def format_paper_citation(paper: Dict[str, Any]) -> str:
    """
    Format a paper citation in a readable format.
    
    Args:
        paper: Paper information dictionary
        
    Returns:
        Formatted citation
    """
    authors = paper["authors"].split(", ")
    if len(authors) > 3:
        authors = f"{authors[0]} et al."
    else:
        authors = ", ".join(authors)
        
    year = paper["published"].split("-")[0]
    
    return f"{authors} ({year}). \"{paper['title']}\". arXiv:{paper['paper_id']}."

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    
    This is a rough estimation based on the average of 4 characters per token.
    
    Args:
        text: The text to estimate
        
    Returns:
        Estimated number of tokens
    """
    # A very simple estimation: ~4 chars per token for English text
    return len(text) // 4 
