"""
Utility functions for ArXiv search and analysis.

This module contains helper functions for various tasks.
"""

import os
import time
from typing import Any, Dict, List, Optional

def get_api_key() -> str:
    """
    Get the Google API key from environment variable.
    
    Returns:
        The Google API key
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please set it with your Google API key."
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