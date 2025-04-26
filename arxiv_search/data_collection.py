"""
Data collection module for ArXiv papers.

This module handles ArXiv API search, HTML content extraction, and paper caching.
"""

import arxiv
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
import time

# Cache for storing paper information
papers_db = {}

def check_html_available(paper_id: str) -> bool:
    """
    Check if HTML version of a paper is available.
    
    HTML versions are typically available for papers published after Dec 2023.
    
    Args:
        paper_id: ArXiv paper ID
        
    Returns:
        True if HTML version is available, False otherwise
    """
    try:
        html_url = f"https://arxiv.org/html/{paper_id}"
        response = requests.head(html_url, timeout=5)
        return response.status_code == 200
    except Exception:
        return False

def get_html_content(paper_id: str) -> str:
    """
    Get the HTML content of a paper.
    
    Args:
        paper_id: ArXiv paper ID
        
    Returns:
        Extracted text content from HTML
    """
    try:
        # Fetch HTML
        html_url = f"https://arxiv.org/html/{paper_id}"
        response = requests.get(html_url, timeout=10)
        
        if response.status_code != 200:
            return ""
        
        # Parse HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.extract()
        
        # Find the main content
        main_content = soup.find("main")
        if main_content:
            content = main_content.get_text(separator=" ", strip=True)
        else:
            # Try to get the body if main tag isn't available
            content = soup.body.get_text(separator=" ", strip=True)
        
        # Clean up whitespace
        content = " ".join(content.split())
        
        return content
    except Exception as e:
        if hasattr(e, "__name__"):
            print(f"Error getting HTML content: {e.__name__}: {e}")
        else:
            print(f"Error getting HTML content: {type(e).__name__}: {e}")
        return ""

def arxiv_api_search(query: str, max_results: int = 20) -> List[Any]:
    """
    Search ArXiv API for papers matching the query.
    
    Args:
        query: Search query
        max_results: Maximum number of papers to return
        
    Returns:
        List of ArXiv paper objects
    """
    # Check if query is specifically about vision
    is_vision_query = any(term in query.lower() for term in 
                          ["vision", "image", "visual", "computer vision", "vit"])
    
    # Set up ArXiv client
    client = arxiv.Client()
    
    # Create ArXiv search
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    # Execute search
    results = list(client.results(search))
    
    # If this is a vision-related query and we didn't get many CS.CV papers,
    # try again with a more specific query for the Computer Vision category
    if is_vision_query:
        # Count CS.CV papers
        cv_papers = sum(1 for paper in results 
                         if any(cat.startswith('cs.CV') for cat in paper.categories))
        
        # If we don't have enough vision papers, search specifically in CS.CV
        if cv_papers < max_results // 2:
            # Add CS.CV category constraint
            specific_search = arxiv.Search(
                query=f"cat:cs.CV AND ({query})",
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # Execute the constrained search
            specific_results = list(client.results(specific_search))
            
            # If we got results, replace our original results
            if specific_results:
                # Combine results, prioritizing CS.CV matches
                # First add the CS.CV papers
                filtered_results = specific_results
                
                # Then add any non-CS.CV papers from the original results
                # until we reach max_results
                original_non_cv = [paper for paper in results 
                                   if not any(cat.startswith('cs.CV') for cat in paper.categories)]
                
                remaining_slots = max_results - len(filtered_results)
                if remaining_slots > 0 and original_non_cv:
                    filtered_results.extend(original_non_cv[:remaining_slots])
                
                results = filtered_results
    
    return results

def process_paper(paper: Any) -> Dict[str, Any]:
    """
    Process an ArXiv paper into a standardized format.
    
    Args:
        paper: ArXiv paper object
        
    Returns:
        Dictionary with paper information
    """
    paper_id = paper.entry_id.split('/')[-1]
    
    # Skip if already in database
    if paper_id in papers_db:
        return papers_db[paper_id]
        
    paper_info = {
        "paper_id": paper_id,
        "title": paper.title,
        "authors": ", ".join(author.name for author in paper.authors),
        "published": paper.published.strftime("%Y-%m-%d"),
        "url": paper.entry_id,
        "abstract": paper.summary,
        "has_html": check_html_available(paper_id)
    }
    
    # Get HTML content if available
    if paper_info["has_html"]:
        paper_info["content"] = get_html_content(paper_id)
    else:
        paper_info["content"] = paper_info["abstract"]
        
    # Store in our database
    papers_db[paper_id] = paper_info
    
    return paper_info 