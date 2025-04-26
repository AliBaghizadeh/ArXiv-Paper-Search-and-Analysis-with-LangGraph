"""
Simple example of using the ArXiv search package.
"""

import os
import sys
import google.generativeai as genai

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arxiv_search.langgraph_workflow import research_arxiv_langgraph, display_langgraph_results
from arxiv_search.utils import get_api_key

def main():
    """Run a simple example search."""
    # Set up API key
    try:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("GOOGLE_API_KEY environment variable not set.")
            print("Please set it with your Google API key.")
            return 1
    except Exception as e:
        print(f"Error getting API key: {e}")
        return 1
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Initialize model
    model = genai.GenerativeModel(
        "models/gemini-1.5-pro-latest",
        generation_config={"temperature": 1.0}
    )
    
    # Define a research query
    query = "what are vision transformers?"
    
    print(f"Running research query: {query}")
    print("This may take a few minutes...")
    
    # Run the research workflow
    results = research_arxiv_langgraph(
        query=query,
        model=model,
        api_key=api_key,
        debug=True
    )
    
    # Display the results
    print("\nResults:")
    
    print("\nQuery Expansion:")
    print(results["expanded_query"][:200] + "...")
    
    print("\nResearch Analysis:")
    print(results["analysis"][:200] + "...")
    
    print("\nTop Papers:")
    for i, paper in enumerate(results["search_results"][:3]):
        print(f"\n{i+1}. {paper['title']}")
        print(f"   Authors: {paper['authors']}")
        print(f"   Published: {paper['published']}")
        print(f"   URL: {paper['url']}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 