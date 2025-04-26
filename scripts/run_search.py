#!/usr/bin/env python
"""
Command-line interface for running ArXiv search and analysis.
"""

import argparse
import os
import sys
import google.generativeai as genai
from rich.console import Console
from rich.markdown import Markdown

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arxiv_search.langgraph_workflow import research_arxiv_langgraph
from arxiv_search.utils import get_api_key

def main():
    """Run the ArXiv search and analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Search and analyze ArXiv papers with LangGraph and Gemini."
    )
    parser.add_argument(
        "query", type=str, help="Research query to search for"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug output"
    )
    parser.add_argument(
        "--api-key", type=str, help="Google API key (overrides environment variable)"
    )
    parser.add_argument(
        "--model", type=str, default="models/gemini-1.5-pro-latest",
        help="Model to use (default: gemini-1.5-pro-latest)"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Temperature for generation (default: 1.0)"
    )
    parser.add_argument(
        "--save", type=str, help="Save results to file"
    )
    
    args = parser.parse_args()
    
    console = Console()
    
    # Get API key
    try:
        api_key = args.api_key if args.api_key else get_api_key()
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Initialize model
    try:
        model = genai.GenerativeModel(
            args.model,
            generation_config={"temperature": args.temperature}
        )
    except Exception as e:
        console.print(f"[bold red]Error initializing model:[/bold red] {e}")
        return 1
    
    # Run search
    console.print(f"[bold]Searching ArXiv for:[/bold] {args.query}")
    console.print("This may take a few minutes...")
    
    try:
        results = research_arxiv_langgraph(
            query=args.query,
            model=model,
            api_key=api_key,
            debug=args.debug
        )
    except Exception as e:
        console.print(f"[bold red]Error during search:[/bold red] {e}")
        return 1
    
    # Display results
    console.print("\n[bold green]Results:[/bold green]\n")
    
    console.print("[bold]QUERY EXPANSION:[/bold]")
    console.print(Markdown(results["expanded_query"]))
    
    console.print("\n[bold]RESEARCH ANALYSIS:[/bold]")
    console.print(Markdown(results["analysis"]))
    
    # Show vector-based results first
    if "embedding_results" in results and results["embedding_results"]:
        console.print(f"\n[bold]TOP PAPERS (SEMANTIC SEARCH):[/bold]")
        console.print(f"Found {len(results['embedding_results'])} papers via semantic search.")
        
        for i, paper in enumerate(results["embedding_results"][:3]):
            console.print(f"\n[bold]{i+1}. {paper['title']}[/bold]")
            console.print(f"Authors: {paper['authors']}")
            console.print(f"Published: {paper['published']}")
            console.print(f"URL: {paper['url']}")
            console.print(f"Abstract: {paper['abstract'][:300]}...")
    
    # Show all results
    console.print(f"\n[bold]ALL TOP PAPERS:[/bold]")
    
    if "search_results" not in results or not results["search_results"]:
        console.print("No papers found in search results.")
    else:
        console.print(f"Found {len(results['search_results'])} papers total.")
        for i, paper in enumerate(results["search_results"][:5]):
            console.print(f"\n[bold]{i+1}. {paper['title']}[/bold]")
            console.print(f"Authors: {paper['authors']}")
            console.print(f"Published: {paper['published']}")
            console.print(f"URL: {paper['url']}")
            console.print(f"Abstract: {paper['abstract'][:300]}...")
    
    # Save results if requested
    if args.save:
        import json
        
        # Prepare results for saving (remove any non-serializable objects)
        save_results = {
            "query": args.query,
            "expanded_query": results["expanded_query"],
            "analysis": results["analysis"],
            "search_results": [{
                "paper_id": p["paper_id"],
                "title": p["title"],
                "authors": p["authors"],
                "published": p["published"],
                "url": p["url"],
                "abstract": p["abstract"]
            } for p in results["search_results"][:10]]
        }
        
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        
        console.print(f"\nResults saved to [bold]{args.save}[/bold]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 