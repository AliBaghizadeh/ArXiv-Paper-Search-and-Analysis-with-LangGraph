"""
LangGraph workflow for ArXiv research.

This module implements a structured workflow with LangGraph for research tasks.
"""

from typing import TypedDict, List, Dict, Any, Sequence, Optional, Callable
from langgraph.graph import StateGraph

from arxiv_search.data_collection import arxiv_api_search, process_paper
from arxiv_search.query_expansion import expand_research_query, extract_domain_specific_terms, create_search_query
from arxiv_search.vector_search import VectorDatabase

class RAGState(TypedDict):
    """Type definition for the RAG state."""
    
    query: str
    expanded_query: str
    search_results: List[Dict[str, Any]]
    analysis: str
    embedding_results: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]]
    messages: List[Dict[str, Any]]

def expand_query_node(state: RAGState, model: Any) -> RAGState:
    """
    LangGraph node that expands the query using the model.
    
    Args:
        state: The current LangGraph state
        model: Generative AI model instance
        
    Returns:
        Updated state with expanded query
    """
    query = state["query"]
    expanded_query = expand_research_query(query, model)
    
    return {"expanded_query": expanded_query}

def search_papers_node(state: RAGState, vector_db: VectorDatabase, debug: bool = False) -> RAGState:
    """
    LangGraph node that searches for papers based on the query.
    
    Uses the original and expanded query to search arXiv for relevant
    papers, processes them, and stores the results in both the regular
    database and the vector database.
    
    Args:
        state: The current LangGraph state
        vector_db: Vector database instance
        debug: Whether to print debug information
        
    Returns:
        Updated state with search results
    """
    if debug:
        print("SEARCH NODE - Input state keys:", list(state.keys()))
    
    query = state["query"]
    expanded_query = state["expanded_query"]
    
    # Extract actual search terms from expanded query
    # First, check if expanded query starts with "Query:" (which would indicate formatting issue)
    if "Query:" in expanded_query and "Expanded:" in expanded_query:
        # Extract just the expansion part
        expanded_query = expanded_query.split("Expanded:")[1].strip()
    
    # Extract domain-specific terms
    domain_terms = extract_domain_specific_terms(query, expanded_query)
    
    # Log the detected terms
    if debug:
        if domain_terms:
            print(f"Detected domain terms: {domain_terms}")
        else:
            print("No specific domain terms detected, using original query only")
    
    # Create a clean search query by combining original and domain terms
    search_query = create_search_query(query, domain_terms)
    
    if debug:
        print(f"Clean search query: {search_query}")
    
    # Regular search via arXiv API
    papers = arxiv_api_search(search_query)
    if debug:
        print(f"Found {len(papers)} papers via API search")
    
    # Process and store papers
    results = []
    for paper in papers:
        paper_info = process_paper(paper)
        
        # Add to vector database
        vector_db.add_paper(paper_info)
        
        results.append(paper_info)
    
    if debug:
        print(f"Processed {len(results)} papers for state")
    
    # Now perform semantic search
    semantic_results = vector_db.search(query)
    if debug:
        print(f"Found {len(semantic_results)} papers via semantic search")
    
    # Combine results but prioritize semantic search results
    combined_results = []
    
    # First add semantic results
    paper_ids_added = set()
    for paper in semantic_results:
        combined_results.append(paper)
        paper_ids_added.add(paper["paper_id"])
    
    # Then add any API results not already included
    for paper in results:
        if paper["paper_id"] not in paper_ids_added:
            combined_results.append(paper)
            paper_ids_added.add(paper["paper_id"])
    
    return {
        "search_results": combined_results,
        "embedding_results": semantic_results
    }

def analyze_papers_node(state: RAGState, model: Any, debug: bool = False) -> RAGState:
    """
    LangGraph node that analyzes papers based on the query.
    
    Generates a comprehensive analysis of the retrieved papers
    in relation to the original query.
    
    Args:
        state: The current LangGraph state
        model: Generative AI model instance
        debug: Whether to print debug information
        
    Returns:
        Updated state with paper analysis
    """
    query = state["query"]
    papers = state["search_results"]
    
    # Format paper information for analysis
    top_papers = papers[:5]  # Analyze top 5 papers
    
    few_shot_examples = """
    Example 1:
    Papers:
    1. "Attention Is All You Need" - Introduced the transformer architecture relying entirely on attention mechanisms without recurrence or convolutions.
    2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" - Proposed bidirectional training for transformers using masked language modeling.
    
    Analysis:
    These papers represent seminal work in transformer architectures for NLP. "Attention Is All You Need" established the foundation with the original transformer design using multi-head self-attention. BERT built upon this by introducing bidirectional context modeling and masked language modeling for pre-training, significantly advancing performance on downstream tasks. Key themes include attention mechanisms, pre-training objectives, and the importance of training methodology.
    
    Example 2:
    Query: "How does the Less-Attention Vision Transformer architecture address the computational inefficiencies and saturation problems of traditional Vision Transformers?"
    Papers: 
    1. "You Only Need Less Attention at Each Stage in Vision Transformers" - Proposed reusing early-layer attention scores through linear transformations to reduce computational costs.
    
    Analysis:
    Less-Attention Vision Transformer reduces ViT's quadratic attention cost by reusing early-layer attention scores through linear transformations. It also mitigates attention saturation using residual downsampling and a custom loss to preserve attention structure. This approach addresses two key limitations of traditional Vision Transformers: computational inefficiency due to quadratic complexity of self-attention, and the saturation problem where attention maps become increasingly similar in deeper layers.
    """
    
    # Format paper information
    paper_info = "\n".join([
        f"{i+1}. \"{p['title']}\" - {p['abstract'][:200]}..." 
        for i, p in enumerate(top_papers)
    ])
    
    prompt = f"""Based on the examples below, analyze the following research papers related to "{query}" to identify key technical contributions, methodologies, and how they address specific challenges:

    {few_shot_examples}
    
    Papers:
    {paper_info}
    
    Analysis:"""
    
    response = model.generate_content(prompt, generation_config={"temperature": 1.0})
    
    analysis = response.text
    
    return {"analysis": analysis}

def create_research_workflow(model: Any, api_key: str, debug: bool = False) -> StateGraph:
    """
    Create the LangGraph workflow for research.
    
    Args:
        model: Generative AI model instance
        api_key: Google API key
        debug: Whether to print debug information
        
    Returns:
        StateGraph workflow
    """
    # Initialize vector database
    vector_db = VectorDatabase(api_key=api_key, model=model, debug=debug)
    
    # Create the graph
    workflow = StateGraph(RAGState)
    
    # Define the nodes
    workflow.add_node("expand_query", lambda state: expand_query_node(state, model))
    workflow.add_node("search_papers", lambda state: search_papers_node(state, vector_db, debug))
    workflow.add_node("analyze_papers", lambda state: analyze_papers_node(state, model, debug))
    
    # Define the edges
    workflow.add_edge("expand_query", "search_papers")
    workflow.add_edge("search_papers", "analyze_papers")
    
    # Set the entry point
    workflow.set_entry_point("expand_query")
    
    return workflow

def research_arxiv_langgraph(query: str, model: Any, api_key: str, debug: bool = False) -> Dict[str, Any]:
    """
    Execute the full research workflow using LangGraph.
    
    Args:
        query: The research query
        model: Generative AI model instance
        api_key: Google API key
        debug: Whether to print debug information
        
    Returns:
        Final state with search results and analysis
    """
    # Create initial state
    initial_state = {
        "query": query,
        "expanded_query": "",
        "context": {},
        "messages": [],
        "search_results": [],
        "analysis": "",
        "embedding_results": []
    }
    
    # Create workflow
    workflow = create_research_workflow(model, api_key, debug)
    
    # Compile workflow
    app = workflow.compile()
    
    # Execute workflow
    if debug:
        print("==================================================")
        print("arXiv Research Pipeline with Embedding-Based Search")
        print("==================================================\n")
        print("Searching arXiv and analyzing papers...")
    
    # Run the workflow
    results = app.invoke(initial_state)
    
    if debug:
        print(f"Final state keys: {list(results.keys())}\n")
        print("Displaying research results:\n")
    
    return results

def display_langgraph_results(results: Dict[str, Any]) -> None:
    """
    Display the results of the LangGraph workflow in a notebook.
    
    Args:
        results: The final state from the research workflow
    """
    from IPython.display import display, Markdown
    
    display(Markdown("### QUERY EXPANSION"))
    display(Markdown(results["expanded_query"]))
    
    display(Markdown("### RESEARCH ANALYSIS"))
    display(Markdown(results["analysis"]))
    
    # Show vector-based results first
    if "embedding_results" in results and results["embedding_results"]:
        display(Markdown("### TOP PAPERS (SEMANTIC SEARCH)"))
        display(Markdown(f"**Found {len(results['embedding_results'])} papers via semantic search.**"))
        
        for i, paper in enumerate(results["embedding_results"][:3]):
            paper_md = f"""
                        **{i+1}. {paper['title']}**

                        *Authors:* {paper['authors']}

                        *Published:* {paper['published']}

                        *URL:* {paper['url']}

                        *Abstract:* {paper['abstract'][:300]}...

                        ---
                        """
            display(Markdown(paper_md))
    
    # Show all results
    display(Markdown("### ALL TOP PAPERS"))
    
    if "search_results" not in results or not results["search_results"]:
        display(Markdown("**No papers found in search results.**"))
    else:
        display(Markdown(f"**Found {len(results['search_results'])} papers total.**"))
        for i, paper in enumerate(results["search_results"][:5]):
            paper_md = f"""
            **{i+1}. {paper['title']}**

            *Authors:* {paper['authors']}

            *Published:* {paper['published']}

            *URL:* {paper['url']}

            *Abstract:* {paper['abstract'][:300]}...

            ---
            """
            display(Markdown(paper_md)) 