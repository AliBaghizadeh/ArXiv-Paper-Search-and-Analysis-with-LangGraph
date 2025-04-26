"""
arXiv Research Pipeline with Embeddings

This module implements an enhanced research pipeline for searching arXiv papers,
analyzing them using Google's Gemini model and embedding-based semantic search,
and presenting the results through a structured workflow built with LangGraph.

The pipeline includes query expansion, paper search with vector embeddings, 
content extraction (HTML/abstract), research analysis, and formatted result presentation.
"""

# ============= Imports and Configuration =============

import os
import arxiv
import requests
import numpy as np
import uuid
from typing import TypedDict, List, Dict, Any, Sequence, Optional, Callable
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# LangGraph imports
from langgraph.graph import StateGraph

# LLM and message handling imports
import google.generativeai as genai
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Vector database imports
import chromadb
from chromadb.utils import embedding_functions


# ============= State and Configuration =============

class RAGState(TypedDict):
    """State for the RAG workflow.
    
    Attributes:
        query: The original user query
        expanded_query: Query after expansion with the LLM
        context: List of contextualized information from papers
        messages: List of chat messages in the conversation
        search_results: List of papers retrieved from arXiv
        analysis: Generated analysis of the papers
        embedding_results: Papers retrieved via vector search
    """
    query: str
    expanded_query: str
    context: List[str]
    messages: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    analysis: str
    embedding_results: List[Dict[str, Any]]


# API Configuration
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "your-api-key-here")
genai.configure(api_key=GOOGLE_API_KEY)

# Create a model instance
model = genai.GenerativeModel('gemini-1.5-pro-latest')

# Create a custom embedding function for ChromaDB
class GoogleEmbeddingFunction(chromadb.utils.embedding_functions.EmbeddingFunction):
    """Custom embedding function that uses Google's Generative AI API."""
    
    def __init__(self, api_key: str, debug: bool = False):
        """Initialize with Google API key.
        
        Args:
            api_key: Google API key
            debug: Whether to print debug information
        """
        self.api_key = api_key
        self.debug = debug
        genai.configure(api_key=api_key)
    
    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for the provided texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            try:
                # Truncate text if too long (API has limits)
                if len(text) > 8000:
                    text = text[:8000]
                
                if self.debug:
                    print(f"Generating embedding for text of length {len(text)}")
                
                # Create embedding request
                embedding_response = genai.embed_content(
                    model="embedding-001",
                    content=text,
                    task_type="retrieval_document",
                )
                
                if self.debug:
                    print(f"Embedding response type: {type(embedding_response)}")
                    print(f"Embedding response attributes: {dir(embedding_response) if hasattr(embedding_response, '__dir__') else 'No attributes'}")
                
                # Try to extract embedding based on different possible response structures
                if hasattr(embedding_response, "embedding"):
                    embedding_vector = embedding_response.embedding
                    if self.debug:
                        print(f"Found embedding with length: {len(embedding_vector)}")
                elif hasattr(embedding_response, "embeddings") and embedding_response.embeddings:
                    embedding_vector = embedding_response.embeddings[0]
                    if self.debug:
                        print(f"Found embedding in embeddings array with length: {len(embedding_vector)}")
                else:
                    # Fallback to dict-style access
                    try:
                        if isinstance(embedding_response, dict) and "embedding" in embedding_response:
                            embedding_vector = embedding_response["embedding"]
                        elif isinstance(embedding_response, dict) and "embeddings" in embedding_response:
                            embedding_vector = embedding_response["embeddings"][0]
                        else:
                            raise ValueError("Could not find embedding in response")
                    except Exception as e:
                        if self.debug:
                            print(f"Dict access failed: {e}")
                        # Use random fallback
                        embedding_vector = np.random.rand(768).tolist()
                        print("Warning: Using random fallback embedding (dict access failed)")
                
                embeddings.append(embedding_vector)
                
            except Exception as e:
                if self.debug:
                    print(f"Error generating embedding (details): {type(e).__name__}: {e}")
                # Fallback: random embeddings
                embeddings.append(np.random.rand(768).tolist())
                print("Warning: Using random fallback embedding due to exception")
                
        return embeddings


# Function to toggle debug output globally
DEBUG_MODE = False

def set_debug_mode(enabled: bool = True):
    """Enable or disable debug output globally.
    
    Args:
        enabled: Whether to enable debug output
    """
    global DEBUG_MODE
    DEBUG_MODE = enabled
    print(f"Debug mode {'enabled' if enabled else 'disabled'}")


# Create a ChromaDB collection
def get_vector_db(debug: bool = False):
    """Get the vector database client and collection.
    
    Creates or retrieves a ChromaDB collection for storing paper embeddings
    using Google's text embeddings model.
    
    Args:
        debug: Whether to print debug information
        
    Returns:
        A tuple of (client, collection)
    """
    client = chromadb.Client()
    
    # Set up our custom embedding function
    embedding_function = GoogleEmbeddingFunction(api_key=GOOGLE_API_KEY, debug=debug)
    
    # Create or get a collection
    collection = client.get_or_create_collection(
        name="arxiv_papers",
        embedding_function=embedding_function
    )
    
    return client, collection

# Initialize the vector database
try:
    client, collection = get_vector_db(debug=DEBUG_MODE)
    print("Successfully initialized vector database")
except Exception as e:
    print(f"Error initializing vector database: {e}")
    # Create a fallback in-memory dictionary to store papers
    # This will allow the pipeline to run without vector search capability
    collection = None
    print("Using fallback storage without vector search capabilities")

# Papers database cache
papers_db = {}


# ============= Query Expansion and Analysis =============

def expand_research_query(query: str) -> str:
    """Expand a research query using few-shot prompting.
    
    Uses domain-specific examples to help the model generate
    a comprehensive expansion of the original query.
    
    Args:
        query: The original research query
        
    Returns:
        An expanded version of the query with additional concepts and terms
    """
    # Vision transformer specific example if the query is about vision transformers
    if "vision transformer" in query.lower() or "vit" in query.lower():
        few_shot_examples = """
        Example 1:
        Query: "vision transformer architecture"
        Expanded: The query is about Vision Transformer (ViT) architectures for computer vision tasks. Key aspects to explore include: original ViT design and patch-based image tokenization; comparison with CNN architectures; attention mechanisms specialized for vision; hierarchical and pyramid vision transformers; efficiency improvements like token pruning and sparse attention; distillation techniques for vision transformers; adaptations for different vision tasks including detection and segmentation; recent innovations addressing quadratic complexity and attention saturation.
        
        Example 2: 
        Query: "how do vision transformers process images"
        Expanded: The query focuses on the internal mechanisms of how Vision Transformers process visual information. Key areas to investigate include: patch embedding processes; position embeddings for spatial awareness; self-attention mechanisms for global context; the role of MLP blocks in feature transformation; how class tokens aggregate information; patch size impact on performance and efficiency; multi-head attention design in vision applications; information flow through vision transformer layers; differences from convolutional approaches to feature extraction.
        """
    else:
        few_shot_examples = """
        Example 1:
        Query: "transformer models for NLP"
        Expanded: The query is about transformer architecture models used in natural language processing. Key aspects to explore include: BERT, GPT, T5, and other transformer variants; attention mechanisms; self-supervision and pre-training approaches; fine-tuning methods; performance on NLP tasks like translation, summarization, and question answering; efficiency improvements like distillation and pruning; recent innovations in transformer architectures.
        
        Example 2:
        Query: "reinforcement learning for robotics"
        Expanded: The query concerns applying reinforcement learning methods to robotic systems. Important areas to investigate include: policy gradient methods; Q-learning variants for continuous control; sim-to-real transfer; imitation learning; model-based RL for robotics; sample efficiency techniques; multi-agent RL for coordinated robots; safety constraints in robotic RL; real-world applications and benchmarks; hierarchical RL for complex tasks.
        
        Example 3:
        Query: "graph neural networks applications"
        Expanded: The query focuses on practical applications of graph neural networks. Key dimensions to explore include: GNN architectures (GCN, GAT, GraphSAGE); applications in chemistry and drug discovery; recommender systems using GNNs; traffic and transportation network modeling; social network analysis; knowledge graph completion; GNNs for computer vision tasks; scalability solutions for large graphs; theoretical foundations of graph representation learning.
        """
    
    prompt = f"""Based on the examples below, expand my research query to identify key concepts, relevant subtopics, and specific areas to explore:

    {few_shot_examples}

    Query: "{query}"
    Expanded:"""
    
    response = model.generate_content(prompt)
    
    return response.text


def analyze_papers(query: str, papers: List[Dict[str, Any]]) -> str:
    """Analyze papers using few-shot prompting with domain-specific examples.
    
    Generates a research analysis based on the retrieved papers and
    the original query using domain-specific examples.
    
    Args:
        query: The original research query
        papers: List of paper dictionaries containing metadata and content
        
    Returns:
        A comprehensive analysis of the papers in relation to the query
    """
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
        for i, p in enumerate(papers[:5])
    ])
    
    prompt = f"""Based on the examples below, analyze the following research papers related to "{query}" to identify key technical contributions, methodologies, and how they address specific challenges:

    {few_shot_examples}
    
    Papers:
    {paper_info}
    
    Analysis:"""
    
    response = model.generate_content(prompt)
    
    return response.text


# ============= arXiv Data Collection Functions =============

def arxiv_api_search(query: str, max_results: int = 20) -> List[Any]:
    """Search arXiv for papers matching the query.
    
    Uses the arXiv API to find relevant papers based on the query,
    with optional filtering for specific categories.
    
    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 20)
        
    Returns:
        List of arxiv.Result objects representing papers
    """
    # Use category filter for more relevant results
    category_filter = "cat:cs.CV" if "vision" in query.lower() or "image" in query.lower() else ""
    search_query = f"{query} {category_filter}".strip()
    
    # Create a Client instance and use it for search
    client = arxiv.Client()
    search = arxiv.Search(
        query=search_query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    return list(client.results(search))


def check_html_available(paper_id: str) -> bool:
    """Check if HTML version is available for a paper.
    
    Tests if the paper has an HTML version on arXiv by
    sending a HEAD request to the HTML URL.
    
    Args:
        paper_id: The arXiv paper ID
        
    Returns:
        True if HTML version is available, False otherwise
    """
    html_url = f"https://arxiv.org/html/{paper_id}"
    response = requests.head(html_url)
    return response.status_code == 200


def get_html_content(paper_id: str) -> Optional[str]:
    """Get HTML content of a paper if available.
    
    Fetches and parses the HTML version of a paper, removing
    irrelevant elements and extracting the main content.
    
    Args:
        paper_id: The arXiv paper ID
        
    Returns:
        Extracted text content from the HTML version or None if unavailable
    """
    html_url = f"https://arxiv.org/html/{paper_id}"
    response = requests.get(html_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove scripts, styles, and navigation elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        # Get main content
        main_content = soup.find('main') or soup.find('body')
        if main_content:
            return main_content.get_text(separator='\n', strip=True)
    return None


# ============= Vector Database Functions =============

def add_paper_to_vector_db(paper_info: Dict[str, Any]) -> None:
    """Add a paper to the vector database.
    
    Stores a paper's metadata and content in the vector database
    for semantic search.
    
    Args:
        paper_info: Dictionary containing paper metadata and content
    """
    # Skip if collection is not available
    if collection is None:
        if DEBUG_MODE:
            print(f"Skipping vector DB storage for: {paper_info['title']} (collection not available)")
        return
        
    # Use a combination of title and content for embedding
    text_to_embed = f"{paper_info['title']} {paper_info['content'][:5000]}"
    
    # Add to the collection
    try:
        collection.add(
            ids=[paper_info['paper_id']],
            documents=[text_to_embed],
            metadatas=[{
                'title': paper_info['title'],
                'authors': paper_info['authors'],
                'published': paper_info['published'],
                'url': paper_info['url'],
                'abstract': paper_info['abstract']
            }]
        )
        
        if DEBUG_MODE:
            print(f"Added paper to vector DB: {paper_info['title']}")
    except Exception as e:
        print(f"Error adding paper to vector DB: {e}")


# Alternative embedding function that doesn't use ChromaDB's interface
def get_text_embedding(text: str) -> List[float]:
    """Get embedding for a single text using Google's API.
    
    This function can be used outside of ChromaDB for testing.
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector or None if failed
    """
    try:
        if len(text) > 8000:
            text = text[:8000]
            
        result = genai.embed_content(
            model="embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        
        print(f"Direct embedding result: {result}")
        print(f"Result type: {type(result)}")
        print(f"Result has embedding attr: {hasattr(result, 'embedding')}")
        
        if hasattr(result, 'embedding'):
            return result.embedding
        else:
            # Try dict-style access
            try:
                if isinstance(result, dict) and "embedding" in result:
                    return result["embedding"]
                else:
                    return None
            except:
                return None
    except Exception as e:
        print(f"Direct embedding error: {type(e).__name__}: {e}")
        return None


def query_vector_db(query: str, n_results: int = 5) -> List[Dict[str, Any]]:
    """Query the vector database for papers semantically similar to the query.
    
    Performs a semantic search in the vector database to find
    papers most relevant to the query.
    
    Args:
        query: The search query
        n_results: Maximum number of results to return
        
    Returns:
        List of paper dictionaries containing metadata
    """
    # Skip if collection is not available
    if collection is None:
        if DEBUG_MODE:
            print("Vector DB search not available, using keyword search only")
        return []
        
    try:
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        papers = []
        
        if results and 'ids' in results and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                paper_id = results['ids'][0][i]
                
                # Skip if no metadata (shouldn't happen, but just in case)
                if 'metadatas' not in results or i >= len(results['metadatas'][0]):
                    continue
                    
                metadata = results['metadatas'][0][i]
                
                # Get the full paper info from our database
                if paper_id in papers_db:
                    papers.append(papers_db[paper_id])
                else:
                    # Reconstruct from metadata if not in cache
                    papers.append({
                        'paper_id': paper_id,
                        'title': metadata.get('title', 'Unknown Title'),
                        'authors': metadata.get('authors', 'Unknown Authors'),
                        'published': metadata.get('published', 'Unknown Date'),
                        'url': metadata.get('url', ''),
                        'abstract': metadata.get('abstract', ''),
                        'content': metadata.get('abstract', '')  # Fall back to abstract
                    })
        
        return papers
    except Exception as e:
        print(f"Error querying vector DB: {e}")
        return []


# ============= LangGraph Workflow Nodes =============

def query_expansion_node(state: RAGState) -> RAGState:
    """LangGraph node that expands the original query.
    
    Takes the original query and generates an expanded version
    with additional concepts and search terms.
    
    Args:
        state: The current LangGraph state
        
    Returns:
        Updated state with expanded query
    """
    query = state["query"]
    
    # Use a very explicit prompt to avoid the model repeating our instructions
    prompt = f"""
    Please expand the following research query:
    
    Query: "{query}"
    
    Provide a detailed expansion that identifies key concepts, 
    terminology, and relevant subtopics. Do not include phrases like
    "Query:" or "Expanded:" in your response. Just provide the expanded content.
    """
    
    response = model.generate_content(prompt)
    expanded_query = response.text.strip()
    
    if DEBUG_MODE:
        print("EXPANSION NODE - Output expanded query:", expanded_query[:100] + "...")
    
    return {"expanded_query": expanded_query}


def search_papers_node(state: RAGState) -> RAGState:
    """LangGraph node that searches for papers based on the query.
    
    Uses the original and expanded query to search arXiv for relevant
    papers, processes them, and stores the results in both the regular
    database and the vector database.
    
    Args:
        state: The current LangGraph state
        
    Returns:
        Updated state with search results
    """
    if DEBUG_MODE:
        print("SEARCH NODE - Input state keys:", list(state.keys()))
    
    query = state["query"]
    expanded_query = state["expanded_query"]
    
    # Extract actual search terms from expanded query
    # First, check if expanded query starts with "Query:" (which would indicate our formatting issue)
    if "Query:" in expanded_query and "Expanded:" in expanded_query:
        # Extract just the expansion part
        expanded_query = expanded_query.split("Expanded:")[1].strip()
    
    # Extract key terms based on domain
    domain_specific_terms = []
    
    # Domain detection logic
    if "vision transformer" in query.lower() or "vit" in query.lower():
        domain_specific_terms = ["Vision Transformer", "ViT", "image patches", 
                          "self-attention", "transformer encoder", 
                          "multi-head attention", "computer vision"]
    elif "graph" in query.lower() and "neural" in query.lower():
        domain_specific_terms = ["Graph Neural Network", "GNN", "node embedding",
                          "message passing", "graph attention", "GraphSAGE"]
    # Add more domain detection as needed
    elif "reinforcement learning" in query.lower() or " rl " in f" {query.lower()} ":
        domain_specific_terms = ["Reinforcement Learning", "RL", "policy gradient", 
                          "Q-learning", "reward function", "MDP", "Markov Decision Process", 
                          "DDPG", "PPO", "TD learning", "actor-critic"]
    elif "large language model" in query.lower() or "llm" in query.lower():
        domain_specific_terms = ["Large Language Model", "LLM", "transformer", 
                          "attention mechanism", "GPT", "BERT", "prompt engineering", 
                          "fine-tuning", "few-shot learning", "instruction tuning"]
    elif "diffusion" in query.lower() and ("model" in query.lower() or "image" in query.lower()):
        domain_specific_terms = ["Diffusion Model", "DDPM", "latent diffusion", 
                          "score-based generative model", "noise prediction", 
                          "reverse diffusion", "U-Net", "text-to-image"]
    elif "robotics" in query.lower() or "robot" in query.lower():
        domain_specific_terms = ["Robotics", "robot learning", "manipulation", 
                          "grasping", "trajectory optimization", "inverse kinematics", 
                          "motion planning", "control policy", "sim2real"]
    elif "recommendation" in query.lower() or "recommender" in query.lower():
        domain_specific_terms = ["Recommender System", "collaborative filtering", 
                          "content-based filtering", "matrix factorization", 
                          "user embedding", "item embedding", "CTR prediction"]
    elif "computer vision" in query.lower() or "image" in query.lower():
        domain_specific_terms = ["Computer Vision", "CNN", "object detection", 
                          "segmentation", "image recognition", "feature extraction", 
                          "SIFT", "ResNet", "Faster R-CNN", "YOLO"]
    elif ("natural language" in query.lower() or "nlp" in query.lower()) and "transformer" not in query.lower():
        domain_specific_terms = ["Natural Language Processing", "NLP", "named entity recognition", 
                          "sentiment analysis", "text classification", "word embedding", 
                          "language model", "sequence-to-sequence", "LSTM", "RNN"]
    elif "generative" in query.lower() or "gan" in query.lower():
        domain_specific_terms = ["Generative Adversarial Network", "GAN", "StyleGAN", 
                          "generator", "discriminator", "adversarial training", 
                          "latent space", "mode collapse", "image synthesis"]
    elif "attention" in query.lower() or "transformer" in query.lower():
        domain_specific_terms = ["Transformer", "attention mechanism", "self-attention", 
                          "multi-head attention", "encoder-decoder", "positional encoding", 
                          "cross-attention", "attention weights"]
    elif "quantum" in query.lower() and ("computing" in query.lower() or "machine learning" in query.lower()):
        domain_specific_terms = ["Quantum Computing", "quantum machine learning", 
                          "quantum circuit", "qubit", "quantum gate", "variational quantum circuit", 
                          "QAOA", "quantum advantage", "quantum supremacy"]
    
    # Generic ML terms for any ML-related query
    if any(term in query.lower() for term in ["machine learning", "neural network", "deep learning", "ai"]):
        generic_ml_terms = ["neural network", "deep learning", "backpropagation", 
                     "gradient descent", "loss function", "activation function", 
                     "hyperparameter tuning", "regularization", "overfitting"]
        domain_specific_terms.extend(generic_ml_terms)
    
    # If no specific domain is detected, extract key terms from the expanded query
    if not domain_specific_terms and expanded_query:
        # Extract potential terms from expanded query
        expanded_lines = expanded_query.split('. ')
        for line in expanded_lines:
            # Find capitalized terms or terms in quotes that might be important concepts
            import re
            potential_terms = re.findall(r'([A-Z][a-zA-Z0-9]+([ \-][A-Z][a-zA-Z0-9]+)*)', line)
            quoted_terms = re.findall(r'"([^"]+)"', line)
            
            # Add these as domain terms
            for term in potential_terms:
                if isinstance(term, tuple):
                    term = term[0]  # Extract the actual term from regex match tuple
                if len(term) > 3 and term not in domain_specific_terms:  # Only terms longer than 3 chars
                    domain_specific_terms.append(term)
                    
            domain_specific_terms.extend(quoted_terms)
    
    # Log the detected terms
    if domain_specific_terms:
        if DEBUG_MODE or len(domain_specific_terms) > 0:
            print(f"Detected domain terms: {domain_specific_terms}")
    else:
        if DEBUG_MODE:
            print("No specific domain terms detected, using original query only")
    
    # Create a clean search query by combining original and expanded
    search_query = query
    if domain_specific_terms:
        expanded_terms = " OR ".join(f'"{term}"' for term in domain_specific_terms)
        search_query = f'"{query}" OR ({expanded_terms})'
    
    if DEBUG_MODE:
        print(f"Clean search query: {search_query}")
    
    # Regular search via arXiv API
    papers = arxiv_api_search(search_query)
    print(f"Found {len(papers)} papers via API search")
    
    # Process and store papers
    results = []
    for paper in papers:
        paper_id = paper.entry_id.split('/')[-1]
        
        # Skip if already in database
        if paper_id in papers_db:
            results.append(papers_db[paper_id])
            continue
            
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
        
        # Add to vector database
        add_paper_to_vector_db(paper_info)
        
        results.append(paper_info)
    
    if DEBUG_MODE:
        print(f"Processed {len(results)} papers for state")
    
    # Now perform semantic search
    semantic_results = query_vector_db(query)
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
    
    # Return updated state
    updated_state = {
        "search_results": combined_results[:10],  # Limit to top 10 papers
        "embedding_results": semantic_results
    }
    if DEBUG_MODE:
        print("SEARCH NODE - Output state keys:", list(updated_state.keys()))
    return updated_state


def analyze_papers_node(state: RAGState) -> RAGState:
    """LangGraph node that analyzes papers.
    
    Takes the search results and generates an analysis
    of the papers in relation to the original query.
    
    Args:
        state: The current LangGraph state
        
    Returns:
        Updated state with paper analysis and context
    """
    query = state["query"]
    search_results = state["search_results"]
    
    # Get the vector search results if available
    embedding_results = state.get("embedding_results", [])
    
    # Prioritize embedding results if available
    papers_to_analyze = embedding_results if embedding_results else search_results
    
    analysis = analyze_papers(query, papers_to_analyze)
    
    # Extract relevant content for context
    context = []
    for paper in papers_to_analyze[:5]:
        context.append(f"Title: {paper['title']}\nAuthors: {paper['authors']}\nAbstract: {paper['abstract']}")
    
    return {
        "analysis": analysis,
        "context": context
    }


def generate_response_node(state: RAGState) -> RAGState:
    """LangGraph node that generates the final response.
    
    Formats the analysis and adds it to the message history
    in the state.
    
    Args:
        state: The current LangGraph state
        
    Returns:
        Updated state with messages
    """
    query = state["query"]
    context = state["context"]
    analysis = state["analysis"]
    
    # Format the message as an AI response
    message = {
        "role": "assistant",
        "content": analysis
    }
    
    # Add the message to the state
    if "messages" not in state:
        state["messages"] = []
    
    state["messages"].append(message)
    
    return {"messages": state["messages"]}


# ============= LangGraph Workflow =============

def create_research_workflow():
    """Create a LangGraph workflow for research.
    
    Defines the workflow graph with nodes for query expansion,
    paper search, analysis, and response generation.
    
    Returns:
        A compiled LangGraph workflow
    """
    # Initialize the workflow with the RAGState
    workflow = StateGraph(RAGState)
    
    # Add nodes to the graph
    workflow.add_node("query_expansion", query_expansion_node)
    workflow.add_node("search_papers", search_papers_node)
    workflow.add_node("analyze_papers", analyze_papers_node)
    workflow.add_node("generate_response", generate_response_node)
    
    # Add edges to connect the nodes
    workflow.add_edge("query_expansion", "search_papers")
    workflow.add_edge("search_papers", "analyze_papers")
    workflow.add_edge("analyze_papers", "generate_response")
    
    # Set the entry point
    workflow.set_entry_point("query_expansion")
    
    # Compile the workflow
    return workflow.compile()


# ============= Interface Functions =============

def research_arxiv_langgraph(query: str) -> Dict[str, Any]:
    """Research arXiv papers using the LangGraph workflow with embeddings.
    
    Main function that executes the full research pipeline
    on a given query, using both keyword and semantic search.
    
    Args:
        query: The research query
        
    Returns:
        The final state with all results
    """
    # Create the workflow
    workflow = create_research_workflow()
    
    # Initialize the state
    initial_state = {
        "query": query,
        "expanded_query": "",
        "context": [],
        "messages": [],
        "search_results": [],
        "embedding_results": [],
        "analysis": ""
    }
    
    # Execute the workflow
    final_state = workflow.invoke(initial_state)
    
    # Debug print state
    print("Final state keys:", list(final_state.keys()))
    
    # Check if search_results exists and has content
    if "search_results" not in final_state or not final_state["search_results"]:
        print("WARNING: No search results found in final state!")
        # If the search_results got lost, we should check if it's available in our papers_db
        if papers_db:
            print(f"Found {len(papers_db)} papers in papers_db, using those instead")
            final_state["search_results"] = list(papers_db.values())
    
    return final_state


# ============= Display Functions =============

def display_langgraph_results(results: Dict[str, Any]) -> None:
    """Display the research results in a formatted way.
    
    Creates formatted Markdown outputs for the expanded query,
    research analysis, and top papers, with separate sections
    for semantic search results.
    
    Args:
        results: The final state from the research workflow
    """
    from IPython.display import display, Markdown, HTML
    
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


# ============= Main Execution =============

if __name__ == "__main__":
    # Control verbosity - set to True if you want debug output
    set_debug_mode(False)
    
    # Clear intro
    print("=" * 50)
    print("arXiv Research Pipeline with Embedding-Based Search")
    print("=" * 50)
    
    # Ask for research query
    query = input("Enter your research query: ")
    
    # Run the research pipeline
    print("\nSearching arXiv and analyzing papers...")
    results = research_arxiv_langgraph(query)
    
    # Display results
    print("\nDisplaying research results:\n")
    display_langgraph_results(results) 