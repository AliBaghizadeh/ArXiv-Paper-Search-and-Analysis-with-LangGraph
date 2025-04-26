# ArXiv Search and Analysis: Usage Guide

This guide provides detailed instructions for using the ArXiv Search and Analysis package to search, analyze, and extract insights from research papers.

## Installation

### From PyPI (Recommended)

```bash
pip install arxiv-search
```

### From Source

```bash
git clone https://github.com/your-username/arxiv-search-analysis.git
cd arxiv-search-analysis
pip install -e .
```

## Setup

### API Key

The package requires a Google API key with access to the Gemini API. You can get one from the [Google AI Studio](https://ai.google.dev/).

You can set your API key in one of two ways:

1. As an environment variable (recommended):
   ```bash
   # For Linux/macOS
   export GOOGLE_API_KEY="your_api_key_here"
   
   # For Windows PowerShell
   $env:GOOGLE_API_KEY="your_api_key_here"
   
   # For Windows Command Prompt
   set GOOGLE_API_KEY=your_api_key_here
   ```

2. Directly in your code (not recommended for shared environments):
   ```python
   api_key = "your_api_key_here"
   ```

## Command Line Usage

The package includes a command-line interface for running searches directly from the terminal:

```bash
python -m scripts.run_search "vision transformers"
```

### Command Line Options

- `--debug`: Enable debug output
- `--api-key KEY`: Provide API key directly (overrides environment variable)
- `--model MODEL`: Specify the model to use (default: models/gemini-1.5-pro-latest)
- `--temperature TEMP`: Set temperature for generation (default: 1.0)
- `--save FILENAME`: Save results to a JSON file

Example with options:

```bash
python -m scripts.run_search "graph neural networks for drug discovery" --temperature 0.5 --save results.json
```

## Python API Usage

### Basic Example

```python
import os
import google.generativeai as genai
from arxiv_search.langgraph_workflow import research_arxiv_langgraph

# Set up API key
api_key = os.environ.get("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=api_key)

# Initialize model
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# Run search
query = "vision transformers for medical imaging"
results = research_arxiv_langgraph(query, model, api_key)

# Access results
expanded_query = results["expanded_query"]
analysis = results["analysis"]
papers = results["search_results"]

# Print top papers
for i, paper in enumerate(papers[:3]):
    print(f"{i+1}. {paper['title']}")
    print(f"   URL: {paper['url']}")
```

### Displaying Results in a Jupyter Notebook

```python
from arxiv_search.langgraph_workflow import research_arxiv_langgraph, display_langgraph_results

# Run the search
results = research_arxiv_langgraph(query, model, api_key)

# Display formatted results
display_langgraph_results(results)
```

## Customizing the Search

### Searching in Specific Domains

Different domains have specialized terminology. The query expansion module automatically detects when your query relates to specific domains like Vision Transformers, NLP, Reinforcement Learning, etc., and will add domain-specific terms to improve search results.

### Query Expansion

You can directly use the query expansion function:

```python
from arxiv_search.query_expansion import expand_research_query

expanded_query = expand_research_query("vision transformers", model)
print(expanded_query)
```

### Vector Search

The package uses vector embeddings for semantic search. You can directly use the vector database:

```python
from arxiv_search.vector_search import VectorDatabase

vector_db = VectorDatabase(api_key=api_key, model=model)

# Add a paper
vector_db.add_paper({
    "paper_id": "2303.12345",
    "title": "Example Paper Title",
    "authors": "Author One, Author Two",
    "published": "2023-03-15",
    "url": "https://arxiv.org/abs/2303.12345",
    "content": "This is the content of the paper...",
    "has_html": False
})

# Search for papers
results = vector_db.search("vision transformers", top_k=5)
```

## Advanced Usage

### Custom Workflow

You can create your own workflow by modifying the LangGraph nodes:

```python
from langgraph.graph import StateGraph
from arxiv_search.langgraph_workflow import expand_query_node, search_papers_node, analyze_papers_node

# Create custom workflow
workflow = StateGraph(RAGState)

# Define nodes
workflow.add_node("expand_query", lambda state: expand_query_node(state, model))
workflow.add_node("search_papers", lambda state: search_papers_node(state, vector_db))
workflow.add_node("analyze_papers", lambda state: analyze_papers_node(state, model))
workflow.add_node("custom_node", your_custom_function)

# Define edges with custom logic
workflow.add_edge("expand_query", "search_papers")
workflow.add_edge("search_papers", "analyze_papers")
workflow.add_edge("analyze_papers", "custom_node")

# Compile and run
app = workflow.compile()
results = app.invoke(initial_state)
```

## Performance Tips

1. **API Key Rate Limits**: Be aware of rate limits associated with your API key.
2. **Caching**: The package automatically caches paper information to avoid redundant processing.
3. **HTML Content**: For recent papers (since Dec 2023), HTML content is used instead of abstracts for richer content.
4. **Vector Database**: The vector database improves search relevance but requires more computational resources.

## Troubleshooting

- **API Key Issues**: Ensure your API key has access to the Gemini API and embeddings.
- **Model Availability**: Verify the model you're using (e.g., "models/gemini-1.5-pro-latest") is available in your region.
- **Connection Errors**: The package includes retry logic for API calls, but check your internet connection if issues persist.
- **Token Limits**: Very large papers may exceed the token limits for the model. The package automatically truncates content.

## Additional Resources

- [ArXiv API Documentation](https://arxiv.org/help/api)
- [LangGraph Documentation](https://github.com/langchain-ai/langgraph)
- [Google Generative AI Documentation](https://ai.google.dev/docs/gemini_api_overview) 