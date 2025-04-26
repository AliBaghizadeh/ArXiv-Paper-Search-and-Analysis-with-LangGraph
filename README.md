# ArXiv Paper Search and Analysis with LangGraph

This project provides an advanced research tool that uses LangGraph workflow and Gemini AI to search ArXiv papers, expand queries, and analyze research papers. It was developed as part of the Google 5-Day GenAI Intensive Course.

## Features

### Data Collection Pipeline
- Uses arXiv API to search for papers based on research queries
- Prioritizes HTML versions of papers (available for papers since Dec 2023)
- Falls back to abstracts for older papers without HTML versions
- Caches paper information to avoid redundant processing
- Extracts clean text content using BeautifulSoup for HTML papers

### Query Enhancement
- Employs few-shot prompting with domain-specific examples
- Expands original queries to include related concepts and terminology
- Extracts key terms from expanded queries for more effective searching
- Adapts to specific research domains with customized examples
- Handles both general research and specialized topics (like Vision Transformers)

### LangGraph RAG Workflow
- Implements a structured workflow with defined nodes and transitions
- Maintains comprehensive state throughout the research process
- Four-stage pipeline: query expansion → paper search → analysis → response generation
- Each node updates specific parts of the state without losing information
- Handles the full research journey from question to comprehensive analysis

### Paper Analysis Capabilities
- Generates in-depth analyses of retrieved papers using few-shot learning
- Identifies connections between papers and research question
- Extracts key contributions, methodologies, and technical details
- Provides research context through carefully selected examples
- Synthesizes information across multiple papers for comprehensive understanding

### User Interaction
- Provides formatted Markdown output for easy reading in notebooks
- Displays expanded query to show understanding of research needs
- Presents comprehensive research analysis with insights and connections
- Lists top papers with titles, authors, publication dates, and abstracts
- Includes direct links to original papers on arXiv
- Simple interface for entering research queries and viewing results

## Setup

1. Clone this repository:
```
git clone https://github.com/your-username/arxiv-search-analysis.git
cd arxiv-search-analysis
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Set up your Google API key:
   - Get a Google API key for Gemini from [Google AI Studio](https://ai.google.dev/)
   - Set your API key as an environment variable:
   ```
   export GOOGLE_API_KEY="your_api_key_here"
   ```
   - Or add it directly in the notebook (not recommended for shared environments)

4. Launch Jupyter notebook:
```
jupyter notebook
```

5. Open `web_search_final.ipynb` and run the cells

## Usage

1. Enter your research query when prompted
2. The system will:
   - Expand your query to include related concepts
   - Search arXiv for relevant papers
   - Generate an in-depth analysis of the top papers
   - Present a formatted output with links to the original papers

## Customizing the Pipeline

The notebook includes several customizable parameters:

### Model Parameters
- Change model version: `model = genai.GenerativeModel('models/gemini-1.5-pro-latest')`
- Adjust temperature: `model = genai.GenerativeModel('models/gemini-1.5-pro-latest', temperature=0.2)`

### Search Parameters
- Change max results: Modify `max_results=20` in `arxiv_api_search()`
- Change sort criteria: Modify `sort_by=arxiv.SortCriterion.Relevance`
- Customize category filters in `arxiv_api_search()`

### Content Processing
- Adjust number of papers analyzed: Change top paper count in `analyze_papers_node()`
- Customize examples: Modify prompts in `expand_research_query()` and `analyze_papers()`

### Display Settings
- Show more papers: Change the slice in `results["search_results"][:5]`
- Adjust abstract length: Modify `paper['abstract'][:300]`

## Requirements

- Python 3.8+
- Google Generative AI API access
- Dependencies listed in requirements.txt:
  - langgraph==0.3.34
  - google-generativeai==0.8.5
  - arxiv
  - requests
  - numpy
  - beautifulsoup4
  - chromadb
  - ipython

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Google 5-Day GenAI Intensive Course
- ArXiv API for providing access to research papers
- LangGraph for workflow implementation
- Google Generative AI for model access 