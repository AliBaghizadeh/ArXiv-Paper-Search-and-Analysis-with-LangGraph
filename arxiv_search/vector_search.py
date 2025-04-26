"""
Vector search module for paper retrieval.

This module handles vector embedding of papers and semantic search.
"""

import chromadb
from chromadb.utils import embedding_functions
import numpy as np
from typing import List, Dict, Any, Optional

class GoogleEmbeddingFunction(embedding_functions.EmbeddingFunction):
    """Custom embedding function that uses Google's Generative AI API."""
    
    def __init__(self, api_key: str, model: Any, debug: bool = False):
        """Initialize with Google API key and model.
        
        Args:
            api_key: Google API key
            model: Generative AI model instance
            debug: Whether to print debug information
        """
        self.api_key = api_key
        self.model = model
        self.debug = debug
    
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
                embedding_response = self.model.embed_content(
                    model="embedding-001",
                    content=text,
                    task_type="retrieval_document",
                )
                
                if self.debug:
                    print(f"Embedding response type: {type(embedding_response)}")
                    print(f"Embedding response attributes: {dir(embedding_response) if hasattr(embedding_response, '__dir__') else 'No attributes'}")
                
                # Extract embedding vector
                embedding_vector = embedding_response.embedding
                
                embeddings.append(embedding_vector)
                
            except Exception as e:
                if self.debug:
                    print(f"Error generating embedding (details): {type(e).__name__}: {e}")
                # Fallback: random embeddings
                embeddings.append(np.random.rand(768).tolist())
                print("Warning: Using random fallback embedding due to exception")
                
        return embeddings

class VectorDatabase:
    """Vector database for paper storage and retrieval."""
    
    def __init__(self, api_key: str, model: Any, collection_name: str = "arxiv_papers", debug: bool = False):
        """Initialize the vector database.
        
        Args:
            api_key: Google API key
            model: Generative AI model instance
            collection_name: Name of the collection
            debug: Whether to print debug information
        """
        self.api_key = api_key
        self.model = model
        self.debug = debug
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.initialize_db()
    
    def initialize_db(self) -> None:
        """Initialize the vector database."""
        try:
            self.client = chromadb.Client()
            
            # Set up our custom embedding function
            embedding_function = GoogleEmbeddingFunction(
                api_key=self.api_key, 
                model=self.model,
                debug=self.debug
            )
            
            # Create or get a collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=embedding_function
            )
            
            if self.debug:
                print("Successfully initialized vector database")
        except Exception as e:
            print(f"Error initializing vector database: {e}")
            # Create a fallback in-memory dictionary to store papers
            self.collection = None
            print("Using fallback storage without vector search capabilities")
    
    def add_paper(self, paper: Dict[str, Any]) -> None:
        """Add a paper to the vector database.
        
        Args:
            paper: Paper information dictionary
        """
        if not self.collection:
            return
            
        try:
            # Check if paper already exists
            existing = self.collection.get(
                ids=[paper["paper_id"]],
                include=["metadatas"]
            )
            
            if existing["ids"] and existing["ids"][0] == paper["paper_id"]:
                if self.debug:
                    print(f"Paper {paper['paper_id']} already exists in vector database")
                return
                
            # Add paper to vector database
            self.collection.add(
                ids=[paper["paper_id"]],
                documents=[paper["content"]],
                metadatas=[{
                    "title": paper["title"],
                    "authors": paper["authors"],
                    "published": paper["published"],
                    "url": paper["url"],
                    "has_html": paper["has_html"]
                }]
            )
            
            if self.debug:
                print(f"Added paper {paper['paper_id']} to vector database")
        except Exception as e:
            if self.debug:
                print(f"Error adding paper to vector database: {e}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for papers matching the query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of paper dictionaries
        """
        if not self.collection:
            return []
            
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["metadatas", "documents"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return []
                
            papers = []
            for i, paper_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                
                paper = {
                    "paper_id": paper_id,
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "published": metadata["published"],
                    "url": metadata["url"],
                    "has_html": metadata["has_html"],
                    "content": results["documents"][0][i],
                    "score": results["distances"][0][i] if "distances" in results else None
                }
                
                papers.append(paper)
                
            return papers
        except Exception as e:
            if self.debug:
                print(f"Error searching vector database: {e}")
            return [] 