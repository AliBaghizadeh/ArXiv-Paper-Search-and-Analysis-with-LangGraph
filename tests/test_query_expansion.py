"""
Tests for the query expansion module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arxiv_search.query_expansion import expand_research_query, extract_domain_specific_terms, create_search_query

class TestQueryExpansion(unittest.TestCase):
    """Tests for query expansion functionality."""
    
    def test_create_search_query_with_terms(self):
        """Test creating a search query with domain terms."""
        # Test data
        query = "vision transformers"
        domain_terms = ["Vision Transformer", "ViT", "image patches"]
        
        # Call function
        result = create_search_query(query, domain_terms)
        
        # Expected result
        expected = '"vision transformers" OR ("Vision Transformer" OR "ViT" OR "image patches")'
        
        # Assertions
        self.assertEqual(result, expected)
    
    def test_create_search_query_no_terms(self):
        """Test creating a search query without domain terms."""
        # Test data
        query = "vision transformers"
        domain_terms = []
        
        # Call function
        result = create_search_query(query, domain_terms)
        
        # Assertions
        self.assertEqual(result, query)
    
    def test_extract_domain_specific_terms_vision(self):
        """Test extracting domain terms for vision transformer query."""
        # Test data
        query = "vision transformer applications"
        expanded_query = "The query is about applications of Vision Transformers in various domains."
        
        # Call function
        result = extract_domain_specific_terms(query, expanded_query)
        
        # Assertions
        self.assertIn("Vision Transformer", result)
        self.assertIn("ViT", result)
        self.assertIn("computer vision", result)
    
    def test_extract_domain_specific_terms_nlp(self):
        """Test extracting domain terms for NLP query."""
        # Test data
        query = "natural language processing techniques"
        expanded_query = "The query focuses on techniques used in Natural Language Processing."
        
        # Call function
        result = extract_domain_specific_terms(query, expanded_query)
        
        # Assertions
        self.assertIn("Natural Language Processing", result)
        self.assertIn("NLP", result)
    
    def test_extract_domain_specific_terms_fallback(self):
        """Test extracting domain terms with fallback to expanded query."""
        # Test data
        query = "quantum computing algorithms"
        expanded_query = "This query explores Quantum Computing algorithms such as Shor's Algorithm and Grover's Algorithm."
        
        # Call function
        result = extract_domain_specific_terms(query, expanded_query)
        
        # Assertions
        self.assertIn("Quantum Computing", result)
        self.assertIn("quantum circuit", result)
    
    @patch("arxiv_search.query_expansion.expand_research_query")
    def test_expand_research_query(self, mock_expand):
        """Test the expand_research_query function."""
        # Setup mock
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Expanded query text"
        mock_model.generate_content.return_value = mock_response
        
        # Set up the mock function
        mock_expand.return_value = "Expanded query text"
        
        # Call function
        result = expand_research_query("vision transformers", mock_model)
        
        # Assertions
        self.assertEqual(result, "Expanded query text")

if __name__ == "__main__":
    unittest.main() 