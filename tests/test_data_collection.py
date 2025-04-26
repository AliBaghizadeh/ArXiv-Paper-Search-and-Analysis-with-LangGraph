"""
Tests for the data collection module.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from arxiv_search.data_collection import check_html_available, get_html_content, arxiv_api_search

class TestDataCollection(unittest.TestCase):
    """Tests for data collection functionality."""
    
    @patch("arxiv_search.data_collection.requests.head")
    def test_check_html_available_success(self, mock_head):
        """Test checking HTML availability when successful."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response
        
        # Call function
        result = check_html_available("2401.12345")
        
        # Assertions
        self.assertTrue(result)
        mock_head.assert_called_once_with("https://arxiv.org/html/2401.12345", timeout=5)
    
    @patch("arxiv_search.data_collection.requests.head")
    def test_check_html_available_failure(self, mock_head):
        """Test checking HTML availability when it fails."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response
        
        # Call function
        result = check_html_available("2001.12345")
        
        # Assertions
        self.assertFalse(result)
        mock_head.assert_called_once_with("https://arxiv.org/html/2001.12345", timeout=5)
    
    @patch("arxiv_search.data_collection.requests.head")
    def test_check_html_available_exception(self, mock_head):
        """Test checking HTML availability when an exception occurs."""
        # Setup mock
        mock_head.side_effect = Exception("Connection error")
        
        # Call function
        result = check_html_available("2401.12345")
        
        # Assertions
        self.assertFalse(result)
        mock_head.assert_called_once_with("https://arxiv.org/html/2401.12345", timeout=5)
    
    @patch("arxiv_search.data_collection.requests.get")
    @patch("arxiv_search.data_collection.BeautifulSoup")
    def test_get_html_content_success(self, mock_bs, mock_get):
        """Test getting HTML content when successful."""
        # Setup mocks
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><main>Test content</main></body></html>"
        mock_get.return_value = mock_response
        
        mock_soup = MagicMock()
        mock_main = MagicMock()
        mock_main.get_text.return_value = "Test content"
        mock_soup.find.return_value = mock_main
        mock_bs.return_value = mock_soup
        
        # Call function
        result = get_html_content("2401.12345")
        
        # Assertions
        self.assertEqual(result, "Test content")
        mock_get.assert_called_once_with("https://arxiv.org/html/2401.12345", timeout=10)
        mock_bs.assert_called_once_with(mock_response.text, "html.parser")
    
    @patch("arxiv_search.data_collection.arxiv.Client")
    @patch("arxiv_search.data_collection.arxiv.Search")
    def test_arxiv_api_search(self, mock_search, mock_client):
        """Test ArXiv API search."""
        # Setup mocks
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance
        
        mock_search_instance = MagicMock()
        mock_search.return_value = mock_search_instance
        
        mock_paper1 = MagicMock()
        mock_paper1.categories = ["cs.CV"]
        mock_paper2 = MagicMock()
        mock_paper2.categories = ["cs.LG"]
        mock_results = [mock_paper1, mock_paper2]
        mock_client_instance.results.return_value = mock_results
        
        # Call function
        result = arxiv_api_search("vision transformers")
        
        # Assertions
        self.assertEqual(result, mock_results)
        mock_client.assert_called_once()
        mock_search.assert_called()
        mock_client_instance.results.assert_called_once_with(mock_search_instance)

if __name__ == "__main__":
    unittest.main() 