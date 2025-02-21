"""
Test script for unified retrieval pipeline
"""
import sys
import os
import json
import logging
import unittest
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now we can import project modules
from config import DB_FILE  # Changed back to root import
from AI_Project_Brain.text_utils import generate_embedding  # Changed back to root import
from AI_Project_Brain.retrieval_pipeline import (
    get_unified_context,
    get_relevant_context,
    get_ranked_similarities,
    compute_similarity
)
from AI_Project_Brain.app import get_ui_settings, init_db

class TestRetrievalPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test database with sample data"""
        logging.basicConfig(level=logging.DEBUG)
        
        # Import necessary modules for setup
        import sqlite3
        
        # Initialize database tables
        init_db()
        
        # Create test data
        test_content = [
            "Artificial intelligence is a broad field of computer science.",
            "Machine learning is a subset of artificial intelligence.",
            "Neural networks are used in deep learning applications."
        ]
        
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                
                # Store test chunks with embeddings
                for idx, content in enumerate(test_content):
                    embedding = generate_embedding(content)
                    cursor.execute(
                        "INSERT INTO text_chunks (file_id, chunk_index, content, embedding) VALUES (?,?,?,?)",
                        (1, idx, content, json.dumps(embedding.tolist()))
                    )
                
                # Add test conversation
                cursor.execute(
                    """
                    INSERT INTO conversation_history 
                    (conversation_id, user_message, assistant_message) 
                    VALUES (?,?,?)
                    """,
                    ("test_conversation", "What is AI?", "AI is artificial intelligence.")
                )
                
                conn.commit()
                
        except Exception as e:
            logging.error(f"Error setting up test data: {e}")
            raise
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test data"""
        import sqlite3
        try:
            with sqlite3.connect(DB_FILE) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM text_chunks WHERE file_id = 1")
                cursor.execute("DELETE FROM conversation_history WHERE conversation_id = 'test_conversation'")
                conn.commit()
        except Exception as e:
            logging.error(f"Error cleaning up test data: {e}")
        
    def test_compute_similarity(self):
        """Test similarity computation"""
        query_embedding = np.array([1.0, 0.0, 0.0]).reshape(1, -1)
        test_embedding = json.dumps([1.0, 0.0, 0.0])
        
        similarity = compute_similarity(query_embedding, test_embedding)
        self.assertEqual(similarity, 1.0)  # Should be identical
        
    def test_get_relevant_context(self):
        """Test context retrieval"""
        query = "test query"
        chunks = get_relevant_context(query)
        
        # Verify we get a list
        self.assertIsInstance(chunks, list)
        
        # Verify settings are respected
        settings = get_ui_settings()
        if chunks:  # If we found any chunks
            self.assertLessEqual(len(chunks), settings.get("top_n", 2))
            
    def test_get_unified_context(self):
        """Test unified context retrieval"""
        query = "test query"
        conversation_id = "test_conversation"
        
        knowledge, conversation = get_unified_context(query, conversation_id)
        
        # Verify types
        self.assertIsInstance(knowledge, list)
        self.assertIsInstance(conversation, list)
        
        # Verify settings are respected
        settings = get_ui_settings()
        if knowledge:
            self.assertLessEqual(len(knowledge), settings.get("top_n", 2))
        if conversation:
            self.assertLessEqual(len(conversation), settings.get("conversation_limit", 5))
            
    def test_error_handling(self):
        """Test error cases"""
        # Test with invalid conversation_id
        knowledge, conversation = get_unified_context("query", "invalid_id")
        self.assertEqual(conversation, [])
        
        # Test with invalid query
        chunks = get_relevant_context(None)
        self.assertEqual(chunks, [])

if __name__ == '__main__':
    unittest.main() 