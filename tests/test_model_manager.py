"""
MODEL MANAGER TESTS
-----------------
Tests for the model manager functionality.
"""

import os
import unittest
import logging
from concurrent import futures
from typing import List

from AI_Project_Brain.model_manager import model_manager, DEFAULT_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logger.info("Starting Model Manager tests")
        
    def setUp(self):
        """Reset before each test"""
        model_manager.clear_model()

    def test_singleton_pattern(self):
        """Test that we always get the same instance"""
        logger.info("Testing singleton pattern")
        instance1 = model_manager
        instance2 = model_manager
        self.assertIs(instance1, instance2)
        logger.debug("Verified singleton instances match")

    def test_model_caching(self):
        """Test that models are properly cached"""
        logger.info("Testing model caching")
        model1 = model_manager.get_model()
        model2 = model_manager.get_model()
        self.assertIs(model1, model2)
        logger.debug(f"Verified cached model IDs match: {id(model1)}")

    def test_concurrent_access(self):
        """Test thread-safe model access"""
        logger.info("Testing concurrent access")
        
        def worker():
            try:
                return model_manager.get_model()
            except Exception as e:
                logger.error(f"Worker thread error: {str(e)}")
                raise

        # Test with multiple threads
        with futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_models = [executor.submit(worker) for _ in range(3)]
            
        # Collect results and check they're the same instance
        results: List = []
        for future in future_models:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Thread execution error: {str(e)}")
                if os.environ.get('HF_HUB_OFFLINE') == '1':
                    logger.warning("Test skipped due to offline mode")
                    return
                raise

        # Verify all threads got the same model instance
        for model in results[1:]:
            self.assertIs(results[0], model)

    def test_different_models(self):
        """Test handling different model types"""
        logger.info("Testing different model types")
        
        # First load default model
        default_model = model_manager.get_model()
        logger.debug(f"Successfully loaded default model: {id(default_model)}")
        
        # Try loading non-existent model (should fall back to default)
        fallback_model = model_manager.get_model("non-existent-model")
        self.assertIs(fallback_model, default_model)
        logger.debug("Verified fallback to default model")

    def test_model_clearing(self):
        """Test model cleanup"""
        logger.info("Testing model clearing")
        
        # Load model and verify it's cached
        model1_id = id(model_manager.get_model())
        
        # Clear and reload
        model_manager.clear_model(DEFAULT_MODEL)
        model2_id = id(model_manager.get_model())
        
        # Verify we got a new instance
        self.assertNotEqual(model1_id, model2_id)
        logger.debug(f"Verified model clearing: {model1_id} != {model2_id}")

    def test_critical_imports(self):
        """Test that critical imports are available"""
        # Test sentence_transformers
        from sentence_transformers import SentenceTransformer
        self.assertTrue(hasattr(SentenceTransformer, 'encode'))
        
        # Test numpy
        import numpy as np
        self.assertTrue(hasattr(np, 'array'))
        
        logger.debug("Critical imports verified")

if __name__ == '__main__':
    unittest.main() 