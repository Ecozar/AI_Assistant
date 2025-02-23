"""
MODEL MANAGER
------------
Core component of the AI Assistant Brain Project's memory system.

This module manages the loading, caching, and access to embedding models 
(currently using sentence-transformers) that are crucial for the project's 
semantic search and auto-tagging capabilities.

Key Design Principles:
1. Singleton Pattern: Ensures only one model instance exists to manage memory
2. Thread Safety: Supports concurrent access in the desktop UI environment
3. Offline Support: Can operate without internet once models are cached
4. Graceful Fallback: Falls back to default model if specific models fail
5. Resource Efficiency: Caches models to avoid reloading

Project Context:
- Part of a larger AI Assistant system designed to run locally on desktop
- Will eventually integrate with local LLMs (e.g., deepseek-r1:8B via ollama)
- Currently focuses on robust memory/embedding foundation
- Must support both conversation history and document chunk embeddings

Technical Constraints:
- Must work offline after initial model download
- Must handle concurrent access from UI and background tasks
- Must manage memory efficiently (models can be large)
- Must be extensible for future LLM integration
"""

import os
import logging
import threading
from typing import Dict, Optional
from sentence_transformers import SentenceTransformer
from config import MODEL_CACHE_DIR, DEFAULT_MODEL

# Configure logging for both development debugging and production monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Project-wide constants
MAX_RETRIES = 3  # Number of attempts for model loading
RETRY_DELAY = 1  # Seconds between retry attempts

class ModelManager:
    """
    Singleton class for managing embedding models.
    
    This class is a critical component that:
    1. Provides thread-safe access to embedding models
    2. Handles model caching and memory management
    3. Supports offline operation
    4. Provides fallback mechanisms for robustness
    
    The singleton pattern ensures consistent model state across the application,
    which is crucial for memory efficiency and consistent embeddings.
    
    Threading Considerations:
    - Uses two levels of locks:
        1. Class-level lock for singleton pattern
        2. Instance-level lock for model loading
    - Designed to handle concurrent requests from UI and background tasks
    
    Memory Management:
    - Models are cached in self._models dict
    - Clear unused models via clear_model method
    - Models persist until explicitly cleared
    
    Future Extensibility:
    - Designed to support additional model types
    - Can be extended for LLM integration
    - Maintains separation of concerns for testing
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """
        Thread-safe singleton implementation.
        
        This ensures only one ModelManager instance exists, which is crucial for:
        1. Memory efficiency (avoiding duplicate models)
        2. Consistent embeddings across the application
        3. Proper resource management
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._models = {}  # Model cache
                cls._instance._init_lock = threading.Lock()  # For thread-safe initialization
            return cls._instance

    def __init__(self):
        """
        Initialize the model manager with type hints for IDE support.
        
        Uses getattr to handle singleton pattern correctly:
        - Avoids re-initialization of existing instance
        - Maintains type hints for IDE support
        - Preserves thread safety
        """
        self._models: Dict[str, SentenceTransformer] = getattr(self, '_models', {})
        self._init_lock: threading.Lock = getattr(self, '_init_lock', threading.Lock())

    def get_model(self, model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
        """
        Get a model instance, loading it if necessary.
        
        This is the primary interface for obtaining embedding models. It:
        1. Checks cache first (fast path)
        2. Loads model if needed (slow path, thread-safe)
        3. Handles offline mode
        4. Provides fallback to default model
        5. Retries failed loads
        
        Args:
            model_name: Name of the model to load (defaults to project standard)
            
        Returns:
            SentenceTransformer: The requested model instance
            
        Raises:
            RuntimeError: If model cannot be loaded and no fallback is available
            
        Threading:
            Thread-safe through _init_lock
            Handles concurrent requests efficiently
        """
        try:
            # Fast path: return cached model if available
            if model_name in self._models:
                return self._models[model_name]
            
            # Slow path: load model with thread safety
            with self._init_lock:
                # Recheck cache in case another thread loaded while waiting
                if model_name in self._models:
                    return self._models[model_name]
                
                # Check offline mode
                offline_mode = os.environ.get('HF_HUB_OFFLINE') == '1' or \
                             os.environ.get('TRANSFORMERS_OFFLINE') == '1'
                
                if offline_mode and not self._is_model_cached(model_name):
                    raise RuntimeError(
                        f"Model {model_name} not found in cache and offline mode is enabled. "
                        "Please run download_models.py first."
                    )

                # Try loading with retries
                for attempt in range(MAX_RETRIES):
                    try:
                        logger.info(f"Loading model: {model_name}")
                        self._models[model_name] = SentenceTransformer(
                            model_name,
                            cache_folder=os.path.join("AI_Project_Brain", "models")
                        )
                        return self._models[model_name]
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:  # Last attempt
                            if model_name != DEFAULT_MODEL:
                                logger.warning(f"Falling back to default model: {DEFAULT_MODEL}")
                                return self.get_model(DEFAULT_MODEL)
                            raise RuntimeError(f"Failed to load model after {MAX_RETRIES} attempts: {str(e)}")
                        logger.error(f"Error loading model {model_name}: {str(e)}")
                        import time
                        time.sleep(RETRY_DELAY)

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            if model_name != DEFAULT_MODEL:
                logger.warning(f"Falling back to default model: {DEFAULT_MODEL}")
                return self.get_model(DEFAULT_MODEL)
            raise RuntimeError("Default model not found in cache. Please run download_models.py first.")

    def _is_model_cached(self, model_name: str) -> bool:
        """
        Check if model files exist in local cache.
        
        This is crucial for offline operation and performance optimization.
        
        Args:
            model_name: Name of model to check
            
        Returns:
            bool: True if model is cached, False otherwise
            
        Note:
            Cache structure follows huggingface_hub conventions
            Checks for directory existence and non-emptiness
        """
        cache_dir = os.path.join("AI_Project_Brain", "models")
        model_dir = os.path.join(cache_dir, f"models--sentence-transformers--{model_name}")
        return os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0

    def clear_model(self, model_name: Optional[str] = None) -> None:
        """
        Clear specific or all models from memory.
        
        This is important for:
        1. Memory management in long-running applications
        2. Testing and development scenarios
        3. Forcing model reloads when needed
        
        Args:
            model_name: Specific model to clear, or None for all models
            
        Threading:
            Thread-safe through _init_lock
            Prevents race conditions with model loading
        """
        with self._init_lock:
            if model_name is None:
                self._models.clear()
                logger.info("All models cleared")
            elif model_name in self._models:
                del self._models[model_name]
                logger.info(f"Model cleared: {model_name}")

# Global instance - used throughout the project
model_manager = ModelManager() 