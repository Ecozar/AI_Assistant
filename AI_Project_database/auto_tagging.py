"""
AUTO TAGGING SYSTEM
------------------
Automatic content categorization for the AI Assistant Brain Project.

This module provides semantic-based automatic tagging by:
1. Computing embeddings for content
2. Comparing against predefined tag descriptions
3. Suggesting relevant tags based on similarity
4. Tracking tag suggestions for review

Design Philosophy:
- Tags must be semantically meaningful
- Suggestions must be confidence-scored
- System must be extensible for new tags
- Must support human review workflow

Technical Requirements:
- Must use same embedding model as rest of system
- Must support offline operation
- Must be efficient for bulk processing
- Must maintain suggestion history
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
from sentence_transformers import util

from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.text_utils import text_processor, generate_embedding
from AI_Project_Brain.model_manager import model_manager
from config import APPROVED_TAGS  # Import approved tags from config

# Configure logging
logger = logging.getLogger(__name__)

class AutoTagger:
    """
    Manages automatic content tagging.
    
    Core Responsibilities:
    1. Compute tag embeddings
    2. Generate tag suggestions
    3. Track suggestion confidence
    4. Maintain suggestion history
    
    Design Notes:
    - Uses model_manager for consistent embeddings
    - Caches tag embeddings for efficiency
    - Supports confidence thresholds
    - Enables suggestion review workflow
    """
    
    def __init__(self):
        """
        Initialize the auto-tagger.
        
        Sets up:
        - Access to embedding model
        - Tag embeddings cache
        - Suggestion tracking
        """
        self.model = model_manager.get_model()
        self._tag_embeddings: Dict[str, np.ndarray] = {}
        self._compute_tag_embeddings()

    def _compute_tag_embeddings(self):
        """
        Compute embeddings for all tag descriptions.
        
        Note:
            Called during initialization
            Caches embeddings for efficiency
            Critical for offline operation
        """
        try:
            for tag, description in APPROVED_TAGS.items():
                self._tag_embeddings[tag] = self.model.encode([description])[0]
                
        except Exception as e:
            logger.error(f"Error computing tag embeddings: {str(e)}")
            raise

    def get_suggested_tags(self,
                         text: str,
                         threshold: float = 0.3,
                         max_suggestions: int = 5) -> List[Tuple[str, float]]:
        """
        Generate tag suggestions for text.
        
        Args:
            text: Content to tag
            threshold: Minimum similarity score (0-1)
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of (tag, confidence) tuples
            
        Note:
            Uses cosine similarity for matching
            Returns confidence scores with suggestions
            Filters by threshold before limit
        """
        try:
            # Get text embedding
            text_embedding = self.model.encode([text])[0]
            
            # Calculate similarities with all tags
            similarities = []
            for tag, tag_embedding in self._tag_embeddings.items():
                similarity = float(util.cos_sim(
                    text_embedding.reshape(1, -1),
                    tag_embedding.reshape(1, -1)
                ))
                if similarity >= threshold:
                    similarities.append((tag, similarity))
            
            # Sort by similarity and limit
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error generating tag suggestions: {str(e)}")
            return []

    def bulk_tag_content(self,
                        texts: List[str],
                        threshold: float = 0.3) -> List[List[Tuple[str, float]]]:
        """
        Generate tags for multiple texts efficiently.
        
        Args:
            texts: List of content to tag
            threshold: Minimum similarity score
            
        Returns:
            List of tag suggestion lists
            
        Note:
            More efficient than individual calls
            Uses batch embedding when possible
        """
        try:
            # Compute embeddings in batch
            text_embeddings = self.model.encode(texts)
            
            results = []
            for text_embedding in text_embeddings:
                # Calculate similarities for this text
                similarities = []
                for tag, tag_embedding in self._tag_embeddings.items():
                    similarity = float(util.cos_sim(
                        text_embedding.reshape(1, -1),
                        tag_embedding.reshape(1, -1)
                    ))
                    if similarity >= threshold:
                        similarities.append((tag, similarity))
                
                # Sort and add to results
                similarities.sort(key=lambda x: x[1], reverse=True)
                results.append(similarities)
                
            return results
            
        except Exception as e:
            logger.error(f"Error in bulk tagging: {str(e)}")
            return [[] for _ in texts]

    def log_suggestion(self,
                      content_id: str,
                      suggested_tags: List[Tuple[str, float]],
                      accepted_tags: Optional[List[str]] = None):
        """
        Log tag suggestions for review.
        
        Args:
            content_id: ID of the tagged content
            suggested_tags: List of (tag, confidence) tuples
            accepted_tags: Optional list of accepted tags
            
        Note:
            Tracks suggestions for quality monitoring
            Enables suggestion review workflow
            Helps improve system over time
        """
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO auto_tag_log 
                    (content_id, suggested_tags, accepted_tags, timestamp)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    content_id,
                    json.dumps(suggested_tags),
                    json.dumps(accepted_tags) if accepted_tags else None
                ))
                
        except Exception as e:
            logger.error(f"Error logging tag suggestion: {str(e)}")

# Global instance for use throughout project
auto_tagger = AutoTagger()

def get_suggested_tags(text: str, threshold: float = 0.3) -> List[str]:
    """
    Convenience function for simple tag suggestions.
    """
    suggestions = auto_tagger.get_suggested_tags(text, threshold)
    return [tag for tag, _ in suggestions]

if __name__ == "__main__":
    # Configure logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Test the function with a sample text
    sample_text = "The quantum theory of gravity remains one of physics' greatest mysteries."
    tags = get_suggested_tags(sample_text)
    print(f"\nSuggested tags for sample text: {tags}")
