from typing import List, Tuple
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.text_utils import get_text_similarity  # Use existing text similarity
import logging
import threading
from config import MEMORY_SETTINGS  # Add this import

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Ensure debug logs are enabled

class AutoTagger:
    def __init__(self):
        """Initialize auto-tagger with available tags"""
        self._lock = threading.Lock()  # Initialize lock first
        self.available_tags = {}
        self.refresh_tags()

    def refresh_tags(self):
        """Force refresh of available tags"""
        with self._lock:
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT name, description FROM tags")
                self.available_tags = {row['name']: row['description'] for row in cursor.fetchall()}
                logger.debug(f"Force refreshed tags: {list(self.available_tags.keys())}")

    def get_suggested_tags(self, text: str, threshold: float = None) -> List[Tuple[str, float]]:
        """Single source of truth for tag suggestions"""
        with self._lock:
            if not self.available_tags:
                self.refresh_tags()
            
            # Use config threshold if none provided
            if threshold is None:
                threshold = MEMORY_SETTINGS['auto_tag']['threshold']
            
            suggested = []
            text = text.lower()
            
            logger.debug(f"Getting suggestions for text: '{text[:100]}...' with threshold {threshold}")
            logger.debug(f"Available tags: {list(self.available_tags.keys())}")
            
            for tag, description in self.available_tags.items():
                confidence = self._calculate_confidence(text, tag, description)
                logger.debug(f"Tag '{tag}' confidence: {confidence}")
                if confidence >= threshold:
                    suggested.append((tag, confidence))
                    logger.debug(f"Suggested tag '{tag}' with confidence {confidence}")
            
            # Sort by confidence and return
            suggestions = sorted(suggested, key=lambda x: x[1], reverse=True)
            logger.debug(f"Final suggestions: {suggestions}")
            return suggestions

    def _calculate_confidence(self, text: str, tag: str, description: str) -> float:
        """Calculate confidence using semantic similarity"""
        text = text.lower()
        tag_text = f"{tag.lower()} {description.lower()}"
        
        # Use our existing semantic similarity function
        similarity = get_text_similarity(text, tag_text)
        
        # Weight exact matches more heavily
        exact_match_bonus = 1.0 if tag.lower() in text else 0.0
        
        confidence = (
            similarity * MEMORY_SETTINGS['auto_tag']['tag_weight'] +
            exact_match_bonus * MEMORY_SETTINGS['auto_tag']['description_weight']
        )
        
        return min(confidence, MEMORY_SETTINGS['auto_tag']['max_confidence'])

# Global instance
auto_tagger = AutoTagger() 