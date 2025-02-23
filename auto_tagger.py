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
        """Get suggested tags with confidence scores"""
        try:
            threshold = threshold or MEMORY_SETTINGS['auto_tag']['default_threshold']
            suggestions = []
            
            for tag, description in self.available_tags.items():
                confidence = self._calculate_confidence(text, tag, description)
                if confidence >= threshold:
                    suggestions.append((tag, confidence))
            
            return sorted(suggestions, key=lambda x: x[1], reverse=True)
        except Exception as e:
            logger.error(f"Error getting suggested tags: {e}")
            return []

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

def get_suggested_tags(text: str, threshold: float = None) -> List[str]:
    """
    Convenience function for simple tag suggestions.
    Returns just the tag names without confidence scores.
    """
    suggestions = auto_tagger.get_suggested_tags(text, threshold)
    return [tag for tag, _ in suggestions]

def load_approved_tags():
    """Load approved tags from the database"""
    tags = {}
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT name, description FROM tags")
            results = cursor.fetchall()
            for name, description in results:
                tags[name] = description if description else ""
    except Exception as e:
        logging.error("Error loading approved tags: %s", e)
    return tags

def add_approved_tag(name: str, description: str = ""):
    """Add a new approved tag to the database."""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("INSERT OR IGNORE INTO tags (name, description) VALUES (?, ?)", 
                         (name, description))
            cursor.connection.commit()
        auto_tagger.refresh_tags()
    except Exception as e:
        logging.error("Error adding approved tag: %s", e)

def remove_approved_tag(name: str):
    """Remove an approved tag from the database."""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM tags WHERE name = ?", (name,))
            cursor.connection.commit()
        auto_tagger.refresh_tags()
    except Exception as e:
        logging.error("Error removing approved tag: %s", e) 