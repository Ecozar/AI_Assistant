# File: AI_Project_database/tag_manager.py
import sqlite3
from config import DB_FILE
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict
from AI_Project_Brain.db_manager import db_manager  # Add at top with other imports
from AI_Project_Brain.auto_tagger import auto_tagger  # Add at top with other imports

logger = logging.getLogger(__name__)  # Add after the logging import

def load_approved_tags():
    """Load approved tags from the database and return a dictionary mapping tag names to descriptions."""
    tags = {}
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name, description FROM tags")
        results = cursor.fetchall()
        print(f"DEBUG: Loading tags from database: {results}")  # Debug print
        for name, description in results:
            tags[name] = description if description else ""
    except Exception as e:
        logging.error("Error loading approved tags: %s", e)
    finally:
        conn.close()
    return tags

def add_approved_tag(name: str, description: str):
    """Add a new approved tag to the database."""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("INSERT OR IGNORE INTO tags (name, description) VALUES (?, ?)", 
                         (name, description))
            cursor.connection.commit()
        auto_tagger.refresh_tags()
    except Exception as e:
        logger.error("Error adding approved tag: %s", e)

def remove_approved_tag(name: str):
    """Remove an approved tag from the database."""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM tags WHERE name = ?", (name,))
            cursor.connection.commit()
        auto_tagger.refresh_tags()
    except Exception as e:
        logger.error("Error removing approved tag: %s", e)

class TagManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._tags = {}
        self._last_refresh = None
        self.refresh_interval = 300  # 5 minutes
        
    def get_approved_tags(self) -> Dict[str, str]:
        """Thread-safe tag access with automatic refresh"""
        with self._lock:
            if (not self._last_refresh or 
                datetime.now() - self._last_refresh > timedelta(seconds=self.refresh_interval)):
                self._refresh_tags()
            return self._tags.copy()
    
    def _refresh_tags(self):
        """Protected tag refresh"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT name, description FROM tags")
            self._tags = {row['name']: row['description'] for row in cursor.fetchall()}
            self._last_refresh = datetime.now()
            logger.debug(f"Refreshed {len(self._tags)} tags")

# Global instance
tag_manager = TagManager()
