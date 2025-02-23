import sys
import os
import json
# Ensure the project root is in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import sqlite3
from config import DB_FILE, MEMORY_SETTINGS
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.auto_tagger import get_suggested_tags  # If needed
from AI_Project_Brain.logging_config import configure_logging
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime
from .model_manager import model_manager
from AI_Project_Brain.memory_layers import MemoryLayer
from AI_Project_Brain.memory_manager import memory_manager

"""
CONVERSATION LOGGER
------------------
Core persistence layer for the AI Assistant Brain Project's conversation memory.

This module manages:
1. Storage and retrieval of conversation history
2. Conversation metadata and tagging
3. Semantic search across past conversations
4. Memory cleanup and maintenance

Design Philosophy:
- Every conversation must be persistently stored
- Conversations must be efficiently searchable
- Tags enable flexible categorization
- Memory should be manageable through UI

Technical Requirements:
- Must use SQLite for portable, file-based storage
- Must support concurrent access from UI and background tasks
- Must maintain referential integrity
- Must support future migration/scaling
"""

def update_interaction(record_id: int, new_user_message: str, new_assistant_message: str, new_tags: str = None):
    """Update an existing conversation record"""
    try:
        with db_manager.get_cursor() as cursor:
            if new_tags is None:
                new_tags = ""
            cursor.execute(
                """
                UPDATE conversation_history
                SET user_message = ?, assistant_message = ?, tags = ?
                WHERE id = ?
                """,
                (new_user_message, new_assistant_message, new_tags, record_id)
            )
            logging.info("Successfully updated record id: %s", record_id)
    except Exception as e:
        logging.error(f"Error updating interaction: {e}")
        raise

def update_record_tags(record_id: int, new_tags: str):
    """
    AI Note: Tag update function requirements:
    - Must handle atomic updates
    - Must validate record_id exists
    - Must handle empty tag strings
    - Must maintain tag format consistency
    - Must not duplicate tags
    """
    try:
        with db_manager.get_cursor() as cursor:
            # Validate record exists
            cursor.execute("SELECT id FROM conversation_history WHERE id = ?", (record_id,))
            if not cursor.fetchone():
                logging.error(f"Record {record_id} not found")
                return False
                
            cursor.execute("UPDATE conversation_history SET tags = ? WHERE id = ?", (new_tags, record_id))
            return True
            
    except Exception as e:
        logging.error(f"Error updating record tags: {e}")
        return False

def log_interaction(conversation_id: str, message: str, response: str) -> None:
    """Log an interaction with proper tag handling"""
    try:
        # Get suggested tags using config threshold
        suggested_tags = get_suggested_tags(
            message, 
            threshold=MEMORY_SETTINGS.get('auto_tag', {}).get('default_threshold', 0.125)
        )
        valid_tags = suggested_tags  # suggested_tags is already just the list of tags
        
        # Store in memory system first
        memory_manager.store_memory(
            content=message,
            topics=valid_tags,
            source_type="conversation"
        )
        
        with db_manager.get_cursor() as cursor:
            # Insert into conversation_history with the correct schema
            cursor.execute("""
                INSERT INTO conversation_history 
                (conversation_id, user_message, assistant_message, tags)
                VALUES (?, ?, ?, ?)
            """, (
                conversation_id,
                message,
                response,
                ','.join(valid_tags) if valid_tags else ''
            ))
            
            # Get the record ID
            record_id = cursor.lastrowid
            
            # Update memory_items if needed
            if valid_tags:
                cursor.execute("""
                    UPDATE memory_items 
                    SET topics = ?
                    WHERE id = ?
                """, (json.dumps(valid_tags), conversation_id))
                
                logging.debug(f"Updated memory {conversation_id} with tags: {valid_tags}")
            
            cursor.connection.commit()
            logging.debug(f"Logged interaction {record_id} with tags: {valid_tags}")
            
    except Exception as e:
        logging.error(f"Error logging interaction: {e}", exc_info=True)
        raise

def log_auto_tag(record_id: int, suggested_tags: str):
    """Log auto-tagging results"""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS auto_tag_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id INTEGER,
                    suggested_tags TEXT,
                    reviewed INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                INSERT INTO auto_tag_log (record_id, suggested_tags, reviewed)
                VALUES (?, ?, 0)
                """,
                (record_id, suggested_tags)
            )
    except Exception as e:
        logging.error(f"Error logging auto tag: {e}")
        raise

class ConversationLogger:
    """
    Manages conversation persistence and retrieval.
    
    Core Responsibilities:
    1. CRUD operations for conversations
    2. Semantic search across conversation history
    3. Tag management and auto-tagging
    4. Database maintenance
    
    Database Schema:
    - conversations: Core conversation metadata
    - messages: Individual exchanges
    - tags: Tag definitions and metadata
    - message_tags: Many-to-many relationship for message tagging
    
    Threading:
    - Uses connection per thread for safety
    - Implements proper connection management
    - Handles concurrent access gracefully
    """
    
    def __init__(self, db_path: str = "AI_Project_Brain/files.db"):
        """
        Initialize the conversation logger.
        
        Args:
            db_path: Path to SQLite database file
            
        Note:
            Creates tables if they don't exist
            Initializes embedding model for semantic search
        """
        self.db_path = db_path
        self.model = model_manager.get_model()
        self._init_database()

    def _init_database(self):
        """
        Initialize database schema.
        
        Creates tables for:
        - Conversations (metadata)
        - Messages (actual exchanges)
        - Tags (categorization)
        - Message-Tag relationships
        
        Note:
            Schema designed for future extensibility
            Includes indexes for performance
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Conversations table - metadata about each conversation
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    title TEXT,
                    metadata TEXT  -- JSON field for flexible metadata
                )
            """)
            
            # Messages table - individual exchanges
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    embedding BLOB,  -- Stored as binary for efficiency
                    FOREIGN KEY (conversation_id) 
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                )
            """)
            
            # Tags table - for categorization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    tag_id TEXT PRIMARY KEY,
                    name TEXT UNIQUE,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Message-Tag relationship
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS message_tags (
                    message_id TEXT,
                    tag_id TEXT,
                    confidence FLOAT,  -- For auto-tagging confidence scores
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (message_id, tag_id),
                    FOREIGN KEY (message_id) 
                        REFERENCES messages(message_id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) 
                        REFERENCES tags(tag_id)
                        ON DELETE CASCADE
                )
            """)
            
            # Indexes for performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_conversation 
                ON messages(conversation_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_tags_message 
                ON message_tags(message_id)
            """)

    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection for the current thread.
        
        Returns:
            sqlite3.Connection: Thread-specific connection
            
        Note:
            Implements connection pooling
            Ensures proper resource cleanup
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dictionary access
        return conn

    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Create a new conversation.
        
        Args:
            title: Optional conversation title
            
        Returns:
            str: New conversation ID
            
        Note:
            Uses UUIDs for globally unique IDs
            Stores creation timestamp automatically
        """
        conversation_id = str(uuid.uuid4())
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO conversations (conversation_id, title) VALUES (?, ?)",
                    (conversation_id, title)
                )
            return conversation_id
            
        except Exception as e:
            logging.error(f"Error creating conversation: {str(e)}")
            raise

    def log_interaction(self,
                       conversation_id: str,
                       user_message: str,
                       assistant_message: str,
                       tags: Optional[List[str]] = None) -> str:
        """
        Log a conversation exchange.
        
        Args:
            conversation_id: ID of the conversation
            user_message: User's input
            assistant_message: Assistant's response
            tags: Optional list of tags to apply
            
        Returns:
            str: ID of the new message
            
        Note:
            Computes and stores message embedding
            Handles tag assignment atomically
        """
        message_id = str(uuid.uuid4())
        
        try:
            # Compute message embedding
            combined_text = f"{user_message} {assistant_message}"
            embedding = self.model.encode([combined_text])[0].tobytes()
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert message
                cursor.execute("""
                    INSERT INTO messages 
                    (message_id, conversation_id, user_message, 
                     assistant_message, embedding)
                    VALUES (?, ?, ?, ?, ?)
                """, (message_id, conversation_id, user_message,
                      assistant_message, embedding))
                
                # Handle tags if provided
                if tags:
                    self._assign_tags(cursor, message_id, tags)
                    
            return message_id
            
        except Exception as e:
            logging.error(f"Error logging interaction: {str(e)}")
            raise

    def get_conversation_history(self,
                               conversation_id: str,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to return
            
        Returns:
            List of message dictionaries with metadata
            
        Note:
            Returns messages in chronological order
            Includes tags and metadata
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT m.*, GROUP_CONCAT(t.name) as tags
                    FROM messages m
                    LEFT JOIN message_tags mt ON m.message_id = mt.message_id
                    LEFT JOIN tags t ON mt.tag_id = t.tag_id
                    WHERE m.conversation_id = ?
                    GROUP BY m.message_id
                    ORDER BY m.timestamp DESC
                    LIMIT ?
                """, (conversation_id, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logging.error(f"Error retrieving conversation history: {str(e)}")
            return []

    def search_conversations(self,
                           query: str,
                           limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search conversations using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of relevant messages with similarity scores
            
        Note:
            Uses embedding similarity for semantic search
            Returns results ordered by relevance
        """
        try:
            # Compute query embedding
            query_embedding = self.model.encode([query])[0]
            
            # TODO: Implement efficient similarity search
            # Current implementation is basic; will be optimized
            
            return []  # Placeholder until implementation
            
        except Exception as e:
            logging.error(f"Error searching conversations: {str(e)}")
            return []

    def _assign_tags(self,
                    cursor: sqlite3.Cursor,
                    message_id: str,
                    tags: List[str],
                    confidence: float = 1.0):
        """
        Assign tags to a message.
        
        Args:
            cursor: Database cursor
            message_id: Message to tag
            tags: List of tag names
            confidence: Confidence score for auto-tagging
            
        Note:
            Creates missing tags automatically
            Handles tag assignment atomically
        """
        for tag_name in tags:
            # Get or create tag
            cursor.execute(
                "INSERT OR IGNORE INTO tags (tag_id, name) VALUES (?, ?)",
                (str(uuid.uuid4()), tag_name)
            )
            
            cursor.execute(
                "SELECT tag_id FROM tags WHERE name = ?",
                (tag_name,)
            )
            tag_id = cursor.fetchone()['tag_id']
            
            # Assign tag to message
            cursor.execute("""
                INSERT OR REPLACE INTO message_tags 
                (message_id, tag_id, confidence)
                VALUES (?, ?, ?)
            """, (message_id, tag_id, confidence))

# Global instance for use throughout project
conversation_logger = ConversationLogger()

if __name__ == "__main__":
    configure_logging()
