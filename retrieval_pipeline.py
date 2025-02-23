"""
UNIFIED RETRIEVAL PIPELINE
-------------------------
This module centralizes all retrieval operations (knowledge chunks and conversation history).

SSOT Requirements:
- All thresholds and limits must come from UI settings
- Consistent similarity metrics across all retrievals
- Unified error handling and logging

Key Components:
1. Knowledge Retrieval
2. Conversation History Retrieval
3. Combined Context Assembly
"""

import logging
import sqlite3
from typing import Tuple, List, Dict
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import get_ui_settings
from config import DB_FILE
from AI_Project_Brain.text_utils import generate_embedding
from AI_Project_Brain.db_manager import db_manager

def get_unified_context(
    query: str,
    conversation_id: str = None
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    AI Note: Unified context retrieval requirements:
    - Must retrieve both knowledge and conversation using same settings
    - Must maintain consistent ordering and relevance scoring
    - Must handle missing/invalid data gracefully
    - Must respect all UI configuration settings
    """
    try:
        settings = get_ui_settings()
        
        # Get knowledge chunks
        knowledge_chunks = get_relevant_context(
            query,
            top_n=settings.get("top_n"),
            min_similarity=settings.get("min_similarity")
        )
        
        # Get conversation history if available
        conversation_context = []
        if conversation_id:
            conversation_context = get_conversation_context(
                conversation_id,
                limit=settings.get("conversation_limit")
            )
        
        logging.debug(
            f"Retrieved {len(knowledge_chunks)} knowledge chunks and "
            f"{len(conversation_context)} conversation turns"
        )
        
        return knowledge_chunks, conversation_context
        
    except Exception as e:
        logging.error(f"Error in unified context retrieval: {e}")
        return [], []

def get_relevant_context(
    query: str,
    top_n: int = None,
    min_similarity: float = None
) -> List[str]:
    """
    AI Note: Knowledge retrieval requirements:
    - Must use consistent similarity metrics
    - Must respect UI settings for thresholds
    - Must handle edge cases gracefully
    """
    try:
        if query is None:
            return []
            
        settings = get_ui_settings()
        top_n = top_n or settings.get("top_n")
        min_similarity = min_similarity or settings.get("min_similarity")
        
        query_embedding = generate_embedding(query)
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT content, embedding FROM text_chunks")
            chunks = cursor.fetchall()
            
            if not chunks:
                return []
            
            similarities = get_ranked_similarities(query_embedding, chunks, min_similarity)
            return [chunk for chunk, _ in similarities[:top_n]]
            
    except Exception as e:
        logging.error(f"Error retrieving knowledge context: {e}")
        return []

def get_ranked_similarities(
    query_embedding,
    chunks: List[Tuple[str, str]],
    min_similarity: float = None
) -> List[Tuple[str, float]]:
    """
    AI Note: Similarity ranking requirements:
    - Must use consistent similarity metric (cosine)
    - Must filter by minimum threshold from settings
    - Must sort by relevance (descending)
    - Must not use hardcoded defaults
    """
    similarities = []
    for content, embedding_str in chunks:
        similarity = compute_similarity(query_embedding, embedding_str)
        if min_similarity is None or similarity >= min_similarity:
            similarities.append((content, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities 

def compute_similarity(query_embedding, embedding_str: str) -> float:
    """
    AI Note: Centralized similarity computation
    - Must use consistent numpy array handling
    - Must handle JSON deserialization
    - Must return float between 0 and 1
    """
    embedding = np.array(json.loads(embedding_str))
    return cosine_similarity(
        query_embedding.reshape(1, -1),
        embedding.reshape(1, -1)
    )[0][0] 

def get_conversation_context(conversation_id: str = "default", limit: int = 5):
    """Get recent conversation history"""
    try:
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT user_message, assistant_message, timestamp
                FROM conversation_history
                WHERE conversation_id = ?
                ORDER BY id ASC
                LIMIT ?
            """, (conversation_id, limit))
            return cursor.fetchall()
    except Exception as e:
        logging.error(f"Error retrieving conversation context: {e}")
        return [] 