"""
MEMORY SEARCH
------------
Handles efficient memory retrieval and context assembly.
"""

from typing import List, Tuple
import numpy as np
from .memory_manager import MemoryManager
from .db_manager import db_manager
from text_utils import generate_embedding

class MemorySearch:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
    
    def search_by_similarity(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar memories to query"""
        query_embedding = generate_embedding(query)
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, content, importance 
                FROM memory_items
                WHERE layer != 'working'  -- Skip temporary memories
                ORDER BY last_accessed DESC
                LIMIT 100  -- Pre-filter for performance
            """)
            
            results = []
            for row in cursor.fetchall():
                memory_embedding = generate_embedding(row['content'])
                similarity = np.dot(query_embedding, memory_embedding)
                results.append((row['id'], similarity))
            
            # Return top_k most similar
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k] 