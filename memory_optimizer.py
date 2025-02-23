"""
MEMORY OPTIMIZER
--------------
Optimizes memory storage based on usage patterns.

Integration Points:
- Uses filtered access patterns from memory_access_tracker
- Stores optimization metadata for future reference
- Coordinates with domain expertise for knowledge preservation
- Prevents optimization based on noise/temporary patterns
"""

import logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Set, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import threading
import time
import json

from .db_manager import db_manager
from .memory_layers import MemoryLayer
from AI_Project_Brain.text_utils import generate_embedding
from config import MEMORY_SETTINGS
from AI_Project_Brain.memory_access_tracker import memory_access_tracker

@dataclass
class OptimizationMetrics:
    """Tracks optimization results"""
    duplicates_merged: int = 0
    memories_consolidated: int = 0
    bytes_compressed: int = 0
    time_taken: float = 0.0

class MemoryOptimizer:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._optimizer_thread = None
        self.last_run = None
        self.metrics = OptimizationMetrics()
        self._init_database()
        logger.info("Memory optimizer initialized")

    def start(self):
        """Start background optimization"""
        self._optimizer_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self._optimizer_thread.start()
        logging.info("Memory optimizer started")

    def stop(self):
        """Stop background optimization"""
        self._stop_event.set()
        if self._optimizer_thread:
            self._optimizer_thread.join()
        logging.info("Memory optimizer stopped")

    def _optimization_loop(self):
        """Main optimization loop"""
        while not self._stop_event.is_set():
            try:
                # Only run during idle times
                if self._should_optimize():
                    self.optimize_memories()
                time.sleep(3600)  # Check every hour
            except Exception as e:
                logging.error(f"Error in optimization loop: {e}")

    def _should_optimize(self) -> bool:
        """Determine if optimization should run"""
        if not self.last_run:
            return True
            
        # Check if enough time has passed
        time_since_last = datetime.now() - self.last_run
        if time_since_last < timedelta(hours=24):
            return False
            
        # Check memory pressure
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM memory_items")
            memory_count = cursor.fetchone()[0]
            
        return memory_count > MEMORY_SETTINGS.get('optimization_threshold', 1000)

    def optimize_memories(self) -> OptimizationMetrics:
        """Run full memory optimization"""
        start_time = time.time()
        
        try:
            with self._lock:
                self.metrics = OptimizationMetrics()
                
                # Run optimization steps
                self._deduplicate_memories()
                self._consolidate_memories()
                self._compress_long_term()
                
                self.metrics.time_taken = time.time() - start_time
                self.last_run = datetime.now()
                
                logging.info(f"Memory optimization completed: {self.metrics}")
                return self.metrics
                
        except Exception as e:
            logging.error(f"Error during memory optimization: {e}")
            raise

    def _deduplicate_memories(self):
        """Merge highly similar memories"""
        with db_manager.get_cursor() as cursor:
            # Get memories to compare
            cursor.execute("""
                SELECT id, content, importance, embedding
                FROM memory_items 
                WHERE layer != ?
                ORDER BY importance DESC
            """, (MemoryLayer.WORKING.value,))
            
            memories = cursor.fetchall()
            merged = set()
            
            for i, mem1 in enumerate(memories):
                if mem1['id'] in merged:
                    continue
                    
                for mem2 in memories[i+1:]:
                    if mem2['id'] in merged:
                        continue
                        
                    similarity = self._compute_similarity(
                        mem1['embedding'], 
                        mem2['embedding']
                    )
                    
                    if similarity > MEMORY_SETTINGS.get('merge_threshold', 0.95):
                        self._merge_memories(cursor, mem1, mem2)
                        merged.add(mem2['id'])
                        self.metrics.duplicates_merged += 1

    def _consolidate_memories(self):
        """Combine related memories into higher-level summaries"""
        with db_manager.get_cursor() as cursor:
            # Find clusters of related memories
            cursor.execute("""
                SELECT source_id, target_id, strength 
                FROM memory_connections
                WHERE strength > ?
                ORDER BY strength DESC
            """, (MEMORY_SETTINGS.get('consolidation_threshold', 0.8),))
            
            connections = cursor.fetchall()
            clusters = self._find_memory_clusters(connections)
            
            for cluster in clusters:
                if len(cluster) >= 3:  # Only consolidate larger clusters
                    self._create_consolidated_memory(cursor, cluster)
                    self.metrics.memories_consolidated += len(cluster)

    def _compress_long_term(self):
        """Compress long-term memories to save space"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, content 
                FROM memory_items
                WHERE layer = ?
            """, (MemoryLayer.LONG_TERM.value,))
            
            for memory in cursor.fetchall():
                original_size = len(memory['content'].encode())
                compressed = self._compress_content(memory['content'])
                
                if len(compressed.encode()) < original_size:
                    cursor.execute("""
                        UPDATE memory_items
                        SET content = ?,
                            compressed = TRUE
                        WHERE id = ?
                    """, (compressed, memory['id']))
                    
                    self.metrics.bytes_compressed += (
                        original_size - len(compressed.encode())
                    )

    def _compute_similarity(self, emb1: str, emb2: str) -> float:
        """Compute cosine similarity between embeddings"""
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def _merge_memories(self, cursor, mem1: Dict, mem2: Dict):
        """Merge two similar memories"""
        # Keep the more important memory
        keep_id = mem1['id'] if mem1['importance'] > mem2['importance'] else mem2['id']
        remove_id = mem2['id'] if keep_id == mem1['id'] else mem1['id']
        
        # Update connections to point to kept memory
        cursor.execute("""
            UPDATE memory_connections
            SET target_id = ?
            WHERE target_id = ?
        """, (keep_id, remove_id))
        
        # Remove the duplicate
        cursor.execute("DELETE FROM memory_items WHERE id = ?", (remove_id,))

    def _find_memory_clusters(self, connections: List[Dict]) -> List[Set[str]]:
        """Find clusters of strongly connected memories"""
        clusters = []
        processed = set()
        
        for conn in connections:
            if conn['source_id'] in processed or conn['target_id'] in processed:
                continue
                
            cluster = {conn['source_id'], conn['target_id']}
            self._expand_cluster(cluster, connections, processed)
            clusters.append(cluster)
            processed.update(cluster)
            
        return clusters

    def _expand_cluster(self, cluster: Set[str], connections: List[Dict], 
                       processed: Set[str]):
        """Expand a cluster with strongly connected memories"""
        added = True
        while added:
            added = False
            for conn in connections:
                if conn['source_id'] in cluster and conn['target_id'] not in processed:
                    cluster.add(conn['target_id'])
                    processed.add(conn['target_id'])
                    added = True
                elif conn['target_id'] in cluster and conn['source_id'] not in processed:
                    cluster.add(conn['source_id'])
                    processed.add(conn['source_id'])
                    added = True

    def _create_consolidated_memory(self, cursor, memory_ids: Set[str]):
        """Create a consolidated memory from a cluster"""
        # Get all memories in cluster
        placeholders = ','.join('?' * len(memory_ids))
        cursor.execute(f"""
            SELECT content, importance
            FROM memory_items
            WHERE id IN ({placeholders})
        """, tuple(memory_ids))
        
        memories = cursor.fetchall()
        
        # Create consolidated content
        contents = [m['content'] for m in memories]
        importance = max(m['importance'] for m in memories)
        
        consolidated = self._generate_summary(contents)
        
        # Store consolidated memory
        cursor.execute("""
            INSERT INTO memory_items (
                content, layer, importance, 
                created_at, last_accessed,
                consolidated_from
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            consolidated,
            MemoryLayer.LONG_TERM.value,
            importance,
            datetime.now(),
            datetime.now(),
            ','.join(memory_ids)
        ))

    def _compress_content(self, content: str) -> str:
        """Compress memory content while preserving meaning"""
        # For now, just do basic compression
        # TODO: Implement more sophisticated compression
        sentences = content.split('. ')
        if len(sentences) <= 3:
            return content
            
        # Keep first and last sentences, summarize middle
        compressed = f"{sentences[0]}. ... {sentences[-1]}"
        return compressed

    def _generate_summary(self, contents: List[str]) -> str:
        """Generate a summary of multiple memory contents"""
        # For now, just combine with markers
        # TODO: Implement proper summarization
        return " [+] ".join(contents)

    # Consider access patterns when optimizing memory storage
    def optimize_memory_storage(self, memory_id: str):
        """Optimize storage based on significant access patterns
        
        Integration Notes:
        - Only considers frequent access patterns (noise filtered)
        - Stores metadata to track optimization decisions
        - Coordinates with domain expertise confidence levels
        """
        patterns = memory_access_tracker.analyze_access_patterns(memory_id)
        # Use significant patterns to inform storage decisions
        frequently_accessed_hours = [p['hour'] for p in patterns['hourly_patterns']]
        
        # Store optimization metadata
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE memory_items
                SET optimization_metadata = ?
                WHERE id = ?
            """, (
                json.dumps({'frequent_hours': frequently_accessed_hours}),
                memory_id
            ))
            logger.debug(f"Updated optimization metadata for memory {memory_id}")

    def _init_database(self):
        """Initialize database tables and columns"""
        with db_manager.get_cursor() as cursor:
            # Check and add embedding column if needed
            try:
                cursor.execute("SELECT embedding FROM memory_items LIMIT 1")
            except:
                logger.info("Adding embedding column")
                cursor.execute("""
                    ALTER TABLE memory_items 
                    ADD COLUMN embedding BLOB
                """)
                
            # Add optimization_metadata column if it doesn't exist
            try:
                cursor.execute("SELECT optimization_metadata FROM memory_items LIMIT 1")
            except:
                logger.info("Adding optimization_metadata column")
                cursor.execute("""
                    ALTER TABLE memory_items 
                    ADD COLUMN optimization_metadata TEXT
                """)
                
            logger.debug("Memory optimizer database initialization complete")

# Global instance
memory_optimizer = MemoryOptimizer() 