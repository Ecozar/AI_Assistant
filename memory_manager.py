"""
MEMORY MANAGER
-------------
Centralized memory management system that handles:
1. Memory layers and transitions
2. Memory connections and relationships
3. Memory decay and cleanup
4. Domain expertise tracking
5. Core memory operations
"""

import logging
import os
import sys
from datetime import datetime, timedelta, timezone, UTC
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass
import threading
import json
import uuid
import sqlite3
from enum import Enum
import time
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Change relative imports to absolute imports
from AI_Project_Brain.memory_layers import MemoryLayer, MemoryItem
from .db_manager import (
    db_manager,
    MEMORY_ITEMS_SCHEMA,
    MEMORY_ACCESS_PATTERNS_SCHEMA,
    MEMORY_CONNECTIONS_SCHEMA,
    MEMORY_INDICES,
    DOMAIN_EXPERTISE_SCHEMA,
    DOMAIN_EXPERTISE_INDICES
)
from config import (
    MEMORY_SETTINGS, 
    EXPERTISE_SETTINGS  # Instead of DOMAIN_EXPERTISE_SETTINGS
)
from AI_Project_Brain.text_utils import generate_embedding
from .memory_access_tracker import memory_access_tracker
from AI_Project_Brain.auto_tagger import auto_tagger

# From memory_layers.py
class MemoryLayer(Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PERMANENT = "permanent"

@dataclass
class MemoryItem:
    content: str
    layer: MemoryLayer
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance: float
    topics: List[str]
    related_items: List[str]

# From memory_connections.py
@dataclass
class Connection:
    source_id: str
    target_id: str
    connection_type: str
    strength: float
    created_at: datetime
    last_accessed: Optional[datetime] = None

class MemoryManager:
    def __init__(self):
        """Initialize memory manager with decay handling"""
        self._lock = threading.Lock()
        self._stop_decay = False
        self._decay_thread = None
        self.start_decay_service()
        self._init_database()
    
    def _init_database(self):
        """Initialize memory tables"""
        with db_manager.get_cursor() as cursor:
            # Check for required tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row['name'] for row in cursor.fetchall()}
            required_tables = {
                'memory_items',
                'memory_access_log',  # Changed from memory_access_patterns
                'memory_connections'
            }
            
            missing = required_tables - existing_tables
            if missing:
                raise Exception(f"Missing required tables: {missing}")
            
            # Log schema for debugging
            cursor.execute("PRAGMA table_info(memory_items)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            logger.info(f"Memory Manager table schema: {json.dumps(columns, indent=2)}")

    def start_decay_service(self):
        """Start the memory decay service"""
        if self._decay_thread and self._decay_thread.is_alive():
            logger.warning("Decay service already running")
            return
            
        self._stop_decay = False
        self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._decay_thread.start()
        logger.info("Memory decay service started")
    
    def stop_decay_service(self):
        """Stop the memory decay service"""
        logger.info("Stopping decay service...")
        self._stop_decay = True
        if self._decay_thread:
            self._decay_thread.join(timeout=5.0)
            if self._decay_thread.is_alive():
                logger.warning("Decay thread did not stop cleanly")
            else:
                logger.info("Decay service stopped")

    def _decay_loop(self):
        """Main decay service loop"""
        while not self._stop_decay:
            try:
                protected_ids = self._get_protected_memories()
                self._apply_decay(protected_ids)
                time.sleep(MEMORY_SETTINGS['decay']['check_interval'])
            except Exception as e:
                logger.error(f"Error in decay loop: {str(e)}", exc_info=True)
                time.sleep(60)  # Wait before retrying

    def _get_protected_memories(self) -> Set[str]:
        """Get IDs of memories that should not decay"""
        protected = set()
        try:
            with db_manager.get_cursor() as cursor:
                # Get recently accessed memories
                cursor.execute("""
                    SELECT DISTINCT id FROM memory_items
                    WHERE julianday('now') - julianday(last_accessed) < ?
                    OR layer = ?
                """, (
                    MEMORY_SETTINGS['decay']['protected_period'] / 24.0,  # Convert hours to days
                    MemoryLayer.PERMANENT.value
                ))
                protected.update(row['id'] for row in cursor.fetchall())
                
        except Exception as e:
            logger.error(f"Error getting protected memories: {str(e)}")
            
        return protected

    def _apply_decay(self, protected_ids: Set[str]):
        """Apply decay to memories and connections"""
        try:
            with db_manager.get_cursor() as cursor:
                # Apply memory decay by layer
                for layer in MemoryLayer:
                    if layer == MemoryLayer.PERMANENT:
                        continue  # Skip permanent memories
                        
                    decay_rate = MEMORY_SETTINGS['decay']['layers'][layer.value]['rate']
                    min_importance = MEMORY_SETTINGS['decay']['layers'][layer.value]['min_importance']
                    
                    # Apply memory decay with protection
                    if protected_ids:
                        cursor.execute("""
                            UPDATE memory_items
                            SET importance = MAX(?, importance * (1.0 - ?))
                            WHERE layer = ? AND id NOT IN ({})
                        """.format(','.join('?' * len(protected_ids))),
                        [min_importance, decay_rate, layer.value] + list(protected_ids))
                    else:
                        cursor.execute("""
                            UPDATE memory_items
                            SET importance = MAX(?, importance * (1.0 - ?))
                            WHERE layer = ?
                        """, (min_importance, decay_rate, layer.value))

                # Decay connections
                cursor.execute("""
                    UPDATE memory_connections
                    SET strength = strength * (1.0 - COALESCE(
                        (SELECT decay_rate FROM connection_metadata 
                         WHERE connection_metadata.source_id = memory_connections.source_id
                         AND connection_metadata.target_id = memory_connections.target_id),
                        0.1  -- Default decay rate if no metadata
                    ))
                    WHERE (source_id, target_id) NOT IN (
                        SELECT source_id, target_id 
                        FROM connection_metadata
                        WHERE julianday('now') - julianday(last_reinforced) < ?
                    )
                """, (MEMORY_SETTINGS['decay']['protected_period'] / 24.0,))

                # Remove very weak connections
                cursor.execute("""
                    DELETE FROM memory_connections 
                    WHERE strength < ?
                """, (MEMORY_SETTINGS['connections']['min_strength'],))

        except Exception as e:
            logger.error(f"Error applying decay: {str(e)}", exc_info=True)

    def store_memory(self, content: str, topics: Optional[List[str]] = None, importance: float = None, source_type: str = None):
        """Store a new memory"""
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO memory_items (
                        id,
                        content,
                        layer,
                        created_at,
                        last_accessed,
                        access_count,
                        importance,
                        topics,
                        related_items,
                        decay_factor,
                        source_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(uuid.uuid4()),
                    content,
                    MemoryLayer.WORKING.value,  # New memories start in working memory
                    datetime.now(),
                    datetime.now(),
                    0,
                    importance or 0.5,  # Default importance
                    json.dumps(list(topics) if topics else []),
                    "[]",  # Empty related items
                    1.0,  # Initial decay factor
                    source_type or "conversation"
                ))
                cursor.connection.commit()
                
            logger.debug(f"Stored memory with topics {topics}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            return False

    def access_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Access a memory, updating its stats and potentially promoting it.
        Each layer maintains its own distinct role:
        - Working: Active conversation context
        - Short-term: Recently important information
        - Long-term: Stable, foundational knowledge
        """
        with self._lock:
            with db_manager.get_cursor() as cursor:
                # Simply update access stats without changing layers
                cursor.execute("""
                    UPDATE memory_items 
                    SET last_accessed = ?, 
                        access_count = access_count + 1,
                        importance = MIN(1.0, importance + ?)
                    WHERE id = ?
                """, (
                    datetime.now(),
                    MEMORY_SETTINGS['decay']['access_boost'],
                    memory_id
                ))
                
                # Get updated memory
                cursor.execute("SELECT * FROM memory_items WHERE id = ?", 
                             (memory_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                    
                # Check for promotion (but only for working->short-term)
                if row['layer'] == MemoryLayer.WORKING.value:
                    self._check_promotion(cursor, row)
                
                return self._row_to_memory_item(row)

    def _check_promotion(self, cursor, memory):
        """Check if memory should be promoted to next layer"""
        if memory['importance'] >= MEMORY_SETTINGS['promotion']['threshold']:
            current_layer = MemoryLayer(memory['layer'])
            if current_layer == MemoryLayer.WORKING:
                new_layer = MemoryLayer.SHORT_TERM
                cursor.execute("""
                    UPDATE memory_items 
                    SET layer = ? 
                    WHERE id = ?
                """, (new_layer.value, memory['id']))

    def cleanup_memories(self):
        """
        Periodic cleanup of memories based on age and importance.
        
        Cleanup Strategy:
        1. Age reduces tolerance for low importance
        2. High importance can extend memory lifetime
        3. Very high importance memories move to short-term instead of deletion
        """
        with self._lock:
            with db_manager.get_cursor() as cursor:
                now = datetime.now()
                settings = MEMORY_SETTINGS['layers']['working']
                cutoff = now - timedelta(seconds=settings['max_age'])
                
                # First try to promote important but old memories
                cursor.execute("""
                    UPDATE memory_items 
                    SET layer = ?, 
                        access_count = 1
                    WHERE layer = ?
                    AND last_accessed < ?
                    AND importance >= ?
                """, (
                    MemoryLayer.SHORT_TERM.value,
                    MemoryLayer.WORKING.value,
                    cutoff,
                    settings.get('promotion_threshold', 0.8)  # High importance memories get promoted
                ))
                
                # Then delete truly stale memories
                cursor.execute("""
                    DELETE FROM memory_items 
                    WHERE layer = ? 
                    AND last_accessed < ?
                    AND importance < ? + (
                        -- Increase importance threshold based on age
                        (julianday(?) - julianday(last_accessed)) * ?
                    )
                """, (
                    MemoryLayer.WORKING.value,
                    cutoff,
                    settings['deletion_threshold'],
                    now,
                    settings.get('age_penalty_rate', 0.1)  # Configurable age penalty
                ))

    def _row_to_memory_item(self, row: sqlite3.Row) -> MemoryItem:
        """Convert database row to MemoryItem"""
        return MemoryItem(
            content=row['content'],
            layer=MemoryLayer(row['layer']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_accessed=datetime.fromisoformat(row['last_accessed']),
            access_count=row['access_count'],
            importance=row['importance'],
            topics=list(json.loads(row['topics'])),
            related_items=list(json.loads(row['related_items']))
        ) 

    # Add memory connection methods
    def connect_memories(self, source_id: str, target_id: str, 
                        connection_type: str = "related"):
        """Create or strengthen a connection between memories"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_connections 
                (source_id, target_id, strength, last_reinforced, connection_type)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (source_id, target_id) DO UPDATE SET
                strength = MIN(1.0, strength + ?),
                last_reinforced = ?
            """, (
                source_id, target_id, 
                MEMORY_SETTINGS['decay']['connection_boost'],
                datetime.now(), connection_type,
                MEMORY_SETTINGS['decay']['connection_boost'],
                datetime.now()
            ))

    def update_domain_confidence(self, 
                               topics: set[str], 
                               interaction_quality: float,
                               source_type: str = 'conversation') -> None:
        """Update confidence in knowledge domains based on interactions."""
        with self._lock:
            with db_manager.get_cursor() as cursor:
                for topic in topics:
                    cursor.execute("""
                        INSERT INTO domain_expertise 
                        (topic, confidence, last_updated, interaction_count, 
                         source_diversity, quality_score)
                        VALUES (?, ?, CURRENT_TIMESTAMP, 1, 1, ?)
                        ON CONFLICT (topic) DO UPDATE SET
                        confidence = (confidence * interaction_count + ?) / (interaction_count + 1),
                        interaction_count = interaction_count + 1,
                        source_diversity = source_diversity + 
                            CASE WHEN NOT EXISTS (
                                SELECT 1 FROM memory_items 
                                WHERE source_type = ? AND topics LIKE '%' || ? || '%'
                            ) THEN 1 ELSE 0 END,
                        quality_score = (quality_score + ?) / 2,
                        last_updated = CURRENT_TIMESTAMP
                    """, (
                        topic, interaction_quality, interaction_quality,
                        interaction_quality, source_type, topic, interaction_quality
                    ))

    def get_domain_confidence(self, topics: set[str]) -> float:
        """
        Get average confidence across topics.
        
        Args:
            topics: Set of topics to check
            
        Returns:
            Average confidence score (0-1)
        """
        if not topics:
            return 0.0
            
        with db_manager.get_cursor() as cursor:
            placeholders = ','.join('?' * len(topics))
            cursor.execute(f"""
                SELECT AVG(
                    confidence * 
                    (1 + (source_diversity - 1) * ?) * 
                    quality_score
                ) as weighted_confidence
                FROM domain_expertise 
                WHERE topic IN ({placeholders})
            """, (EXPERTISE_SETTINGS['confidence']['diversity_weight'], *topics))
            
            result = cursor.fetchone()[0]
            return float(result) if result is not None else 0.0

    def get_topic_expertise(self, topic: str) -> Dict[str, Any]:
        """
        Get detailed expertise information for a topic.
        
        Returns:
            Dict containing:
            - confidence: Overall confidence
            - interaction_count: Number of interactions
            - source_diversity: Number of different sources
            - quality_score: Average quality of interactions
            - last_updated: Timestamp of last update
        """
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    confidence,
                    interaction_count,
                    source_diversity,
                    quality_score,
                    last_updated
                FROM domain_expertise
                WHERE topic = ?
            """, (topic,))
            
            row = cursor.fetchone()
            if row:
                return {
                    'confidence': row['confidence'],
                    'interaction_count': row['interaction_count'],
                    'source_diversity': row['source_diversity'],
                    'quality_score': row['quality_score'],
                    'last_updated': row['last_updated']
                }
            return None

    def get_strongest_domains(self, limit: int = EXPERTISE_SETTINGS['default_limit']) -> List[Dict[str, Any]]:
        """Get topics with highest expertise levels"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT 
                    topic,
                    confidence,
                    interaction_count,
                    source_diversity,
                    quality_score
                FROM domain_expertise
                ORDER BY confidence * quality_score * source_diversity DESC
                LIMIT ?
            """, (limit,))
            
            return [{
                'topic': row['topic'],
                'confidence': row['confidence'],
                'interactions': row['interaction_count'],
                'sources': row['source_diversity'],
                'quality': row['quality_score']
            } for row in cursor.fetchall()]

    def get_memory_metrics(self, memory_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a memory"""
        metrics = {
            'access': {},
            'patterns': {},
            'connections': {}
        }
        
        with db_manager.get_cursor() as cursor:
            # Get basic access metrics
            cursor.execute("""
                SELECT COUNT(*) as count, MAX(access_time) as last_time
                FROM memory_access_log 
                WHERE memory_id = ?
            """, (memory_id,))
            access_data = cursor.fetchone()
            metrics['access'] = {
                'count': access_data['count'],
                'last_access': access_data['last_time']
            }
            
            # Get connection metrics
            cursor.execute("""
                SELECT COUNT(*) as count,
                       AVG(strength) as avg_strength
                FROM memory_connections
                WHERE source_id = ? OR target_id = ?
            """, (memory_id, memory_id))
            conn_metrics = cursor.fetchone()
            metrics['connections'] = {
                'count': conn_metrics['count'],
                'average_strength': conn_metrics['avg_strength'] or 0.0
            }
        
        # Get access patterns
        patterns = memory_access_tracker.analyze_access_patterns(memory_id)
        metrics['patterns'] = patterns
        
        return metrics

    def _calculate_usage_value(self, memory_id: str) -> float:
        """Calculate usage value based on access patterns and connections"""
        with db_manager.get_cursor() as cursor:
            # Get access count
            cursor.execute("""
                SELECT COUNT(*) as access_count
                FROM memory_access_log
                WHERE memory_id = ?
            """, (memory_id,))
            access_count = cursor.fetchone()['access_count']
            
            # Get connection count
            cursor.execute("""
                SELECT COUNT(*) as conn_count
                FROM memory_connections
                WHERE source_id = ? OR target_id = ?
            """, (memory_id, memory_id))
            connection_count = cursor.fetchone()['conn_count']
            
            # Normalize counts
            normalized_access = min(access_count / MEMORY_SETTINGS['metrics']['max_access_norm'], 1.0)
            normalized_connections = min(connection_count / MEMORY_SETTINGS['metrics']['max_connection_norm'], 1.0)
            
            # Calculate weighted value
            return (
                normalized_access * MEMORY_SETTINGS['metrics']['access_weight'] +
                normalized_connections * MEMORY_SETTINGS['metrics']['connection_weight']
            )

    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """Get a memory by ID"""
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT id, content, layer, importance, created_at, 
                           last_accessed, access_count, topics
                    FROM memory_items 
                    WHERE id = ?
                """, (memory_id,))
                memory = cursor.fetchone()
                if not memory:
                    return None
                
                # Convert row to dict
                return {
                    'id': memory['id'],
                    'content': memory['content'],
                    'layer': memory['layer'],
                    'importance': memory['importance'],
                    'created_at': memory['created_at'],
                    'last_accessed': memory['last_accessed'],
                    'access_count': memory['access_count'],
                    'topics': json.loads(memory['topics']) if memory['topics'] else []
                }
        except Exception as e:
            logging.error(f"Error getting memory: {e}")
            return None

    def _check_layer_transition(self, memory_id: str):
        """Check if memory should transition between layers"""
        try:
            with db_manager.get_cursor() as cursor:
                # Get current memory state
                cursor.execute("""
                    SELECT layer, importance
                    FROM memory_items
                    WHERE id = ?
                """, (memory_id,))
                memory = cursor.fetchone()
                if not memory:
                    return
                    
                current_layer = memory['layer']
                importance = memory['importance']
                
                # Check for promotion
                if current_layer == MemoryLayer.WORKING.value:
                    if importance >= MEMORY_SETTINGS['layers']['working']['promotion_threshold']:
                        new_layer = MemoryLayer.SHORT_TERM.value
                        
                        cursor.execute("""
                            UPDATE memory_items
                            SET layer = ?
                            WHERE id = ?
                        """, (new_layer, memory_id))
                        
                        logging.info(f"Memory {memory_id} promoted to {new_layer}")
                        
        except Exception as e:
            logging.error(f"Error checking layer transition: {e}")

    def find_similar_memories(self, content: str, threshold: float = None) -> List[Dict]:
        """Find potentially duplicate memories"""
        try:
            threshold = threshold or MEMORY_SETTINGS['deduplication']['similarity_threshold']
            content_embedding = generate_embedding(content)
            
            with db_manager.get_cursor() as cursor:
                # Get recent memories to check
                cursor.execute("""
                    SELECT id, content, importance, topics, created_at
                    FROM memory_items 
                    WHERE datetime('now', '-1 hour') > created_at
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (MEMORY_SETTINGS['deduplication']['batch_size'],))
                
                similar_memories = []
                for memory in cursor.fetchall():
                    memory_embedding = generate_embedding(memory['content'])
                    similarity = np.dot(content_embedding, memory_embedding)
                    
                    if similarity >= threshold:
                        similar_memories.append({
                            'id': memory['id'],
                            'content': memory['content'],
                            'importance': memory['importance'],
                            'topics': json.loads(memory['topics']) if memory['topics'] else [],
                            'similarity': similarity
                        })
                
                return similar_memories
                
        except Exception as e:
            logging.error(f"Error finding similar memories: {e}")
            return []

    def deduplicate_memory(self, memory_id: str, similar_id: str, similarity: float = None) -> Optional[str]:
        """Merge two similar memories"""
        try:
            with db_manager.get_cursor() as cursor:
                # Get both memories
                cursor.execute("""
                    SELECT id, content, importance, topics, layer
                    FROM memory_items
                    WHERE id IN (?, ?)
                """, (memory_id, similar_id))
                memories = cursor.fetchall()
                logging.debug(f"Found memories to deduplicate: {len(memories)}")
                if len(memories) != 2:
                    logging.error(f"Could not find both memories: {memory_id}, {similar_id}")
                    return None
                    
                # Determine which to keep
                mem1, mem2 = memories
                keep_memory = mem1 if mem1['importance'] >= mem2['importance'] else mem2
                remove_memory = mem2 if mem1['importance'] >= mem2['importance'] else mem1
                logging.debug(f"Keeping memory {keep_memory['id']} (importance: {keep_memory['importance']})")
                logging.debug(f"Removing memory {remove_memory['id']} (importance: {remove_memory['importance']})")
                
                # Merge topics
                topics1 = json.loads(mem1['topics']) if mem1['topics'] else []
                topics2 = json.loads(mem2['topics']) if mem2['topics'] else []
                merged_topics = list(set(topics1 + topics2))
                logging.debug(f"Merged topics: {merged_topics}")
                
                # Update kept memory
                cursor.execute("""
                    UPDATE memory_items
                    SET topics = ?,
                        importance = MAX(importance, ?)
                    WHERE id = ?
                """, (
                    json.dumps(merged_topics),
                    remove_memory['importance'],
                    keep_memory['id']
                ))
                
                # Transfer any connections
                cursor.execute("""
                    UPDATE memory_connections
                    SET source_id = ?
                    WHERE source_id = ?
                """, (keep_memory['id'], remove_memory['id']))
                
                cursor.execute("""
                    UPDATE memory_connections
                    SET target_id = ?
                    WHERE target_id = ?
                """, (keep_memory['id'], remove_memory['id']))
                
                # Delete duplicate
                cursor.execute("DELETE FROM memory_items WHERE id = ?", 
                             (remove_memory['id'],))
                
                try:
                    # Record deduplication using same cursor/transaction
                    memory_access_tracker.record_deduplication(
                        keep_memory['id'], 
                        remove_memory['id'],
                        similarity or 0.0,
                        cursor=cursor
                    )
                except Exception as e:
                    logging.error(f"Failed to record deduplication access: {e}")
                    # Continue even if access tracking fails
                
                logging.info(f"Deduplicated memory {remove_memory['id']} into {keep_memory['id']}")
                return keep_memory['id']
                
        except Exception as e:
            logging.error(f"Error deduplicating memories: {e}")
            return None

    def update_domain_expertise(self, topic: str, interaction_data: Dict):
        """Update domain expertise based on interaction"""
        try:
            quality_score = interaction_data.get('quality_score', 
                                               EXPERTISE_SETTINGS['quality']['default'])
            source_type = interaction_data.get('source_type', 'unknown')
            
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR IGNORE INTO domain_expertise
                    (topic, confidence, quality_score, source_diversity)
                    VALUES (?, ?, ?, 1)
                """, (
                    topic,
                    EXPERTISE_SETTINGS['confidence']['default'],
                    quality_score
                ))
                
                cursor.execute("""
                    UPDATE domain_expertise
                    SET confidence = MIN(confidence + ?, ?),
                        quality_score = ?,
                        source_diversity = CASE 
                            WHEN source_type = ? THEN source_diversity 
                            ELSE source_diversity + 1 
                        END
                    WHERE topic = ?
                """, (
                    EXPERTISE_SETTINGS['confidence']['memory_weight'],
                    EXPERTISE_SETTINGS['confidence']['max'],
                    quality_score,
                    source_type,
                    topic
                ))
                cursor.execute("COMMIT")
                
        except Exception as e:
            logger.error(f"Error updating domain expertise: {str(e)}", exc_info=True)
            raise

# Global instance
memory_manager = MemoryManager() 