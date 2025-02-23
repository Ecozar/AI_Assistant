"""
MEMORY CONNECTIONS
----------------
Handles relationships between memory items with enhanced tracking.

Core Responsibilities:
1. Track memory relationships and their evolution
2. Measure connection strength and reinforcement
3. Support relationship-based memory retrieval
4. Enable memory clustering and organization
"""

import sys
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict, Set, List
import threading

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

@dataclass
class Connection:
    source_id: str
    target_id: str
    connection_type: str
    strength: float
    created_at: datetime
    last_accessed: Optional[datetime] = None

import logging
from typing import Set, List, Dict
import json
from datetime import datetime

from .memory_layers import MemoryItem
from .db_manager import db_manager
from config import MEMORY_SETTINGS, EXPERTISE_SETTINGS

@dataclass
class ConnectionMetadata:
    """Tracks detailed connection information"""
    last_reinforced: datetime
    reinforcement_count: int
    context_types: Set[str]  # e.g., 'conversation', 'search', 'learning'
    strength_history: List[Tuple[datetime, float]]
    decay_rate: float = 0.1

class MemoryConnections:
    def __init__(self):
        self._init_database()
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize connection tracking tables with enhanced metrics"""
        with db_manager.get_cursor() as cursor:
            # Enhance memory_connections table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_connections (
                    source_id TEXT,
                    target_id TEXT,
                    connection_type TEXT,
                    strength FLOAT,
                    reinforcement_count INTEGER DEFAULT 0,
                    last_reinforced TIMESTAMP,
                    creation_context TEXT,
                    metadata TEXT,  -- JSON field for flexible attributes
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY (source_id) REFERENCES memory_items(id),
                    FOREIGN KEY (target_id) REFERENCES memory_items(id)
                )
            """)
            
            # Add connection metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connection_metrics (
                    connection_id TEXT PRIMARY KEY,
                    source_id TEXT,
                    target_id TEXT,
                    total_activations INTEGER DEFAULT 0,
                    avg_activation_interval FLOAT,
                    last_activation TIMESTAMP,
                    pattern_confidence FLOAT DEFAULT 0.0,
                    FOREIGN KEY (source_id, target_id) 
                        REFERENCES memory_connections(source_id, target_id)
                )
            """)
            
            # Add metadata columns to memory_connections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connection_metadata (
                    source_id TEXT,
                    target_id TEXT,
                    last_reinforced TIMESTAMP,
                    reinforcement_count INTEGER DEFAULT 1,
                    context_types TEXT,  -- JSON array
                    strength_history TEXT,  -- JSON array of [timestamp, strength]
                    decay_rate FLOAT DEFAULT 0.1,
                    PRIMARY KEY (source_id, target_id),
                    FOREIGN KEY (source_id, target_id) 
                        REFERENCES memory_connections(source_id, target_id)
                        ON DELETE CASCADE
                )
            """)
    
    def connect_memories(self, source_id: str, target_id: str, 
                        connection_type: str = "related",
                        context: str = None,
                        metadata: Dict = None,
                        strength: float = None) -> bool:
        """Create or strengthen a connection between memories"""
        try:
            with self._lock, db_manager.get_cursor() as cursor:
                # Get current timestamp
                current_time = datetime.now()
                
                # Use provided strength or default
                initial_strength = strength if strength is not None else MEMORY_SETTINGS['connections']['initial_strength']
                
                # Insert or update connection
                cursor.execute("""
                    INSERT INTO memory_connections 
                    (source_id, target_id, connection_type, strength, 
                     reinforcement_count, last_reinforced, creation_context, metadata)
                    VALUES (?, ?, ?, ?, 1, ?, ?, ?)
                    ON CONFLICT (source_id, target_id) DO UPDATE SET
                        strength = MIN(1.0, ?),
                        reinforcement_count = memory_connections.reinforcement_count + 1,
                        last_reinforced = ?,
                        metadata = CASE 
                            WHEN ? IS NOT NULL THEN ?
                            ELSE memory_connections.metadata
                        END
                """, (
                    source_id, target_id, connection_type,
                    initial_strength,
                    current_time, context,
                    json.dumps(metadata) if metadata else None,
                    strength if strength is not None else 
                        f"MIN(1.0, memory_connections.strength + {MEMORY_SETTINGS['connections']['reinforcement_boost']})",
                    current_time,
                    json.dumps(metadata) if metadata else None,
                    json.dumps(metadata) if metadata else None
                ))
                
                # Update metrics
                connection_id = f"{source_id}:{target_id}"
                cursor.execute("""
                    INSERT INTO connection_metrics
                    (connection_id, source_id, target_id, total_activations, 
                     last_activation)
                    VALUES (?, ?, ?, 1, ?)
                    ON CONFLICT (connection_id) DO UPDATE SET
                        total_activations = connection_metrics.total_activations + 1,
                        avg_activation_interval = CASE 
                            WHEN connection_metrics.last_activation IS NOT NULL THEN
                                ((julianday(?) - julianday(connection_metrics.last_activation)) +
                                 (connection_metrics.avg_activation_interval * 
                                  connection_metrics.total_activations)) /
                                (connection_metrics.total_activations + 1)
                            ELSE NULL
                        END,
                        last_activation = ?
                """, (
                    connection_id, source_id, target_id, 
                    current_time, current_time, current_time
                ))
                
                # Then update the metadata
                cursor.execute("""
                    INSERT INTO connection_metadata 
                    (source_id, target_id, last_reinforced, context_types, strength_history)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(source_id, target_id) DO UPDATE SET
                    last_reinforced = ?,
                    reinforcement_count = reinforcement_count + 1,
                    context_types = json_set(COALESCE(context_types, '[]'), 
                                          '$[#]', ?),
                    strength_history = json_set(COALESCE(strength_history, '[]'),
                                             '$[#]', json_array(?, ?))
                """, (
                    source_id, target_id, current_time, 
                    json.dumps([context] if context else []),
                    json.dumps([[current_time.isoformat(), initial_strength]]),
                    current_time,
                    context,
                    current_time.isoformat(), initial_strength
                ))
                
                # Update domain expertise if connection is strong
                if strength and strength >= EXPERTISE_SETTINGS['clustering']['similarity_threshold']:
                    cursor.execute("""
                        SELECT DISTINCT d.domain_id 
                        FROM domain_memories d
                        WHERE d.memory_id IN (?, ?)
                    """, (source_id, target_id))
                    
                    for row in cursor.fetchall():
                        domain_id = row['domain_id']
                        self.logger.debug(f"Reinforcing domain {domain_id} due to strong connection")
                        
                        cursor.execute("""
                            UPDATE expertise_domains
                            SET confidence_level = MIN(
                                confidence_level + ?,
                                ?
                            )
                            WHERE domain_id = ?
                        """, (
                            EXPERTISE_SETTINGS['confidence']['memory_weight'],
                            EXPERTISE_SETTINGS['confidence']['max'],
                            domain_id
                        ))
                
                self.logger.debug(
                    f"Connected memories {source_id} -> {target_id} "
                    f"(type: {connection_type}, context: {context})"
                )
                return True
                
        except Exception as e:
            self.logger.error(f"Error connecting memories: {str(e)}", exc_info=True)
            return False

    def get_connected_memories(self, memory_id: str, min_strength: float = 0.3) -> List[Dict]:
        """Retrieve connected memories above strength threshold"""
        try:
            with self._lock, db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT mc.*, m.content, m.importance, m.layer,
                           cm.total_activations, cm.avg_activation_interval
                    FROM memory_connections mc
                    JOIN memory_items m ON mc.target_id = m.id
                    JOIN connection_metrics cm ON cm.connection_id = mc.source_id || ':' || mc.target_id
                    WHERE mc.source_id = ? AND mc.strength >= ?
                    ORDER BY mc.strength DESC
                """, (memory_id, min_strength))
                
                self.logger.debug(f"Retrieving connections for memory {memory_id}")
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Error retrieving connected memories: {e}")
            return []

    def find_memory_clusters(self, min_connections: int = 3) -> List[Set[str]]:
        """Find clusters of strongly connected memories"""
        try:
            with self._lock, db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT source_id, target_id
                    FROM memory_connections
                    WHERE strength >= ?
                """, (MEMORY_SETTINGS['connections']['cluster_threshold'],))
                
                # Build adjacency graph
                connections = cursor.fetchall()
                clusters = []
                processed = set()
                
                for source_id, target_id in connections:
                    if source_id in processed:
                        continue
                        
                    # Find connected component
                    cluster = {source_id, target_id}
                    queue = [source_id, target_id]
                    
                    while queue:
                        current = queue.pop(0)
                        for s, t in connections:
                            if s == current and t not in cluster:
                                cluster.add(t)
                                queue.append(t)
                            elif t == current and s not in cluster:
                                cluster.add(s)
                                queue.append(s)
                    
                    if len(cluster) >= min_connections:
                        clusters.append(cluster)
                        processed.update(cluster)
                
                self.logger.debug(f"Found {len(clusters)} memory clusters")
                return clusters
                
        except Exception as e:
            self.logger.error(f"Error finding memory clusters: {e}")
            return []

# Global instance
memory_connections = MemoryConnections() 