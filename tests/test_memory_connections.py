"""
MEMORY CONNECTIONS TESTS
-----------------------
Tests memory relationship tracking and clustering functionality.
"""

import sys
import os
import unittest
import logging
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.memory_connections import memory_connections
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.memory_decay import memory_decay
from config import MEMORY_SETTINGS
from reset_database import reset_memory_connections

class TestMemoryConnections(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Reset memory connection tables first
        reset_memory_connections()
        
        self.test_memories = [f"test_memory_{i}" for i in range(4)]
        
        # Create test memories in database
        with db_manager.get_cursor() as cursor:
            for memory_id in self.test_memories:
                cursor.execute("""
                    INSERT OR IGNORE INTO memory_items 
                    (id, content, layer, importance, created_at)
                    VALUES (?, ?, 'working', 0.5, ?)
                """, (memory_id, f"Content for {memory_id}", datetime.now()))

    def test_connect_memories(self):
        """Test basic connection creation and reinforcement"""
        # Create initial connection
        success = memory_connections.connect_memories(
            self.test_memories[0],
            self.test_memories[1],
            connection_type="test",
            context="test_context"
        )
        self.assertTrue(success)
        
        # Verify connection
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT strength, reinforcement_count 
                FROM memory_connections
                WHERE source_id = ? AND target_id = ?
            """, (self.test_memories[0], self.test_memories[1]))
            conn = cursor.fetchone()
            
            self.assertEqual(
                conn['strength'],
                MEMORY_SETTINGS['connections']['initial_strength']
            )
            self.assertEqual(conn['reinforcement_count'], 1)

    def test_get_connected_memories(self):
        """Test retrieval of connected memories"""
        # Create test connections
        memory_connections.connect_memories(
            self.test_memories[0],
            self.test_memories[1],
            strength=0.8
        )
        memory_connections.connect_memories(
            self.test_memories[0],
            self.test_memories[2],
            strength=0.3
        )
        
        # Test retrieval with threshold
        connections = memory_connections.get_connected_memories(
            self.test_memories[0],
            min_strength=0.5
        )
        self.assertEqual(len(connections), 1)
        self.assertEqual(connections[0]['target_id'], self.test_memories[1])

    def test_find_memory_clusters(self):
        """Test memory cluster detection"""
        # Create a cluster of connected memories
        memory_connections.connect_memories(
            self.test_memories[0],
            self.test_memories[1],
            strength=0.8
        )
        memory_connections.connect_memories(
            self.test_memories[1],
            self.test_memories[2],
            strength=0.8
        )
        memory_connections.connect_memories(
            self.test_memories[2],
            self.test_memories[0],
            strength=0.8
        )
        
        # Find clusters
        clusters = memory_connections.find_memory_clusters(min_connections=3)
        self.assertEqual(len(clusters), 1)
        
        # Verify cluster contents
        cluster = clusters[0]
        self.assertEqual(len(cluster), 3)
        self.assertTrue(all(mem in cluster for mem in self.test_memories[:3]))

    def test_connection_metadata(self):
        """Test connection metadata tracking"""
        source_id = self.test_memories[0]
        target_id = self.test_memories[1]
        
        # Create connection with context
        memory_connections.connect_memories(
            source_id, 
            target_id,
            strength=0.7,
            context="test_context"
        )
        
        # Verify metadata was created
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM connection_metadata 
                WHERE source_id = ? AND target_id = ?
            """, (source_id, target_id))
            metadata = cursor.fetchone()
            
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata['reinforcement_count'], 1)
            
            # Check context types
            context_types = json.loads(metadata['context_types'])
            self.assertIn("test_context", context_types)
            
            # Check strength history
            history = json.loads(metadata['strength_history'])
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0][1], 0.7)  # Check initial strength

    def test_connection_reinforcement(self):
        """Test connection reinforcement with metadata updates"""
        source_id = self.test_memories[0]
        target_id = self.test_memories[1]
        
        # Create initial connection
        memory_connections.connect_memories(
            source_id, target_id, 
            strength=0.5,
            context="context1"
        )
        
        # Reinforce with different context
        memory_connections.connect_memories(
            source_id, target_id,
            strength=0.6,
            context="context2"
        )
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT m.*, c.strength 
                FROM connection_metadata m
                JOIN memory_connections c 
                    ON m.source_id = c.source_id 
                    AND m.target_id = c.target_id
                WHERE m.source_id = ? AND m.target_id = ?
            """, (source_id, target_id))
            result = cursor.fetchone()
            
            self.assertEqual(result['reinforcement_count'], 2)
            self.assertGreater(result['strength'], 0.5)  # Should be reinforced
            
            contexts = json.loads(result['context_types'])
            self.assertIn("context1", contexts)
            self.assertIn("context2", contexts)

    def test_decay_with_protection(self):
        """Test connection decay with protection for recent reinforcement"""
        source_id = self.test_memories[0]
        target_id = self.test_memories[1]
        
        # Create connection with old timestamp
        memory_connections.connect_memories(
            source_id, target_id,
            strength=0.8
        )
        
        # Artificially age the connection
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE connection_metadata
                SET last_reinforced = datetime('now', '-2 days')
                WHERE source_id = ? AND target_id = ?
            """, (source_id, target_id))
        
        # Get initial strength
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT strength FROM memory_connections WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            )
            initial_strength = cursor.fetchone()['strength']
        
        # Apply decay
        memory_decay._apply_decay(set())
        
        # Check strength after decay
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT strength FROM memory_connections WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            )
            decayed_strength = cursor.fetchone()['strength']
            
            # Should be decayed
            self.assertLess(decayed_strength, initial_strength)
            
        # Reinforce connection
        memory_connections.connect_memories(
            source_id, target_id,
            strength=0.9
        )
        
        # Apply decay again
        memory_decay._apply_decay(set())
        
        # Check strength - should be protected due to recent reinforcement
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT strength FROM memory_connections WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            )
            protected_strength = cursor.fetchone()['strength']
            self.assertEqual(protected_strength, 0.9)  # Should not decay

    def test_weak_connection_cleanup(self):
        """Test removal of very weak connections"""
        source_id = self.test_memories[0]
        target_id = self.test_memories[1]
        
        # Create weak connection
        memory_connections.connect_memories(
            source_id, target_id,
            strength=MEMORY_SETTINGS['connections']['min_strength'] + 0.1
        )
        
        # Apply strong decay
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE memory_connections 
                SET strength = strength * 0.1
                WHERE source_id = ? AND target_id = ?
            """, (source_id, target_id))
        
        # Apply decay cleanup
        memory_decay._apply_decay(set())
        
        # Verify connection was removed
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) as count FROM memory_connections WHERE source_id = ? AND target_id = ?",
                (source_id, target_id)
            )
            self.assertEqual(cursor.fetchone()['count'], 0)

    def tearDown(self):
        """Clean up test data"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'")
            cursor.execute("DELETE FROM connection_metadata WHERE source_id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main() 