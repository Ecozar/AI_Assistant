"""
Test memory decay functionality
"""

import sys
import os
import json
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import unittest
import time
from datetime import datetime, timedelta
import logging
from AI_Project_Brain.memory_decay import memory_decay
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.memory_layers import MemoryLayer
from config import MEMORY_SETTINGS

class TestMemoryDecay(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        self.test_memories = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with db_manager.get_cursor() as cursor:
            # Create test memories
            for i in range(5):
                memory_id = f"test_decay_{i}"
                self.test_memories.append(memory_id)
                cursor.execute("""
                    INSERT INTO memory_items 
                    (id, content, importance, layer, access_count, last_accessed, created_at, topics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    f"Test content {i}",
                    1.0,  # Start with max importance
                    MemoryLayer.WORKING.value,
                    i,  # Varying access counts
                    current_time,
                    current_time,
                    json.dumps(['test_topic'])  # Add topics field
                ))
            
            # Create some connections
            cursor.execute("""
                INSERT INTO memory_connections
                (source_id, target_id, strength)
                VALUES (?, ?, ?)
            """, (
                self.test_memories[0],
                self.test_memories[1],
                0.5
            ))
    
    def test_protection_mechanisms(self):
        """Test that memories are properly protected"""
        protected = memory_decay._get_protected_memories()
        
        # Verify connection-based protection
        self.assertIn(self.test_memories[0], protected)
        self.assertIn(self.test_memories[1], protected)
        
        # Verify access-based protection
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE memory_items 
                SET access_count = ? 
                WHERE id = ?
            """, (
                MEMORY_SETTINGS['protection']['min_access_count'] + 1,
                self.test_memories[2]
            ))
        
        protected = memory_decay._get_protected_memories()
        self.assertIn(self.test_memories[2], protected)
    
    def test_decay_application(self):
        """Test that decay is properly applied"""
        # Get initial importance
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT importance FROM memory_items WHERE id = ?",
                (self.test_memories[3],)
            )
            initial_importance = cursor.fetchone()['importance']
        
        # Apply decay
        memory_decay._apply_decay(set())
        
        # Check decayed importance
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT importance FROM memory_items WHERE id = ?",
                (self.test_memories[3],)
            )
            decayed_importance = cursor.fetchone()['importance']
        
        self.assertLess(
            decayed_importance,
            initial_importance,
            "Importance should decrease after decay"
        )
    
    def test_minimum_importance(self):
        """Test that importance never goes below minimum threshold"""
        # Force multiple decay cycles to test floor
        for _ in range(10):  # Apply decay multiple times
            memory_decay._apply_decay(set())
        
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT importance FROM memory_items WHERE id = ?",
                (self.test_memories[0],)
            )
            final_importance = cursor.fetchone()['importance']
            
            # Should not decay below min_importance
            self.assertGreaterEqual(
                final_importance,
                MEMORY_SETTINGS['decay']['layers'][MemoryLayer.WORKING.value]['min_importance'],
                "Importance should not decay below minimum threshold"
            )

    def test_permanent_memory_protection(self):
        """Test that permanent memories never decay"""
        # Create a permanent memory
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        permanent_id = "test_decay_permanent"
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, importance, layer, access_count, last_accessed, created_at, topics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                permanent_id,
                "Permanent memory",
                1.0,
                MemoryLayer.PERMANENT.value,
                0,
                current_time,
                current_time,
                json.dumps(['test_topic'])
            ))
            
        # Apply decay
        memory_decay._apply_decay(set())
        
        # Verify importance unchanged
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "SELECT importance FROM memory_items WHERE id = ?",
                (permanent_id,)
            )
            importance = cursor.fetchone()['importance']
            self.assertEqual(importance, 1.0, "Permanent memories should not decay")
    
    def tearDown(self):
        """Clean up test data"""
        with db_manager.get_cursor() as cursor:
            cursor.execute(
                "DELETE FROM memory_items WHERE id LIKE 'test_decay_%'"
            )
            cursor.execute(
                "DELETE FROM memory_connections WHERE source_id LIKE 'test_decay_%'"
            )

if __name__ == '__main__':
    unittest.main() 