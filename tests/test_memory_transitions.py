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

from AI_Project_Brain.memory_access_tracker import memory_access_tracker
from AI_Project_Brain.memory_layers import MemoryLayer
from AI_Project_Brain.domain_expertise import domain_expertise
from AI_Project_Brain.db_manager import db_manager
from config import MEMORY_SETTINGS
from AI_Project_Brain.memory_transition_manager import memory_transition_manager

class TestMemoryTransitions(unittest.TestCase):
    def setUp(self):
        logging.getLogger().setLevel(logging.DEBUG)
        self.memory_id = "test_transition_memory"
        self.domain_id = "test_domain_python"
        self.clean_test_data()
        
        # Create test memory in working memory with topics
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, layer, importance, created_at, topics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.memory_id,
                "Test memory content about Python programming",
                MemoryLayer.WORKING.value,
                0.5,
                datetime.now(),
                json.dumps(["python", "programming", "learning"])  # These look unique...
            ))
    
    def test_access_based_promotion(self):
        """Test that memories are promoted based on access patterns"""
        # Create significant access pattern
        test_times = [
            "2024-02-01 09:00:00",
            "2024-02-01 09:30:00",
            "2024-02-01 10:00:00"
        ]
        
        # Record multiple accesses
        for timestamp in test_times:
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                "learning_context",
                "test_source",
                timestamp=dt
            )
        
        # Trigger layer transition check
        memory_transition_manager.check_transitions(self.memory_id)
        
        # Verify promotion to short-term memory
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT layer 
                FROM memory_items 
                WHERE id = ?
            """, (self.memory_id,))
            current_layer = cursor.fetchone()['layer']
            
        self.assertEqual(current_layer, MemoryLayer.SHORT_TERM.value,
                        "Memory should be promoted to short-term based on access pattern")
    
    def test_domain_expertise_influence(self):
        """Test that domain expertise affects layer transitions"""
        # Set up domain with high confidence
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO expertise_domains 
                (domain_id, confidence_level, topic_cluster)
                VALUES (?, ?, ?)
            """, (
                self.domain_id,
                0.8,  # High confidence
                '["python", "programming"]'
            ))
        
        # Create memory access pattern
        test_times = [
            "2024-02-01 09:00:00",
            "2024-02-01 14:00:00"
        ]
        
        for timestamp in test_times:
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                "python_learning",  # Match domain
                "test_source",
                timestamp=dt
            )
        
        # Trigger layer transition check
        memory_transition_manager.check_transitions(self.memory_id)
        
        # Verify faster promotion due to domain expertise
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT layer, importance
                FROM memory_items 
                WHERE id = ?
            """, (self.memory_id,))
            memory = cursor.fetchone()
        
        self.assertEqual(memory['layer'], MemoryLayer.SHORT_TERM.value,
                        "Memory should be promoted based on domain expertise")
        self.assertGreater(memory['importance'], 0.5,
                          "Memory importance should increase with domain confidence")
    
    def test_coherent_transitions(self):
        """Test that related memories transition together"""
        # Create two related memories with shared topics
        related_memory_id = "test_related_memory"
        with db_manager.get_cursor() as cursor:
            # Create related memory
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, layer, importance, created_at, topics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                related_memory_id,
                "Related test content about Python classes",
                MemoryLayer.WORKING.value,
                0.5,
                datetime.now(),
                json.dumps(["python", "classes", "oop"])  # Related topics
            ))
            
            # Create connection in same transaction
            cursor.execute("""
                INSERT INTO memory_connections
                (source_id, target_id, connection_type, strength)
                VALUES (?, ?, ?, ?)
            """, (
                self.memory_id,
                related_memory_id,
                "related",
                0.9  # Strong connection above threshold
            ))
        
        # Now do the access pattern test
        test_times = [
            "2024-02-01 09:00:00",
            "2024-02-01 09:30:00",
            "2024-02-01 10:00:00"
        ]
        
        for timestamp in test_times:
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                "test_context",
                "test_source",
                timestamp=dt
            )
        
        # Trigger layer transition check
        memory_transition_manager.check_transitions(self.memory_id)
        
        # Verify both memories transitioned
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT id, layer
                FROM memory_items 
                WHERE id IN (?, ?)
            """, (self.memory_id, related_memory_id))
            memories = cursor.fetchall()
            
        for memory in memories:
            self.assertEqual(memory['layer'], MemoryLayer.SHORT_TERM.value,
                           f"Memory {memory['id']} should transition with its related memory")
    
    def test_last_access_tracking(self):
        """Test that last access time is properly tracked"""
        current_time = datetime.now()  # Store current time before access
        
        # Record an access
        memory_access_tracker._test_mode_record_access(
            self.memory_id,
            "read",
            "test_context",
            "test_source",
            timestamp=current_time  # Use our stored time
        )
        
        # Verify last_accessed was updated
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT last_accessed
                FROM memory_items 
                WHERE id = ?
            """, (self.memory_id,))
            memory = cursor.fetchone()
            
            self.assertIsNotNone(memory['last_accessed'],
                "last_accessed should be updated on access")
            
            # Convert to datetime for comparison
            last_accessed = datetime.fromisoformat(memory['last_accessed'].replace('Z', '+00:00'))
            self.assertEqual(current_time.replace(microsecond=0),
                last_accessed.replace(microsecond=0),
                "last_accessed should match access time")
    
    def clean_test_data(self):
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%'")
            cursor.execute("DELETE FROM expertise_domains WHERE domain_id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'")

    def tearDown(self):
        self.clean_test_data()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main(verbosity=2) 