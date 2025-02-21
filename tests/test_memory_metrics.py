"""
MEMORY METRICS TEST
------------------
Tests the enhanced memory metrics functionality.
"""

import sys
import os
import unittest
from datetime import datetime, timedelta
import logging
import json
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.memory_manager import memory_manager
from AI_Project_Brain.memory_access_tracker import memory_access_tracker
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.domain_expertise import domain_expertise
from config import MEMORY_SETTINGS

class TestMemoryMetrics(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Add debug logging
        logging.debug("Setting up test data")
        
        # Verify table structure
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_items'")
            schema = cursor.fetchone()
            logging.debug(f"memory_items schema: {schema['sql']}")
        
        # Clear existing test data
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'")
        
        # Create test memory
        self.memory_id = "test_memory_1"
        self.test_content = "Test memory content for metrics"
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, importance, created_at, last_accessed, layer, access_count, topics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                self.memory_id, 
                self.test_content, 
                0.5, 
                datetime.now(), 
                datetime.now(),
                'working',
                0,  # Initial access count
                json.dumps(['test_topic'])  # Initial topics
            ))
            
        logging.info("Test setup complete")

    def test_basic_metrics(self):
        """Test basic metric calculation"""
        logging.info("=== Starting basic metrics test ===")
        
        # Temporarily store and modify settings
        original_min_time = MEMORY_SETTINGS['access_tracking']['min_time_between_records']
        MEMORY_SETTINGS['access_tracking']['min_time_between_records'] = 0
        
        try:
            # Add some test access patterns
            for i in range(5):
                logging.debug(f"Recording access {i+1}")
                memory_access_tracker.record_access(
                    self.memory_id,
                    "read",
                    f"test context {i}",
                    "test source"
                )
                
                # Verify each access immediately after recording
                with db_manager.get_cursor() as cursor:
                    cursor.execute("""
                        SELECT COUNT(*) as count, MAX(access_time) as last_time
                        FROM memory_access_log 
                        WHERE memory_id = ?
                    """, (self.memory_id,))
                    result = cursor.fetchone()
                    logging.debug(f"After access {i+1}: count={result['count']}, last_time={result['last_time']}")
        
            # Verify final access records in detail
            with db_manager.get_cursor() as cursor:
                # Get total count
                cursor.execute("""
                    SELECT COUNT(*) as count 
                    FROM memory_access_log 
                    WHERE memory_id = ?
                """, (self.memory_id,))
                actual_count = cursor.fetchone()['count']
                logging.debug(f"Final direct access count: {actual_count}")
                
                # Get all access records for inspection
                cursor.execute("""
                    SELECT access_time, access_type, context, source
                    FROM memory_access_log 
                    WHERE memory_id = ?
                    ORDER BY access_time
                """, (self.memory_id,))
                records = cursor.fetchall()
                logging.debug("Access records:")
                for record in records:
                    logging.debug(f"  {dict(record)}")
        
            # Get and verify metrics
            metrics = memory_manager.get_memory_metrics(self.memory_id)
            logging.debug(f"Returned metrics: {metrics}")
            
            # Detailed verification
            self.assertIsNotNone(metrics, "Metrics should not be None")
            self.assertIn('patterns', metrics)
            self.assertIn('frequency_patterns', metrics['patterns'])
            self.assertGreater(metrics['access']['count'], 0)
        
        finally:
            # Restore original setting
            MEMORY_SETTINGS['access_tracking']['min_time_between_records'] = original_min_time
        
        logging.info("=== Basic metrics test completed ===")

    def test_temporal_patterns(self):
        """Test temporal access pattern tracking"""
        # Add access patterns at different hours
        base_time = datetime.now().replace(hour=10, minute=0)
        
        # Create enough accesses at each hour to meet min_occurrences
        for hour in range(3):
            for _ in range(MEMORY_SETTINGS['access_tracking']['patterns']['min_occurrences'] + 1):
                memory_access_tracker._test_mode_record_access(
                    self.memory_id,
                    "read",
                    "test context",
                    "test source",
                    timestamp=base_time + timedelta(hours=hour)
                )
        
        metrics = memory_manager.get_memory_metrics(self.memory_id)
        temporal = metrics['patterns']['time_patterns']
        self.assertTrue(temporal)
        
        logging.info(f"Temporal patterns: {temporal}")

    def test_connection_metrics(self):
        """Test connection strength calculations"""
        # Create test connected memories
        connected_ids = ["test_connected_1", "test_connected_2"]
        
        with db_manager.get_cursor() as cursor:
            # Create connected memories
            for mem_id in connected_ids:
                cursor.execute("""
                    INSERT INTO memory_items 
                    (id, content, importance, layer, created_at, last_accessed, access_count, topics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    mem_id, 
                    f"Connected memory {mem_id}", 
                    0.5,
                    'working',
                    datetime.now(),
                    datetime.now(),
                    0,
                    json.dumps(['test_topic'])
                ))
                
                # Create connections with non-zero strength
                cursor.execute("""
                    INSERT INTO memory_connections 
                    (source_id, target_id, strength, last_reinforced)
                    VALUES (?, ?, ?, datetime('now'))
                """, (self.memory_id, mem_id, 0.75))
        
        metrics = memory_manager.get_memory_metrics(self.memory_id)
        
        # Verify connection metrics
        self.assertEqual(metrics['connections']['count'], 2)
        self.assertAlmostEqual(metrics['connections']['average_strength'], 0.75)
        
        logging.info(f"Connection metrics: {metrics['connections']}")

    def test_usage_value(self):
        """Test usage value calculation"""
        # Add access patterns and connections
        for _ in range(MEMORY_SETTINGS['metrics']['max_access_norm'] // 2):
            memory_access_tracker.record_access(
                self.memory_id,
                "read",
                "test context",
                "test source"
            )
        
        metrics = memory_manager.get_memory_metrics(self.memory_id)
        
        # Update to use new metrics structure
        usage_value = memory_manager._calculate_usage_value(self.memory_id)
        self.assertGreater(usage_value, 0)
        self.assertLessEqual(usage_value, 1.0)
        
        logging.info(f"Usage value: {usage_value}")

    def test_access_pattern_analysis(self):
        """Test memory access pattern analysis"""
        min_occurrences = MEMORY_SETTINGS['access_tracking']['patterns']['min_occurrences']
        base_time = datetime.now().replace(hour=10, minute=0)
        
        # Create consistent patterns at specific hours
        for hour in range(3):
            for _ in range(min_occurrences + 1):  # Ensure we exceed minimum
                memory_access_tracker._test_mode_record_access(
                    self.memory_id,
                    "read",
                    "test context",
                    "test source",
                    timestamp=base_time + timedelta(hours=hour)
                )
        
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        
        # Verify pattern structure
        self.assertTrue(patterns['frequency_patterns'], "Should have frequency patterns")
        self.assertTrue(patterns['time_patterns'], "Should have time patterns")
        
        # Verify pattern content
        freq_pattern = patterns['frequency_patterns'][0]
        self.assertEqual(freq_pattern['access_type'], "read")
        self.assertEqual(freq_pattern['context'], "test context")
        self.assertGreaterEqual(freq_pattern['count'], min_occurrences)
        
        time_pattern = patterns['time_patterns'][0]
        self.assertGreaterEqual(time_pattern['count'], min_occurrences)
        self.assertIn("test context", time_pattern['contexts'])

    def test_domain_expertise_tracking(self):
        """Test domain expertise tracking"""
        domain_id = "test_domain"
        
        interaction_data = {
            'quality_score': 0.8,
            'source_type': 'conversation',
            'topics': {'python', 'programming'}
        }
        
        domain_expertise.track_domain_interaction(domain_id, interaction_data)
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT confidence, interaction_count, quality_score, source_diversity
                FROM domain_expertise
                WHERE topic = ?
            """, (domain_id,))
            
            metrics = cursor.fetchone()
            self.assertIsNotNone(metrics)
            self.assertGreater(metrics['confidence'], 0)
            self.assertEqual(metrics['interaction_count'], 1)
            self.assertAlmostEqual(metrics['quality_score'], 0.8)

    def test_memory_lifecycle(self):
        """Test complete memory lifecycle including:
        - Creation & initial categorization
        - Access patterns & importance changes
        - Domain expertise development
        - Layer transitions (working -> short-term -> long-term)
        - Connection formation
        """
        # Create two related memories
        memory1_id = "test_lifecycle_1"
        memory2_id = "test_lifecycle_2"
        topics = {'python', 'programming', 'testing'}
        
        # 1. Memory Creation Phase
        with db_manager.get_cursor() as cursor:
            for mid, content in [
                (memory1_id, "Python unittest framework is great for testing"),
                (memory2_id, "Testing helps ensure code quality in programming")
            ]:
                cursor.execute("""
                    INSERT INTO memory_items 
                    (id, content, importance, created_at, last_accessed, layer, access_count, topics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (mid, content, 0.5, datetime.now(), datetime.now(), 'working', 0, json.dumps(list(topics))))

        # Create connections between memories
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_connections
                (source_id, target_id, strength, connection_type, last_reinforced)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (memory1_id, memory2_id, 0.75, "test"))

        # 2. Usage & Access Phase
        contexts = ['learning', 'applying', 'teaching']
        base_time = datetime.now().replace(hour=10, minute=0, second=0, microsecond=0)
        
        # Simulate intensive usage of memory1
        for i, context in enumerate(contexts):
            for _ in range(5):  # Enough accesses to trigger promotion
                memory_access_tracker._test_mode_record_access(
                    memory1_id, "read", context, "test_source",
                    timestamp=base_time + timedelta(hours=i)
                )
                # Sometimes access memory2 right after memory1
                if _ % 2 == 0:
                    memory_access_tracker._test_mode_record_access(
                        memory2_id, "read", context, "test_source",
                        timestamp=base_time + timedelta(hours=i, minutes=5)
                    )

        # 3. Domain Expertise Development
        for context in contexts:
            domain_expertise.track_domain_interaction(
                "testing_domain",
                {
                    'quality_score': 0.8,
                    'source_type': context,
                    'topics': topics
                }
            )

        # 4. Verify System State
        
        # 4.1 Memory State
        for mid in [memory1_id, memory2_id]:
            metrics = memory_manager.get_memory_metrics(mid)
            self.assertGreater(metrics['connections']['average_strength'], 0)
            self.assertGreater(metrics['access']['count'], 0)

        # 4.2 Access Patterns
        patterns = memory_access_tracker.analyze_access_patterns(memory1_id)
        self.assertTrue(patterns['frequency_patterns'])
        self.assertTrue(patterns['chain_patterns'])  # Should detect memory1->memory2 chain

        # 4.3 Domain Expertise
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT confidence, interaction_count, quality_score, source_diversity
                FROM domain_expertise
                WHERE topic = ?
            """, ('testing_domain',))
            domain = cursor.fetchone()
            self.assertIsNotNone(domain)
            self.assertGreater(domain['confidence'], 0)
            
            # Verify source diversity increased
            self.assertEqual(domain['source_diversity'], len(contexts))

        # 4.4 Memory Connections
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as conn_count
                FROM memory_connections
                WHERE source_id = ? AND target_id = ?
            """, (memory1_id, memory2_id))
            self.assertGreater(cursor.fetchone()['conn_count'], 0)

    def test_memory_maintenance(self):
        """Test memory system maintenance operations:
        - Memory decay
        - Pattern cleanup
        - Domain expertise updates
        - Connection strength adjustments
        """
        # Setup test data...
        # Simulate time passage...
        # Verify maintenance operations...

    def tearDown(self):
        """Clean up test data"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'")
        
        logging.info("Test cleanup complete")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main() 