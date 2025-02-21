"""
KNOWLEDGE DEPTH TESTS
--------------------
Tests the knowledge depth analysis functionality.
"""

import sys
import os
import unittest
import logging
from datetime import datetime
import uuid
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.knowledge_depth import knowledge_depth
from AI_Project_Brain.domain_expertise import domain_expertise
from AI_Project_Brain.db_manager import db_manager
from config import KNOWLEDGE_DEPTH_SETTINGS

class TestKnowledgeDepth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logging.info("=== Setting up knowledge depth test environment ===")
        # Initialize database
        with db_manager.get_cursor() as cursor:
            cursor.executescript("""
                DROP TABLE IF EXISTS expertise_domains;
                
                CREATE TABLE expertise_domains (
                    domain_id TEXT PRIMARY KEY,
                    topic_cluster TEXT NOT NULL,
                    confidence_level FLOAT NOT NULL DEFAULT 0.0,
                    first_emergence TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    evidence_sources TEXT
                );
            """)

    def setUp(self):
        """Set up test data"""
        logging.info("=== Setting up knowledge depth test ===")
        self.domain_id = self._create_test_domain()
        
    def _create_test_domain(self) -> str:
        """Create a test domain with some knowledge"""
        # Create domain with varying complexity
        test_content = [
            ("Basic Python syntax includes variables and loops", 0.5),
            ("Advanced Python concepts include decorators and metaclasses", 0.8),
            ("Python's metaclass system allows dynamic class creation and modification through __new__ and __init_subclass__ hooks", 0.9)
        ]
        
        domain_id = None
        with db_manager.get_cursor() as cursor:
            for i, (content, quality) in enumerate(test_content):
                memory_id = f"test_mem_{i}"
                # Insert memory with all required fields including topics
                cursor.execute("""
                    INSERT INTO memory_items 
                    (id, content, importance, layer, access_count, created_at, last_accessed, topics)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
                """, (
                    memory_id, 
                    content, 
                    quality, 
                    'long_term', 
                    1,
                    json.dumps(['python', 'programming'])  # Add topics as JSON array
                ))
                
                # Create connections between memories
                if i > 0:
                    cursor.execute("""
                        INSERT INTO memory_connections (source_id, target_id, strength)
                        VALUES (?, ?, ?)
                    """, (f"test_mem_{i-1}", memory_id, 0.8))
        
            # Create domain with UUID
            domain_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO expertise_domains 
                (domain_id, topic_cluster, confidence_level, first_emergence, last_activity, evidence_sources)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, (
                domain_id, 
                json.dumps(['python', 'programming']), 
                0.7,
                json.dumps(['document', 'document', 'document'])
            ))
            
            # Link memories to domain
            for i in range(len(test_content)):
                cursor.execute("""
                    INSERT INTO domain_memories (domain_id, memory_id)
                    VALUES (?, ?)
                """, (domain_id, f"test_mem_{i}"))
        
        return domain_id
        
    def test_depth_analysis(self):
        """Test basic depth analysis"""
        logging.info("Testing depth analysis")
        
        # Get depth scores
        scores = knowledge_depth.analyze_depth(self.domain_id)
        
        # Verify score structure
        self.assertIn('vertical_depth', scores)
        self.assertIn('horizontal_breadth', scores)
        self.assertIn('application_capability', scores)
        self.assertIn('overall_depth', scores)
        
        # Verify score ranges use config values
        min_depth = KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth']
        max_depth = KNOWLEDGE_DEPTH_SETTINGS['analysis']['max_depth']
        
        for score_name, value in scores.items():
            self.assertGreaterEqual(value, min_depth)
            self.assertLessEqual(value, max_depth)
            logging.debug(f"{score_name}: {value}")
            
        # Verify complexity threshold from config
        self.assertGreater(
            scores['vertical_depth'], 
            KNOWLEDGE_DEPTH_SETTINGS['thresholds']['complexity']
        )
        
    def test_settings_boundaries(self):
        """Test that settings boundaries are respected"""
        logging.info("Testing knowledge depth settings boundaries")
        
        with db_manager.get_cursor() as cursor:
            # Test min/max complexity
            content_low = "the " * 10  # Very low complexity
            content_high = " ".join([f"word{i}" for i in range(100)])  # High complexity
            
            # Insert test memories with different complexities
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, importance, layer, access_count, created_at, last_accessed, topics)
                VALUES 
                (?, ?, 0.1, 'long_term', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, ('test_low', content_low, json.dumps(['test'])))
            
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, importance, layer, access_count, created_at, last_accessed, topics)
                VALUES 
                (?, ?, 0.9, 'long_term', 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, ?)
            """, ('test_high', content_high, json.dumps(['test'])))
            
            # Link to test domain
            cursor.execute("""
                INSERT INTO domain_memories (domain_id, memory_id)
                VALUES (?, ?), (?, ?)
            """, (self.domain_id, 'test_low', self.domain_id, 'test_high'))
            
            # Get depth scores
            scores = knowledge_depth.analyze_depth(self.domain_id)
            
            self.assertGreaterEqual(scores['vertical_depth'], 
                                  KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth'])
            self.assertLessEqual(scores['vertical_depth'], 
                               KNOWLEDGE_DEPTH_SETTINGS['analysis']['max_depth'])

    def test_weight_calculations(self):
        """Test that weights are properly applied"""
        logging.info("Testing knowledge depth weight calculations")
        
        # Create test data with known scores
        with db_manager.get_cursor() as cursor:
            # Insert test memories with unique IDs
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, importance, layer, created_at, last_accessed, access_count, topics)
                VALUES 
                ('test_weight_1', 'Test content 1', 0.5, 'long_term', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?),
                ('test_weight_2', 'Test content 2', 0.6, 'long_term', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?)
            """, (json.dumps(['test']), json.dumps(['test'])))
            
            # Insert domain memories
            cursor.execute("""
                INSERT INTO domain_memories (domain_id, memory_id)
                VALUES (?, ?), (?, ?)
            """, (self.domain_id, 'test_weight_1', self.domain_id, 'test_weight_2'))
            
            # Insert test connection with high strength
            cursor.execute("""
                INSERT INTO memory_connections (source_id, target_id, strength)
                VALUES (?, ?, ?)
            """, ('test_weight_1', 'test_weight_2', 0.8))
        
        scores = knowledge_depth.analyze_depth(self.domain_id)
        
        # Verify weights sum to 1
        weights = KNOWLEDGE_DEPTH_SETTINGS['weights']
        self.assertAlmostEqual(
            sum(weights.values()), 
            1.0,
            msg="Weights should sum to 1.0"
        )
        
        # Verify overall score calculation
        expected_overall = (
            scores['vertical_depth'] * weights['vertical_depth'] +
            scores['horizontal_breadth'] * weights['horizontal_breadth'] +
            scores['application_capability'] * weights['application']
        )
        self.assertAlmostEqual(scores['overall_depth'], expected_overall)

    def test_thresholds(self):
        """Test that thresholds are properly applied"""
        logging.info("Testing knowledge depth thresholds")
        
        with db_manager.get_cursor() as cursor:
            # Add test memories with application content (need at least 4 to exceed 0.4 threshold)
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, importance, layer, created_at, last_accessed, access_count, topics)
                VALUES 
                ('test_app_1', 'Here is an example of how to use this in practice...', 0.8, 'long_term', 
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?),
                ('test_app_2', 'A common application of this concept is...', 0.8, 'long_term',
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?),
                ('test_app_3', 'This can be used in cases where...', 0.8, 'long_term',
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?),
                ('test_app_4', 'Another example of practical application...', 0.8, 'long_term',
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?),
                ('test_app_5', 'Here is a use case demonstrating...', 0.8, 'long_term',
                 CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, ?)
            """, (json.dumps(['test']), json.dumps(['test']), json.dumps(['test']), 
                  json.dumps(['test']), json.dumps(['test'])))
            
            # Create connections
            cursor.execute("""
                INSERT INTO memory_connections (source_id, target_id, strength)
                VALUES 
                ('test_app_1', 'test_app_2', ?),
                ('test_app_2', 'test_app_3', ?),
                ('test_app_3', 'test_app_4', ?),
                ('test_app_4', 'test_app_5', ?)
            """, (
                KNOWLEDGE_DEPTH_SETTINGS['thresholds']['connection'] + 0.1,
                KNOWLEDGE_DEPTH_SETTINGS['thresholds']['connection'] + 0.1,
                KNOWLEDGE_DEPTH_SETTINGS['thresholds']['connection'] + 0.1,
                KNOWLEDGE_DEPTH_SETTINGS['thresholds']['connection'] + 0.1
            ))
            
            # Link to domain
            cursor.execute("""
                INSERT INTO domain_memories (domain_id, memory_id)
                VALUES (?, ?), (?, ?), (?, ?), (?, ?), (?, ?)
            """, (
                self.domain_id, 'test_app_1',
                self.domain_id, 'test_app_2',
                self.domain_id, 'test_app_3',
                self.domain_id, 'test_app_4',
                self.domain_id, 'test_app_5'
            ))
        
        scores = knowledge_depth.analyze_depth(self.domain_id)
        
        # Verify application threshold
        self.assertGreaterEqual(
            scores['application_capability'],
            KNOWLEDGE_DEPTH_SETTINGS['thresholds']['application']
        )

    def tearDown(self):
        """Clean up test data"""
        with db_manager.get_cursor() as cursor:
            # Clean up test domains
            cursor.execute("DELETE FROM expertise_domains WHERE domain_id = ?", 
                          (self.domain_id,))
            # Clean up test memories and connections
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'")
            cursor.execute("DELETE FROM domain_memories WHERE memory_id LIKE 'test_%'")
        
        logging.info("=== Knowledge depth test cleanup complete ===")

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main() 