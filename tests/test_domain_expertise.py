"""
DOMAIN EXPERTISE TESTS
---------------------
Tests domain knowledge tracking and evolution.
"""

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import unittest
import logging
from datetime import datetime, timedelta
import json
import sqlite3
from AI_Project_Brain.domain_expertise import domain_expertise
from AI_Project_Brain.db_manager import db_manager
from config import EXPERTISE_SETTINGS
from AI_Project_Brain.memory_manager import memory_manager

class TestDomainExpertise(unittest.TestCase):
    def setUp(self):
        """Reset test data before each test"""
        self.test_ids = set()  # Track created test IDs
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM expertise_domains WHERE domain_id LIKE 'test_%'")
            cursor.execute("DELETE FROM domain_memories WHERE domain_id LIKE 'test_%'")
            cursor.execute("DELETE FROM domain_evolution WHERE domain_id LIKE 'test_%'")
            cursor.execute("COMMIT")  # Ensure transaction is committed

    def tearDown(self):
        """Clean up after each test"""
        with db_manager.get_cursor() as cursor:
            # Clean up any test data we created
            cursor.execute("DELETE FROM expertise_domains WHERE domain_id LIKE 'test_%'")
            cursor.execute("DELETE FROM domain_memories WHERE domain_id LIKE 'test_%'")
            cursor.execute("DELETE FROM domain_evolution WHERE domain_id LIKE 'test_%'")
            cursor.execute("COMMIT")
        db_manager.close_connections()  # Release any held connections

    def _track_test_id(self, domain_id):
        """Track test IDs for cleanup"""
        if domain_id:
            self.test_ids.add(domain_id)

    def test_table_creation(self):
        """Test that expertise tables are created properly"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='expertise_domains'
            """)
            self.assertIsNotNone(cursor.fetchone(), "expertise_domains table should exist")
            
            cursor.execute("PRAGMA table_info(expertise_domains)")
            columns = {row[1] for row in cursor.fetchall()}
            
            expected_columns = {
                'domain_id', 'topic_cluster', 'confidence_level',
                'first_emergence', 'last_activity', 'memory_count',
                'avg_quality', 'evidence_sources', 'depth_score',
                'breadth_score'
            }
            
            self.assertEqual(expected_columns, columns,
                           "Table should have all expected columns")

    def test_confidence_calculation(self):
        """Test domain confidence calculation"""
        test_topics = {'python', 'programming'}
        
        # Add some knowledge with test_mode=True
        domain_id = domain_expertise.update_domain_knowledge(
            content="Python is a programming language",
            topics=test_topics,
            source_type='conversation',
            quality_score=0.8,
            test_mode=True
        )
        self._track_test_id(domain_id)  # Track for cleanup
        
        # Check confidence
        confidence = domain_expertise.get_domain_confidence(test_topics)
        self.assertGreater(confidence, 0.0, "Should have non-zero confidence")
        self.assertLessEqual(confidence, 1.0, "Confidence should not exceed 1.0")

    def test_topic_clustering(self):
        """Test the topic clustering functionality"""
        # Create test memories with related topics
        test_memories = [
            ("Python is a programming language", ["python", "programming", "coding"]),
            ("Programming requires logical thinking", ["programming", "logic", "problem_solving"]),
            ("Python code needs testing", ["python", "testing", "coding"])
        ]
        
        # Add test memories
        for content, topics in test_memories:
            memory_id = memory_manager.store_memory(
                content=content,
                topics=set(topics)
            )
        
        # Test clustering
        clusters = domain_expertise._cluster_related_topics("python")
        
        # Verify clusters
        self.assertIn("programming", clusters)
        self.assertIn("coding", clusters)
        self.assertGreater(len(clusters), 0)
        self.assertLessEqual(len(clusters), EXPERTISE_SETTINGS['clustering']['max_cluster_size'])

    def test_knowledge_updates(self):
        """Test that knowledge updates properly affect metrics"""
        test_topics = {'physics'}
        test_id = None
        
        # Add initial knowledge
        test_id = domain_expertise.update_domain_knowledge(
            content="Basic physics concepts",
            topics=test_topics,
            source_type='conversation',
            quality_score=0.5,
            test_mode=True
        )
        self._track_test_id(test_id)
        
        initial_confidence = domain_expertise.get_domain_confidence(test_topics)
        
        # Add more advanced knowledge
        domain_expertise.update_domain_knowledge(
            content="Advanced quantum mechanics",
            topics=test_topics,
            source_type='document',
            quality_score=0.9,
            test_mode=True
        )
        
        updated_confidence = domain_expertise.get_domain_confidence(test_topics)
        self.assertGreater(updated_confidence, initial_confidence,
                          "Confidence should increase with more knowledge")

    def test_evidence_tracking(self):
        """Test tracking of evidence sources"""
        test_topics = {'history'}
        test_id = None
        
        # Add knowledge from different sources
        sources = ['conversation', 'document', 'analysis']
        for source in sources:
            domain_id = domain_expertise.update_domain_knowledge(
                content=f"History content from {source}",
                topics=test_topics,
                source_type=source,
                quality_score=0.7,
                test_mode=True
            )
            if test_id is None:
                test_id = domain_id
                self._track_test_id(test_id)
        
        # Check evidence sources
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT evidence_sources 
                FROM expertise_domains 
                WHERE domain_id = ?
            """, (test_id,))
            evidence = cursor.fetchone()[0]
            evidence_sources = {s.strip('"') for s in json.loads(evidence)}  # Remove quotes
            
            for source in sources:
                self.assertIn(source, evidence_sources,
                            f"Should track {source} as evidence source")

    def test_related_topic_updates(self):
        """Test that knowledge updates affect related topics"""
        # Add knowledge about Python
        domain_id = domain_expertise.update_domain_knowledge(
            content="Python is a programming language",
            topics={'python'},
            source_type='document',
            quality_score=0.8,
            test_mode=True
        )
        
        # Check confidence for related topics
        python_conf = domain_expertise.get_domain_confidence({'python'})
        programming_conf = domain_expertise.get_domain_confidence({'programming'})
        
        # Related topic should have some confidence, but less than direct topic
        self.assertGreater(programming_conf, 0.0, "Related topic should have some confidence")
        self.assertGreater(python_conf, programming_conf, "Direct topic should have higher confidence")
        
        # Add knowledge specifically about programming
        domain_expertise.update_domain_knowledge(
            content="Programming involves problem solving",
            topics={'programming'},
            source_type='document',
            quality_score=0.8,
            test_mode=True
        )
        
        # Check updated confidences
        new_python_conf = domain_expertise.get_domain_confidence({'python'})
        new_programming_conf = domain_expertise.get_domain_confidence({'programming'})
        
        # Both confidences should increase
        self.assertGreater(new_python_conf, python_conf, "Python confidence should increase")
        self.assertGreater(new_programming_conf, programming_conf, "Programming confidence should increase")

    def test_domain_creation(self):
        """Test basic domain creation and retrieval"""
        # Add debug logging
        domain_id = domain_expertise.update_domain_knowledge(
            content="Python is a programming language",
            topics={"python", "programming"},
            source_type="test",
            quality_score=0.8
        )
        
        with db_manager.get_cursor() as cursor:
            # Check memory items
            cursor.execute("SELECT COUNT(*) as count FROM memory_items")
            memory_count = cursor.fetchone()['count']
            print(f"Total memories created: {memory_count}")
            
            # Check domain memories
            cursor.execute("""
                SELECT m.* FROM memory_items m
                JOIN domain_memories dm ON m.id = dm.memory_id
                WHERE dm.domain_id = ?
            """, (domain_id,))
            memories = cursor.fetchall()
            print("Domain memories:")
            for mem in memories:
                print(f"- {mem['content']}")

    def test_domain_evolution(self):
        """Test domain knowledge evolution over time"""
        print("\n=== Starting Domain Evolution Test ===")
        
        # Initial knowledge
        print("\nStep 1: Creating initial knowledge...")
        domain_id = domain_expertise.update_domain_knowledge(
            content="Python basics",
            topics={"python"},
            source_type="test",
            quality_score=0.8
        )
        
        # Debug check after first creation
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM memory_items")
            print(f"Total memories after first creation: {cursor.fetchone()['count']}")
            cursor.execute("SELECT COUNT(*) as count FROM domain_memories WHERE domain_id = ?", (domain_id,))
            print(f"Domain memories after first creation: {cursor.fetchone()['count']}")
        
        print("\nStep 2: Adding more knowledge...")
        # Add more knowledge
        domain_expertise.update_domain_knowledge(
            content="Advanced Python concepts",
            topics={"python", "programming"},
            source_type="test",
            quality_score=0.9
        )
        
        # Debug check after second creation
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) as count FROM memory_items")
            print(f"Total memories after second creation: {cursor.fetchone()['count']}")
            cursor.execute("SELECT COUNT(*) as count FROM domain_memories WHERE domain_id = ?", (domain_id,))
            print(f"Domain memories for original domain: {cursor.fetchone()['count']}")
            cursor.execute("SELECT * FROM expertise_domains")
            domains = cursor.fetchall()
            print("\nAll domains:")
            for d in domains:
                print(f"Domain {d['domain_id']}: {d['topic_cluster']} - {d['memory_count']} memories")
                cursor.execute("SELECT memory_id FROM domain_memories WHERE domain_id = ?", (d['domain_id'],))
                memories = cursor.fetchall()
                print(f"Associated memories: {[m['memory_id'] for m in memories]}")

    def test_domain_constraints(self):
        """Test domain table constraints"""
        with db_manager.get_cursor() as cursor:
            # Test confidence constraint
            with self.assertRaises(sqlite3.IntegrityError):
                cursor.execute("""
                    INSERT INTO expertise_domains 
                    (domain_id, topic_cluster, confidence_level)
                    VALUES (?, ?, ?)
                """, ('test_invalid', '["test"]', 1.5))  # Invalid confidence
            
            # Test quality constraint
            with self.assertRaises(sqlite3.IntegrityError):
                cursor.execute("""
                    INSERT INTO expertise_domains 
                    (domain_id, topic_cluster, avg_quality)
                    VALUES (?, ?, ?)
                """, ('test_invalid', '["test"]', -0.1))  # Invalid quality

    def test_system_integration(self):
        """Test domain expertise integration with other system components"""
        from AI_Project_Brain.memory_connections import memory_connections
        from AI_Project_Brain.memory_decay import memory_decay
        
        # Create test data
        domain_id = domain_expertise.update_domain_knowledge(
            content="Python programming basics",
            topics={"python", "programming"},
            source_type="test",
            quality_score=0.8
        )
        
        # Verify memory connections integration
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM memory_connections mc
                JOIN domain_memories dm ON mc.source_id = dm.memory_id
                WHERE dm.domain_id = ?
            """, (domain_id,))
            self.assertGreater(cursor.fetchone()['count'], 0)
        
        # Verify decay protection
        protected = memory_decay._get_protected_memories()
        self.assertIn(domain_id, protected)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    unittest.main() 