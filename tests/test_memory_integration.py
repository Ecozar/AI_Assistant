import sys
import os
import unittest
import logging
from datetime import datetime
import json

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.memory_access_tracker import memory_access_tracker
from AI_Project_Brain.domain_expertise import domain_expertise
from AI_Project_Brain.memory_optimizer import memory_optimizer
from AI_Project_Brain.db_manager import db_manager
from config import MEMORY_SETTINGS
from AI_Project_Brain.memory_layers import MemoryLayer
from AI_Project_Brain.memory_transition_manager import memory_transition_manager

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class TestMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.memory_id = "test_integration_memory"
        self.domain_id = "test_domain_python"
        self.clean_test_data()
        
        # Create test domain with correct schema
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO expertise_domains 
                (domain_id, confidence_level, topic_cluster)  -- Removed name column
                VALUES (?, ?, ?)
            """, (
                self.domain_id,
                0.5,  # Initial confidence
                '["python", "programming"]'  # Initial topics
            ))
    
    def test_noise_filtering_integration(self):
        """Test that noise filtering works across all components"""
        
        # Create mixed access pattern (frequent + noise)
        test_times = [
            # Significant pattern
            "2024-02-01 09:00:00",
            "2024-02-01 09:30:00",
            "2024-02-01 09:45:00",
            # Noise
            "2024-02-01 14:00:00",
        ]
        
        # Record accesses
        for timestamp in test_times:
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                "python_learning",
                "test_source",
                timestamp=dt
            )
        
        # 1. Check memory access patterns
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        hour_counts = {p['hour']: p['count'] for p in patterns['hourly_patterns']}
        
        # Should include frequent patterns
        self.assertIn('09', hour_counts)
        # Should filter out noise
        self.assertNotIn('14', hour_counts)
        
        # 2. Check domain expertise integration
        domain_expertise.update_domain_confidence(self.memory_id, self.domain_id)
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT confidence_level
                FROM expertise_domains
                WHERE domain_id = ?
            """, (self.domain_id,))
            confidence = cursor.fetchone()['confidence_level']
            
            # Confidence should be influenced by significant patterns only
            self.assertGreater(confidence, 0.5, 
                "Confidence should increase based on significant patterns")
        
        # 3. Check memory optimizer integration
        memory_optimizer.optimize_memory_storage(self.memory_id)
        
        # Verify optimization considered only significant patterns
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT optimization_metadata
                FROM memory_items
                WHERE id = ?
            """, (self.memory_id,))
            metadata = cursor.fetchone()
            
            if metadata and metadata['optimization_metadata']:
                opt_data = json.loads(metadata['optimization_metadata'])
                # Should only include significant hours in optimization
                self.assertIn('09', opt_data.get('frequent_hours', []))
                self.assertNotIn('14', opt_data.get('frequent_hours', []))
    
    def test_complex_memory_web(self):
        """Test how memories interact in a complex web of relationships
        
        This tests:
        1. Memory transitions affecting connected memories
        2. Domain expertise influence on connected memories
        3. Access patterns propagating through connections
        
        Future features this helps scaffold:
        - Memory compression (identifying clusters)
        - Pattern recognition across memory groups
        - Domain expertise reinforcement
        """
        # Create a web of related memories about programming
        memories = {
            "python_basics": {
                "content": "Basic Python programming concepts",
                "topics": ["python", "programming", "basics"],
                "connections": ["oop_concepts", "python_functions"]
            },
            "oop_concepts": {
                "content": "Object-oriented programming in Python",
                "topics": ["python", "oop", "classes"],
                "connections": ["design_patterns"]
            },
            "python_functions": {
                "content": "Python function definitions and usage",
                "topics": ["python", "functions", "basics"],
                "connections": ["design_patterns"]
            },
            "design_patterns": {
                "content": "Common design patterns in Python",
                "topics": ["python", "patterns", "architecture"],
                "connections": []
            }
        }
        
        # Set up the memory web
        self._create_memory_web(memories)
        
        # Test scenarios:
        
        # 1. Access Pattern Propagation
        test_times = [
            "2024-02-01 09:00:00",
            "2024-02-01 09:30:00",
            "2024-02-01 10:00:00"
        ]
        
        # Access python_basics frequently
        for timestamp in test_times:
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                "python_basics",
                "read",
                "learning_python",
                "test_source",
                timestamp=dt
            )
        
        # Trigger transitions
        memory_transition_manager.check_transitions("python_basics")
        
        # Verify states using helper
        for memory_id in ['python_basics', 'oop_concepts', 'python_functions']:
            self._verify_memory_state(memory_id, {
                'layer': MemoryLayer.SHORT_TERM.value,
                'min_importance': 0.5,
                'topics': ['python'],  # All should have at least python
                'connection_count': 1  # Each should have at least one connection
            })
        
        # 2. Domain Expertise Influence
        # Create Python domain with high confidence
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO expertise_domains 
                (domain_id, confidence_level, topic_cluster)
                VALUES (?, ?, ?)
            """, (
                "python_domain",
                0.9,
                json.dumps(["python", "programming", "oop"])
            ))
        
        # Verify domain influence on memory transitions
        # ... continue with more test scenarios
        
        # 3. Pattern Recognition Across Groups
        test_sequences = [
            # Python basics -> OOP sequence
            ("python_basics", "2024-02-01 09:00:00"),
            ("oop_concepts", "2024-02-01 09:05:00"),
            # Python basics -> Functions sequence
            ("python_basics", "2024-02-01 10:00:00"),
            ("python_functions", "2024-02-01 10:05:00"),
            # Complete path to design patterns
            ("python_basics", "2024-02-01 11:00:00"),
            ("oop_concepts", "2024-02-01 11:05:00"),
            ("design_patterns", "2024-02-01 11:10:00")
        ]
        
        # Record sequences
        for memory_id, timestamp in test_sequences:
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                memory_id,
                "read",
                "learning_path",
                "test_source",
                timestamp=dt,
                chain_id="learning_sequence_1"
            )
        
        # Verify patterns using helper
        self._verify_access_patterns("python_basics", {
            'hourly_counts': {
                '09': 2,  # Two accesses in 09:00 hour
                '10': 1,  # One access in 10:00 hour
                '11': 1   # One access in 11:00 hour
            },
            'contexts': ['learning_path'],
            'chain_sequences': [
                ['python_basics', 'oop_concepts'],
                ['python_basics', 'python_functions'],
                ['python_basics', 'oop_concepts', 'design_patterns']
            ]
        })
        
        # 4. Compression Hints (for future memory optimizer)
        with db_manager.get_cursor() as cursor:
            # Get all connected memories in frequently accessed chains
            cursor.execute("""
                SELECT DISTINCT m.id, m.content, m.topics,
                       COUNT(DISTINCT al.chain_id) as chain_count
                FROM memory_items m
                JOIN memory_connections mc ON m.id = mc.target_id
                JOIN memory_access_log al ON m.id = al.memory_id
                WHERE mc.source_id = ?
                GROUP BY m.id
                HAVING chain_count >= 2
            """, ("python_basics",))
            compression_candidates = cursor.fetchall()
            
            # Verify we found potential compression groups
            self.assertGreaterEqual(len(compression_candidates), 2,
                "Should identify memories frequently accessed together")
            
            # Check that related topics are grouped
            topics_by_chain = {}
            for memory in compression_candidates:
                memory_topics = json.loads(memory['topics'])
                for topic in memory_topics:
                    topics_by_chain[topic] = topics_by_chain.get(topic, 0) + 1
            
            # Verify topic clustering
            self.assertGreaterEqual(topics_by_chain.get('python', 0), 2,
                "Should find multiple memories with shared topics")
        
        # 5. Domain Expertise Reinforcement
        domain_expertise.update_domain_confidence("python_basics", "python_domain")
        
        with db_manager.get_cursor() as cursor:
            # Check if domain confidence increased from related memory access
            cursor.execute("""
                SELECT confidence_level, 
                       json_array_length(topic_cluster) as topic_count
                FROM expertise_domains
                WHERE domain_id = 'python_domain'
            """)
            domain_data = cursor.fetchone()
            
            self.assertGreater(domain_data['confidence_level'], 0.9,
                "Domain confidence should increase from related memory access")
            self.assertGreater(domain_data['topic_count'], 3,
                "Domain should learn new topics from related memories")
    
    def test_helper_methods(self):
        """Verify all helper methods work correctly"""
        # 1. Create test data
        memories = {
            "helper_test_1": {
                "content": "Test memory one",
                "topics": ["python", "testing"],
                "connections": ["helper_test_2"]
            },
            "helper_test_2": {
                "content": "Test memory two",
                "topics": ["python", "helpers"],
                "connections": []
            }
        }
        
        # Create memories using helper
        self._create_memory_web(memories)
        
        # 2. Create test access patterns
        test_times = [
            ("helper_test_1", "2024-02-01 09:00:00"),
            ("helper_test_2", "2024-02-01 09:05:00"),
            ("helper_test_1", "2024-02-01 09:30:00")
        ]
        
        # Record accesses with chain
        for memory_id, timestamp in test_times:
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                memory_id,
                "read",
                "helper_test",
                "test_source",
                timestamp=dt,
                chain_id="helper_chain_1"
            )
        
        # 3. Create test domain
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO expertise_domains 
                (domain_id, confidence_level, topic_cluster)
                VALUES (?, ?, ?)
            """, (
                "helper_domain",
                0.7,
                json.dumps(["python", "testing"])
            ))
        
        # 4. Test each helper
        
        # Test memory state verification
        self._verify_memory_state("helper_test_1", {
            'layer': MemoryLayer.WORKING.value,
            'min_importance': 0.5,
            'topics': ['python'],
            'connection_count': 1
        })
        
        # Test access pattern verification
        self._verify_access_patterns("helper_test_1", {
            'hourly_counts': {'09': 2},
            'contexts': ['helper_test'],
            'chain_sequences': [['helper_test_1', 'helper_test_2']]
        })
        
        # Test domain expertise verification
        self._verify_domain_expertise("helper_domain", {
            'min_confidence': 0.7,
            'topics': ['python', 'testing'],
            'min_memory_count': 1
        })
        
        # Test memory chain verification
        self._verify_memory_chain("helper_chain_1", {
            'sequence': ['helper_test_1', 'helper_test_2', 'helper_test_1'],
            'min_strength': 0.8,
            'shared_topics': ['python'],
            'context': 'helper_test'
        })
    
    def clean_test_data(self):
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%' OR memory_id LIKE 'helper_%' OR memory_id LIKE 'python_%'")
            cursor.execute("DELETE FROM expertise_domains WHERE domain_id LIKE 'test_%' OR domain_id LIKE 'helper_%' OR domain_id LIKE 'python_%'")
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%' OR id LIKE 'helper_%' OR id LIKE 'python_%'")
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%' OR source_id LIKE 'helper_%' OR source_id LIKE 'python_%'")
    
    def tearDown(self):
        self.clean_test_data()

    def _create_memory_web(self, memories):
        """Create a web of interconnected memories with specified relationships"""
        with db_manager.get_cursor() as cursor:
            # First create all memories
            for memory_id, data in memories.items():
                cursor.execute("""
                    INSERT INTO memory_items 
                    (id, content, layer, importance, topics)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    data["content"],
                    MemoryLayer.WORKING.value,
                    0.5,  # Default importance
                    json.dumps(data["topics"])
                ))
            
            # Then create all connections
            for memory_id, data in memories.items():
                for target_id in data["connections"]:
                    cursor.execute("""
                        INSERT INTO memory_connections
                        (source_id, target_id, connection_type, strength)
                        VALUES (?, ?, ?, ?)
                    """, (
                        memory_id,
                        target_id,
                        "related",
                        0.8  # Strong connection by default
                    ))

    def _verify_access_patterns(self, memory_id: str, expected_patterns: dict):
        """Verify access patterns match expectations
        
        Args:
            memory_id: ID of memory to check
            expected_patterns: Dict containing:
                - hourly_counts: Dict of hour -> min_count
                - contexts: List of expected contexts
                - chain_sequences: List of expected memory chains
        """
        patterns = memory_access_tracker.analyze_access_patterns(memory_id)
        
        # Check hourly patterns
        hour_counts = {p['hour']: p['count'] for p in patterns['hourly_patterns']}
        for hour, min_count in expected_patterns.get('hourly_counts', {}).items():
            self.assertGreaterEqual(
                hour_counts.get(hour, 0), 
                min_count,
                f"Hour {hour} should have at least {min_count} accesses"
            )
        
        # Check contexts
        found_contexts = set()
        for pattern in patterns['hourly_patterns']:
            found_contexts.update(pattern['contexts'])
        
        for context in expected_patterns.get('contexts', []):
            self.assertIn(
                context,
                found_contexts,
                f"Should find accesses with context '{context}'"
            )
        
        # Check chain sequences
        for sequence in expected_patterns.get('chain_sequences', []):
            self.assertTrue(
                any(chain['memories'] == sequence 
                    for chain in patterns['chain_patterns']),
                f"Should find chain sequence {sequence}"
            )

    def _verify_memory_state(self, memory_id: str, expected_state: dict):
        """Verify memory attributes match expected state
        
        Args:
            memory_id: ID of memory to check
            expected_state: Dict containing expected:
                - layer
                - min_importance
                - topics
                - connection_count
        """
        with db_manager.get_cursor() as cursor:
            # Get memory data
            cursor.execute("""
                SELECT m.*, 
                       COUNT(mc.target_id) as connection_count
                FROM memory_items m
                LEFT JOIN memory_connections mc ON m.id = mc.source_id
                WHERE m.id = ?
                GROUP BY m.id
            """, (memory_id,))
            memory = cursor.fetchone()
            
            # Verify state
            if 'layer' in expected_state:
                self.assertEqual(
                    memory['layer'],
                    expected_state['layer'],
                    f"Memory {memory_id} should be in {expected_state['layer']} layer"
                )
            
            if 'min_importance' in expected_state:
                self.assertGreaterEqual(
                    memory['importance'],
                    expected_state['min_importance'],
                    f"Memory {memory_id} importance should be at least {expected_state['min_importance']}"
                )
            
            if 'topics' in expected_state:
                memory_topics = set(json.loads(memory['topics']))
                expected_topics = set(expected_state['topics'])
                self.assertTrue(
                    expected_topics.issubset(memory_topics),
                    f"Memory {memory_id} should contain topics {expected_topics}"
                )
            
            if 'connection_count' in expected_state:
                self.assertGreaterEqual(
                    memory['connection_count'],
                    expected_state['connection_count'],
                    f"Memory {memory_id} should have at least {expected_state['connection_count']} connections"
                )

    def _verify_domain_expertise(self, domain_id: str, expected_state: dict):
        """Verify domain expertise state and influence
        
        Args:
            domain_id: ID of domain to check
            expected_state: Dict containing:
                - min_confidence: Minimum confidence level
                - topics: Expected topics in cluster
                - min_memory_count: Minimum number of memories in domain
                - min_interaction_count: Minimum number of interactions
        """
        with db_manager.get_cursor() as cursor:
            # Get domain data with memory counts
            cursor.execute("""
                SELECT d.*,
                       COUNT(DISTINCT m.id) as memory_count
                FROM expertise_domains d
                LEFT JOIN memory_items m ON json_each.value IN (
                    SELECT value FROM json_each(m.topics)
                )
                JOIN json_each(d.topic_cluster)
                WHERE d.domain_id = ?
                GROUP BY d.domain_id
            """, (domain_id,))
            domain = cursor.fetchone()
            
            if 'min_confidence' in expected_state:
                self.assertGreaterEqual(
                    domain['confidence_level'],
                    expected_state['min_confidence'],
                    f"Domain {domain_id} confidence should be at least {expected_state['min_confidence']}"
                )
            
            if 'topics' in expected_state:
                domain_topics = set(json.loads(domain['topic_cluster']))
                expected_topics = set(expected_state['topics'])
                self.assertTrue(
                    expected_topics.issubset(domain_topics),
                    f"Domain {domain_id} should contain topics {expected_topics}"
                )
            
            if 'min_memory_count' in expected_state:
                self.assertGreaterEqual(
                    domain['memory_count'],
                    expected_state['min_memory_count'],
                    f"Domain {domain_id} should have at least {expected_state['min_memory_count']} memories"
                )
            
            if 'min_interaction_count' in expected_state:
                self.assertGreaterEqual(
                    domain['interaction_count'],
                    expected_state['min_interaction_count'],
                    f"Domain {domain_id} should have at least {expected_state['min_interaction_count']} interactions"
                )

    def _verify_memory_chain(self, chain_id: str, expected_chain: dict):
        """Verify memory chain properties and relationships
        
        Args:
            chain_id: Chain identifier
            expected_chain: Dict containing:
                - sequence: List of memory IDs in expected order
                - min_strength: Minimum connection strength
                - shared_topics: Topics that should appear in all memories
                - context: Expected access context
        """
        with db_manager.get_cursor() as cursor:
            # Get chain access data
            cursor.execute("""
                SELECT memory_id, access_time, context
                FROM memory_access_log
                WHERE chain_id = ?
                ORDER BY access_time
            """, (chain_id,))
            chain_accesses = cursor.fetchall()
            
            if 'sequence' in expected_chain:
                actual_sequence = [access['memory_id'] for access in chain_accesses]
                self.assertEqual(
                    actual_sequence,
                    expected_chain['sequence'],
                    f"Chain {chain_id} should follow sequence {expected_chain['sequence']}"
                )
            
            if 'min_strength' in expected_chain:
                # Check connection strengths between sequential memories
                for i in range(len(chain_accesses) - 1):
                    cursor.execute("""
                        SELECT strength
                        FROM memory_connections
                        WHERE source_id = ? AND target_id = ?
                    """, (chain_accesses[i]['memory_id'], chain_accesses[i+1]['memory_id']))
                    connection = cursor.fetchone()
                    
                    self.assertIsNotNone(connection, "Sequential memories should be connected")
                    self.assertGreaterEqual(
                        connection['strength'],
                        expected_chain['min_strength'],
                        f"Connection strength should be at least {expected_chain['min_strength']}"
                    )
            
            if 'shared_topics' in expected_chain:
                # Get all memory topics in chain
                cursor.execute("""
                    SELECT DISTINCT value as topic
                    FROM memory_items m
                    JOIN json_each(m.topics)
                    WHERE m.id IN (
                        SELECT DISTINCT memory_id
                        FROM memory_access_log
                        WHERE chain_id = ?
                    )
                    GROUP BY value
                    HAVING COUNT(DISTINCT m.id) = ?
                """, (chain_id, len(chain_accesses)))
                
                shared_topics = {row['topic'] for row in cursor.fetchall()}
                expected_topics = set(expected_chain['shared_topics'])
                
                self.assertTrue(
                    expected_topics.issubset(shared_topics),
                    f"Chain memories should share topics {expected_topics}"
                )
            
            if 'context' in expected_chain:
                contexts = {access['context'] for access in chain_accesses}
                self.assertIn(
                    expected_chain['context'],
                    contexts,
                    f"Chain should have context {expected_chain['context']}"
                )

if __name__ == '__main__':
    unittest.main(verbosity=2) 