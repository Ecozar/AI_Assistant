"""
MEMORY SYSTEM TESTS
------------------
Comprehensive test suite for the memory system including:
1. Core memory operations
2. Memory access tracking
3. Memory connections
4. Memory decay
5. Pattern detection
"""

import sys
import os
import unittest
import logging
from datetime import datetime, timedelta, UTC
import json
import sqlite3
from typing import List, Dict, Any
import time
import gc
from multiprocessing import Queue

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# First import DatabaseManager class
from AI_Project_Brain.db_manager import (
    DatabaseManager, 
    MEMORY_ITEMS_SCHEMA,
    ACCESS_TRACKER_SCHEMA,
    MEMORY_CONNECTIONS_SCHEMA,
    db_manager
)
from AI_Project_Brain.memory_manager import memory_manager, MemoryLayer, MemoryItem
from AI_Project_Brain.memory_access_tracker import memory_access_tracker
from config import MEMORY_SETTINGS, TEST_DB_FILE
from AI_Project_Brain.domain_expertise import domain_expertise

# Configure logging
logger = logging.getLogger(__name__)

class TestMemorySystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        cls.logger = logging.getLogger('MemoryTest')
        cls.logger.setLevel(logging.DEBUG)
        
        # Clear test data
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%'")  # Changed from memory_access_patterns
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'")
        
        cls.logger.info("Starting Memory System Tests")

        cls.logger.info("=== DATABASE INITIALIZATION SEQUENCE ===")
        
        # Initialize test database
        cls.db_manager = DatabaseManager(test_mode=True)
        cls.logger.info(f"1. Created DatabaseManager with path: {cls.db_manager.db_path}")
        
        cls.logger.info("2. Starting force cleanup...")
        cls._force_cleanup()
        cls.logger.info("3. Force cleanup complete")
        
        cls.logger.info("4. Starting database reset...")
        cls._reset_database()
        cls.logger.info("5. Database reset complete")
        
        cls.logger.info("6. Checking memory_items schema...")
        with sqlite3.connect(str(cls.db_manager.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(memory_items)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            cls.logger.info(f"Current columns: {json.dumps(columns, indent=2)}")
        
        cls.logger.info("7. Initializing memory manager...")
        memory_manager._db = cls.db_manager
        memory_manager._init_database()
        cls.logger.info("8. Memory manager initialization complete")
        
        cls.logger.info("=== INITIALIZATION COMPLETE ===")

        cls.logger.info("\n=== TEST SETUP START ===")
        
        # Verify database path
        cls.logger.info(f"Test database path: {TEST_DB_FILE}")
        if os.path.exists(TEST_DB_FILE):
            cls.logger.info(f"Existing database size: {os.path.getsize(TEST_DB_FILE)}")
        
        # Log Python runtime info
        cls.logger.info(f"Python version: {sys.version}")
        cls.logger.info(f"SQLite version: {sqlite3.sqlite_version}")
        
        # Log memory info
        import psutil
        process = psutil.Process()
        cls.logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

        cls.logger.info("\n=== TEST SETUP: INITIALIZATION START ===")
        
        # Track database state through setup
        def log_db_state(stage: str):
            with sqlite3.connect(str(TEST_DB_FILE)) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                cls.logger.info(f"\nDatabase state at {stage}:")
                cls.logger.info(f"Tables: {json.dumps([dict(zip(['name','sql'], t)) for t in tables], indent=2)}")
                
                if any(t[0] == 'memory_items' for t in tables):
                    cursor.execute("PRAGMA table_info(memory_items)")
                    columns = cursor.fetchall()
                    cls.logger.info(f"memory_items columns: {json.dumps([dict(zip(['cid','name','type','notnull','dflt_value','pk'], c)) for c in columns], indent=2)}")
        
        # Log state at key points
        log_db_state("SETUP START")
        cls._reset_database()
        log_db_state("AFTER RESET")
        memory_manager._init_database()
        log_db_state("AFTER MEMORY MANAGER INIT")

    @classmethod
    def _force_cleanup(cls):
        """Aggressive cleanup of resources"""
        cls.logger.info("Forcing cleanup of resources...")
        
        # First stop all background threads
        if hasattr(memory_manager, '_decay_thread'):
            memory_manager._stop_decay = True
            if memory_manager._decay_thread.is_alive():
                memory_manager._decay_thread.join(timeout=2.0)
                cls.logger.info("Stopped decay thread")

        # Close all database connections
        cls.db_manager.close_connections()
        
        # Force garbage collection
        gc.collect()
        time.sleep(1.0)  # Give OS time to release file handles
        
        # Now try to disable WAL mode
        try:
            with sqlite3.connect(str(cls.db_manager.db_path)) as conn:
                conn.execute("PRAGMA journal_mode=DELETE")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            cls.logger.info("Disabled WAL mode")
        except Exception as e:
            cls.logger.error(f"Error disabling WAL mode: {e}")
        
        # Delete database files
        for suffix in ['', '-shm', '-wal']:
            file_path = str(cls.db_manager.db_path) + suffix
            if os.path.exists(file_path):
                retry_count = 0
                while retry_count < 5:
                    try:
                        os.remove(file_path)
                        cls.logger.info(f"Deleted {file_path}")
                        break
                    except PermissionError:
                        retry_count += 1
                        time.sleep(1.0)
                        gc.collect()

        cls.logger.info("Resource cleanup complete")

    @classmethod
    def _reset_database(cls):
        """Reset the database to a clean state"""
        cls.logger.info("=== TEST: RESETTING DATABASE ===")
        
        # Log all active connections
        with cls.db_manager.get_cursor() as cursor:
            cursor.execute("PRAGMA database_list")
            dbs = cursor.fetchall()
            cls.logger.info(f"Active database connections: {dbs}")
            
            # Drop existing tables
            cls.logger.info("Dropping existing tables...")
            cursor.executescript("""
                DROP TABLE IF EXISTS memory_connections;
                DROP TABLE IF EXISTS memory_access_patterns;
                DROP TABLE IF EXISTS memory_items;
                DROP TABLE IF EXISTS domain_expertise;
            """)
            
            # Initialize fresh tables
            cls.logger.info("Initializing fresh database...")
            cls.db_manager._init_database()
            
            # Verify tables were created
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            cls.logger.info(f"Created tables: {tables}")
            
            # Verify memory_items schema
            cursor.execute("PRAGMA table_info(memory_items)")
            columns = {row[1]: row[2] for row in cursor.fetchall()}
            cls.logger.info(f"memory_items schema:\n{json.dumps(columns, indent=2)}")
            
            # Verify WAL mode
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
            cls.logger.info(f"Current journal mode: {mode}")
        
        # Reset connection pools to ensure fresh schema
        cls.db_manager.reset_pools()
        cls.logger.info("Reset connection pools")
        
        cls.logger.info("=== TEST: RESETTING DATABASE COMPLETE ===")

    def setUp(self):
        """Setup before each test"""
        self.test_time = datetime.now(UTC)
        self.logger.info("\n" + "="*50)
        self.logger.info(f"Starting test: {self._testMethodName}")
        self._clear_test_data()
        
    def tearDown(self):
        """Cleanup after each test"""
        self._clear_test_data()
        self.logger.info(f"Completed test: {self._testMethodName}\n")

    def _clear_test_data(self):
        """Clean up test data"""
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
                cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%'")  # Changed from memory_access_patterns
                cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'")
                cursor.execute("DELETE FROM domain_expertise WHERE topic LIKE 'test_%'")
        except Exception as e:
            self.logger.error(f"Error clearing test data: {str(e)}")
            raise

    def _verify_memory_state(self, memory_id: str) -> Dict:
        """Helper to verify memory state"""
        with self.db_manager.get_cursor() as cursor:
            cursor.execute("SELECT * FROM memory_items WHERE id = ?", (memory_id,))
            row = cursor.fetchone()
            if row:
                state = dict(row)
                self.logger.debug(f"Memory state: {json.dumps(state, indent=2)}")
                return state
            self.logger.error(f"Memory {memory_id} not found")
            return None

    # Core Memory Tests
    def test_memory_storage_and_retrieval(self):
        """Test basic memory operations"""
        # Store memory
        memory_id = memory_manager.store_memory(
            "test_content",
            {"test_topic"},
            initial_importance=0.5
        )
        self.assertIsNotNone(memory_id)
        
        # Verify storage
        memory = memory_manager.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory['content'], "test_content")
        self.assertEqual(memory['layer'], MemoryLayer.WORKING.value)

    # Access Pattern Tests
    def test_access_pattern_detection(self):
        """Test access pattern tracking and analysis"""
        logger = logging.getLogger('MemoryTest')
        logger.info("\n==================================================")
        logger.info("Starting test: test_access_pattern_detection")

        try:
            # Create test memory
            memory_id = "test_pattern_memory"
            logger.debug(f"Creating test memory with ID: {memory_id}")
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    INSERT INTO memory_items (id, content, layer)
                    VALUES (?, ?, 'working')
                """, (memory_id, "Test content"))
                logger.debug("Memory created successfully")

            # Record multiple accesses at specific hours
            logger.debug("Recording test access patterns...")
            for hour in [9, 9, 9, 14, 14]:  # Create a clear pattern
                access_time = datetime.now().replace(hour=hour)
                logger.debug(f"Recording access at hour {hour}")
                memory_access_tracker.record_access(
                    memory_id=memory_id,
                    access_type="test",
                    context="test context",
                    source="test",
                    access_time=access_time
                )

            # Verify access records
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM memory_access_log WHERE memory_id = ?", (memory_id,))
                count = cursor.fetchone()['count']
                logger.debug(f"Verified access records: {count} records found")

            # Analyze patterns
            logger.debug("Analyzing access patterns...")
            patterns = memory_access_tracker.analyze_access_patterns(memory_id)
            logger.debug(f"Returned patterns: {patterns}")

            # Test assertions
            self.assertTrue(patterns['frequency_patterns'], "Should have frequency patterns")
            self.assertTrue(patterns['time_patterns'], "Should have time patterns")

        except Exception as e:
            logger.error(f"Test failed with error: {e}", exc_info=True)
            raise
        finally:
            logger.info("Completed test: test_access_pattern_detection")

    # Memory Connection Tests
    def test_memory_connections(self):
        """Test memory connection creation and strengthening"""
        # Create two memories
        memory1_id = memory_manager.store_memory(
            "First test memory",
            {"test"},
            initial_importance=0.5
        )
        memory2_id = memory_manager.store_memory(
            "Second test memory",
            {"test"},
            initial_importance=0.5
        )
        
        # Create connection with initial strength
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_connections
                (source_id, target_id, strength, connection_type, last_reinforced)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (memory1_id, memory2_id, 0.5, "test"))
        
        # Verify connection
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT * FROM memory_connections 
                WHERE source_id = ? AND target_id = ?
            """, (memory1_id, memory2_id))
            connection = cursor.fetchone()
            self.assertIsNotNone(connection)
            self.assertEqual(connection['strength'], 0.5)

    # Memory Decay Tests
    def test_memory_decay(self):
        """Test memory importance decay"""
        # Create test memory with high importance
        memory_id = memory_manager.store_memory(
            "test_decay_content",
            initial_importance=1.0
        )
        self.assertIsNotNone(memory_id)
        
        # Apply decay
        memory_manager._apply_decay(memory_id)
        
        # Verify decay
        memory = memory_manager.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertLess(memory['importance'], 1.0)
        self.assertGreater(memory['importance'], 0.0)

    # Memory Layer Transition Tests
    def test_layer_transitions(self):
        """Test memory promotion between layers"""
        # Create memory with high importance
        memory_id = memory_manager.store_memory(
            "test_promotion_content",
            initial_importance=0.9  # High importance to trigger promotion
        )
        self.assertIsNotNone(memory_id)
        
        # Verify initial layer
        memory = memory_manager.get_memory(memory_id)
        self.assertEqual(memory['layer'], MemoryLayer.WORKING.value)
        
        # Trigger promotion check
        memory_manager._check_layer_transition(memory_id)
        
        # Verify promotion
        memory = memory_manager.get_memory(memory_id)
        self.assertEqual(memory['layer'], MemoryLayer.SHORT_TERM.value,
                        "Memory should be promoted to short-term due to high importance")

    # Domain Expertise Tests
    def test_domain_expertise_integration(self):
        """Test domain expertise tracking with memory system"""
        # Create test memory with topics
        memory_id = memory_manager.store_memory(
            "Python is a great programming language",
            {"python", "programming"},
            initial_importance=0.5
        )
        
        # Track interactions
        for _ in range(3):  # Multiple interactions to build confidence
            domain_expertise.track_domain_interaction(
                "python",  # Use consistent domain ID
                {
                    'quality_score': 0.8,
                    'source_type': 'test',
                    'topics': {"python", "programming"}
                }
            )

    # Memory Deduplication Tests
    def test_memory_deduplication(self):
        """Test detection and merging of similar memories"""
        # Create two similar memories with timestamps 2 hours apart
        content1 = "The quick brown fox jumps over the lazy dog"
        content2 = "A quick brown fox jumped over a lazy dog"
        
        # First memory created 2 hours ago
        two_hours_ago = datetime.now(UTC) - timedelta(hours=2)
        memory1_id = memory_manager.store_memory(
            content1,
            topics={"animals", "fox"},
            initial_importance=0.5
        )
        logging.debug(f"Created first memory: {memory1_id}")
        
        # Manually update the creation time
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                UPDATE memory_items 
                SET created_at = ? 
                WHERE id = ?
            """, (two_hours_ago, memory1_id))
            
            # Verify the update
            cursor.execute("SELECT created_at FROM memory_items WHERE id = ?", (memory1_id,))
            result = cursor.fetchone()
            logging.debug(f"Updated creation time: {result['created_at']}")
        
        # Create a connection to memory1
        other_memory_id = memory_manager.store_memory(
            "Some other memory",
            topics={"test"},
            initial_importance=0.5
        )
        logging.debug(f"Created connection memory: {other_memory_id}")
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_connections
                (source_id, target_id, strength, connection_type, last_reinforced)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, (other_memory_id, memory1_id, 0.5, "test"))
            
            # Verify connection
            cursor.execute("SELECT * FROM memory_connections WHERE source_id = ?", (other_memory_id,))
            connection = cursor.fetchone()
            logging.debug(f"Created connection: {dict(connection)}")
        
        # Find similar before storing
        similar = memory_manager.find_similar_memories(content2)
        logging.debug(f"Found similar memories: {similar}")
        self.assertTrue(similar, "Should find similar existing memory")
        self.assertEqual(similar[0]['id'], memory1_id)
        
        # Store second memory
        memory2_id = memory_manager.store_memory(
            content2,
            topics={"animals", "dogs"},
            initial_importance=0.7
        )
        logging.debug(f"Created second memory: {memory2_id}")
        
        # Deduplicate
        kept_id = memory_manager.deduplicate_memory(memory1_id, memory2_id, similar[0]['similarity'])
        self.assertIsNotNone(kept_id, "Deduplication failed")
        
        # Verify merged memory
        merged = memory_manager.get_memory(kept_id)
        logging.debug(f"Merged memory: {merged}")
        self.assertEqual(merged['importance'], 0.7)  # Should keep higher importance
        self.assertIn("fox", merged['topics'])  # Should have topics from both
        self.assertIn("dogs", merged['topics'])

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        cls.logger.info("Cleaning up after tests")
        
        # Stop decay thread
        if hasattr(memory_manager, '_decay_thread'):
            memory_manager._stop_decay = True
            if memory_manager._decay_thread:
                memory_manager._decay_thread.join(timeout=1.0)
        
        # Close all connections
        cls.db_manager.close_connections()
        
        # Reset memory manager to default state
        memory_manager._db = DatabaseManager()
        
        # Clean up logging
        for handler in cls.logger.handlers[:]:
            handler.close()
            cls.logger.removeHandler(handler)
        logging.shutdown()
        
        # Wait for any pending transactions
        time.sleep(0.5)
        
        cls.logger.info("Test cleanup complete")

if __name__ == '__main__':
    unittest.main() 