"""
CONNECTION POOL TEST SUITE
-------------------------
Tests the database connection pooling functionality.

Test Requirements:
- Must verify pool initialization
- Must test connection reuse
- Must verify concurrent access
- Must test pool overflow handling
- Must verify connection timing
"""

import sys
import os
import unittest
import logging
import time
import threading
import concurrent.futures
from datetime import datetime
from queue import Empty

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.db_manager import DBManager, db_manager
from AI_Project_Brain.logging_config import configure_logging

# Add test constants here
TEST_POOL_SIZE = 3
TEST_ITERATIONS = 10
TEST_WORKERS = 10

class TestConnectionPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with detailed logging"""
        cls.log_file = f"test_pool_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        configure_logging(log_file=cls.log_file)
        logging.info("Starting Connection Pool tests")

    @classmethod
    def tearDownClass(cls):
        """Clean up logging handlers and connections"""
        logging.info("Cleaning up test environment")
        
        # Clean up test pools
        try:
            test_pools = [obj for obj in globals().values() 
                         if isinstance(obj, DBManager) and obj is not db_manager]
            for pool in test_pools:
                while not pool._pool.empty():
                    conn = pool._pool.get_nowait()
                    conn.close()
        except Exception as e:
            print(f"Cleanup warning: {e}")

    def test_pool_initialization(self):
        """Test pool creation and size"""
        logging.info("Testing pool initialization")
        pool = DBManager(pool_size=TEST_POOL_SIZE)
        
        self.assertEqual(pool._pool.qsize(), TEST_POOL_SIZE)
        logging.debug(f"Pool initialized with {TEST_POOL_SIZE} connections")

    def test_connection_reuse(self):
        """Test that connections are being reused"""
        logging.info("Testing connection reuse")
        used_connections = set()
        
        for i in range(TEST_ITERATIONS):
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                conn_id = id(cursor.connection)
                used_connections.add(conn_id)
                logging.debug(f"Operation {i} using connection {conn_id}")
        
        # Should have fewer unique connections than operations
        self.assertLess(len(used_connections), TEST_ITERATIONS)
        logging.debug(f"Used {len(used_connections)} unique connections")

    def test_concurrent_access(self):
        """Test pool behavior under concurrent load"""
        logging.info("Testing concurrent access")
        results = []
        
        def worker(worker_id):
            try:
                with db_manager.get_cursor() as cursor:
                    cursor.execute("SELECT 1")
                    time.sleep(0.1)  # Simulate work
                    conn_id = id(cursor.connection)
                    logging.debug(f"Worker {worker_id} using connection {conn_id}")
                    return conn_id
            except Exception as e:
                logging.error(f"Worker {worker_id} error: {e}")
                return None

        # Run more workers than pool size
        with concurrent.futures.ThreadPoolExecutor(max_workers=TEST_WORKERS) as executor:
            futures = [executor.submit(worker, i) for i in range(TEST_WORKERS)]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())

        unique_connections = len(set(filter(None, results)))
        logging.debug(f"Used {unique_connections} unique connections")
        self.assertLessEqual(unique_connections, db_manager._pool_size * 2)

    def test_pool_overflow(self):
        """Test behavior when pool is exhausted"""
        logging.info("Testing pool overflow")
        test_pool = DBManager(pool_size=1)
        active_connections = []
        
        # Try to get more connections than pool size
        for i in range(3):
            try:
                conn = test_pool._get_connection()
                active_connections.append(conn)
                logging.debug(f"Got connection {id(conn)}")
            except Exception as e:
                logging.error(f"Error getting connection {i}: {e}")

        self.assertGreater(len(active_connections), 1)
        logging.debug(f"Successfully got {len(active_connections)} connections")

        # Clean up
        for conn in active_connections:
            test_pool._return_connection(conn)

    def test_connection_timing(self):
        """Test connection timing metrics"""
        logging.info("Testing connection timing")
        
        with self.assertLogs(level='DEBUG') as logs:
            with db_manager.get_cursor() as cursor:
                cursor.execute("SELECT 1")
                time.sleep(0.1)  # Simulate work
        
        # Verify timing logs
        timing_logs = [log for log in logs.output if 'took' in log]
        self.assertTrue(timing_logs)
        logging.debug(f"Timing log captured: {timing_logs[-1]}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 