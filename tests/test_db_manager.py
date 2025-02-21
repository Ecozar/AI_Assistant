"""
DATABASE MANAGER TEST SUITE
-------------------------
Tests the database connection manager with comprehensive debugging.

Test Requirements:
- Must verify connection lifecycle
- Must test transaction handling
- Must verify error cases
- Must test debug helpers
- Must maintain test isolation
"""

import sys
import os
import unittest
import logging
import sqlite3
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.db_manager import (
    DatabaseManager,
    MEMORY_ITEMS_SCHEMA,
    MEMORY_ACCESS_PATTERNS_SCHEMA,
    MEMORY_CONNECTIONS_SCHEMA,
    MEMORY_INDICES,
    db_manager
)
from config import DB_FILE

class TestDBManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with detailed logging"""
        # Configure detailed logging for tests
        log_file = f"test_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        logging.info("Starting DB Manager tests")
        
        # Create test tables
        with db_manager.get_cursor() as cursor:
            cursor.executescript('''
                CREATE TABLE IF NOT EXISTS db_test_table (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    value INTEGER
                );
                CREATE TABLE IF NOT EXISTS test_foreign (
                    id INTEGER PRIMARY KEY,
                    test_id INTEGER,
                    data TEXT,
                    FOREIGN KEY (test_id) REFERENCES db_test_table(id)
                );
            ''')
    
    def setUp(self):
        """Clean up before each test"""
        logging.info(f"\n{'='*50}\nStarting test: {self._testMethodName}\n{'='*50}")
        with db_manager.get_cursor() as cursor:
            cursor.executescript('''
                DELETE FROM test_foreign;
                DELETE FROM db_test_table;
            ''')
    
    def test_connection_lifecycle(self):
        """Test connection opening, transaction, and closing"""
        logging.info("Testing connection lifecycle")
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("INSERT INTO db_test_table (name, value) VALUES (?, ?)", 
                             ("test1", 100))
                # Verify the insert worked
                cursor.execute("SELECT * FROM db_test_table")
                result = cursor.fetchone()
                self.assertEqual(result[1], "test1")
                logging.debug(f"Inserted and retrieved: {result}")
        except Exception as e:
            logging.error(f"Connection lifecycle test failed: {e}", exc_info=True)
            raise
    
    def test_transaction_rollback(self):
        """Test transaction rollback on error"""
        logging.info("=== Starting Transaction Rollback Test ===")
        
        # First insert
        logging.debug("Attempting first insert...")
        with db_manager.get_cursor() as cursor:
            cursor.execute("INSERT INTO db_test_table (name, value) VALUES (?, ?)", 
                          ("test1", 100))
            cursor.execute("SELECT COUNT(*) FROM db_test_table")
            count = cursor.fetchone()[0]
            logging.debug(f"After first insert, count = {count}")

        # Second insert that should rollback
        logging.debug("Attempting second insert (should rollback)...")
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("INSERT INTO db_test_table (name, value) VALUES (?, ?)", 
                             ("test2", 200))
                cursor.execute("SELECT COUNT(*) FROM db_test_table")
                count = cursor.fetchone()[0]
                logging.debug(f"Before rollback, count = {count}")
                raise ValueError("2 != 1")  # Force rollback
        except ValueError:
            logging.debug("Caught expected ValueError, checking rollback...")

        # Verify rollback
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM db_test_table")
            count = cursor.fetchone()[0]
            logging.debug(f"Final count after rollback = {count}")
            cursor.execute("SELECT * FROM db_test_table")
            rows = cursor.fetchall()
            logging.debug(f"Remaining records: {[dict(row) for row in rows]}")
            self.assertEqual(count, 1, "Transaction should have rolled back second insert")
    
    def test_debug_helpers(self):
        """Test the debug helper methods"""
        logging.info("=== Starting Debug Helpers Test ===")
        
        # Test table info
        logging.debug("Getting table info...")
        table_info = db_manager.get_table_info("db_test_table")
        logging.debug(f"Table info returned: {table_info}")
        
        # Check column names
        column_names = [col[1] for col in table_info]
        logging.debug(f"Column names: {column_names}")
        self.assertTrue("name" in column_names, f"Expected 'name' column in {column_names}")
        
        # Test row count
        logging.debug("Testing row count...")
        with db_manager.get_cursor() as cursor:
            cursor.execute("INSERT INTO db_test_table (name, value) VALUES (?, ?)", 
                          ("test4", 400))
            logging.debug("Inserted test record")
        
        count = db_manager.get_table_count("db_test_table")
        logging.debug(f"Table count returned: {count}")
        self.assertEqual(count, 1)
    
    def test_concurrent_transactions(self):
        """Test handling of multiple transactions"""
        logging.info("Testing concurrent transactions")
        
        # Create multiple cursors
        with db_manager.get_cursor() as cursor1:
            with db_manager.get_cursor() as cursor2:
                cursor1.execute("INSERT INTO db_test_table (name, value) VALUES (?, ?)", 
                              ("test5", 500))
                cursor2.execute("SELECT COUNT(*) FROM db_test_table")
                # Should not see the uncommitted change
                count = cursor2.fetchone()[0]
                self.assertEqual(count, 0)
                logging.debug(f"Concurrent transaction count: {count}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        logging.info("Cleaning up test environment")
        with db_manager.get_cursor() as cursor:
            cursor.executescript('''
                DROP TABLE IF EXISTS test_foreign;
                DROP TABLE IF EXISTS db_test_table;
            ''')
        logging.info("DB Manager tests completed")

if __name__ == '__main__':
    unittest.main(verbosity=2) 