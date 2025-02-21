"""
BACKUP SYSTEM TESTS
------------------
Tests database backup and restore functionality.
"""

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import unittest
import tempfile
import shutil
from pathlib import Path
import sqlite3
import gzip
import json
from datetime import datetime, timedelta
import time
import logging

from AI_Project_Brain.db_manager import db_manager
from config import DB_SETTINGS, BACKUP_SETTINGS
from AI_Project_Brain.backup_manager import backup_manager

# Configure logging for tests
logging.basicConfig(level=logging.DEBUG)

class TestBackupSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment with temporary database"""
        # Create temporary directory for test database and backups
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "test.db"
        self.test_backup_dir = self.temp_dir / "backups"
        self.test_backup_dir.mkdir()
        
        # Store original paths and create test database
        self.original_db_path = db_manager.db_path
        db_manager.db_path = str(self.test_db_path)
        
        # Reset backup manager state with test mode
        backup_manager._initialized = False
        backup_manager.__init__(test_mode=True)
        
        # Close any existing connections
        db_manager.close_connections()
        
        # Initialize test database with required tables
        with sqlite3.connect(self.test_db_path) as conn:
            cursor = conn.cursor()
            # Create test table
            cursor.execute("""
                CREATE TABLE test_data (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create required tables from BACKUP_SETTINGS
            cursor.execute("""
                CREATE TABLE conversation_history (
                    id INTEGER PRIMARY KEY,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE personality_state (
                    id INTEGER PRIMARY KEY,
                    state_data TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add test data
            cursor.execute("""
                INSERT INTO test_data (content) VALUES 
                ('test1'), ('test2'), ('test3')
            """)
            
            # Add minimal required data
            cursor.execute("""
                INSERT INTO personality_state (state_data) 
                VALUES ('{"mood": "neutral"}')
            """)
            
            conn.commit()
        
        # Force a small delay after setup
        time.sleep(0.2)
    
    def tearDown(self):
        """Clean up temporary files"""
        try:
            # Close all connections
            db_manager.close_connections()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Reset path
            db_manager.db_path = self.original_db_path
            
            # Force delay before cleanup
            time.sleep(0.2)
            
            # Remove test directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logging.error(f"Cleanup error: {e}")
    
    def test_backup_creation(self):
        """Test basic backup creation"""
        # Create backup
        backup_path = db_manager.create_backup(compress=True)
        
        # Verify backup exists
        self.assertTrue(backup_path.exists())
        self.assertTrue(str(backup_path).endswith('.sqlite.gz'))
        
        # Verify backup content
        with gzip.open(backup_path, 'rb') as f:
            with sqlite3.connect(':memory:') as test_conn:
                test_conn.executescript(f.read().decode('utf-8'))
                cursor = test_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM test_data")
                count = cursor.fetchone()[0]
                self.assertEqual(count, 3)
    
    def test_backup_rotation(self):
        """Test backup rotation functionality"""
        created_backups = []
        
        # Create multiple backups with delay between
        for i in range(DB_SETTINGS['backup']['max_backups'] + 2):
            backup_path = db_manager.create_backup(compress=True)
            self.assertTrue(backup_path.exists())
            created_backups.append(backup_path)
            # Verify each backup is unique
            self.assertEqual(len(set(created_backups)), i + 1)
            time.sleep(0.1)  # Ensure different timestamps
        
        # Check number of backups
        backups = list(self.test_backup_dir.glob('*.sqlite.gz'))
        logging.debug(f"Final backup count: {len(backups)}")
        logging.debug(f"Backup files: {[b.name for b in backups]}")
        
        self.assertEqual(
            len(backups), 
            DB_SETTINGS['backup']['max_backups'],
            f"Expected {DB_SETTINGS['backup']['max_backups']} backups, got {len(backups)}"
        )
    
    def test_backup_restore(self):
        """Test backup restore functionality"""
        # Add more data and create backup
        with db_manager.get_cursor() as cursor:
            cursor.execute("INSERT INTO test_data (content) VALUES ('test4')")
        
        backup_path = db_manager.create_backup(compress=True)
        
        # Add different data
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM test_data")
            cursor.execute("INSERT INTO test_data (content) VALUES ('different')")
        
        # Restore from backup
        success = db_manager.restore_from_backup(backup_path)
        self.assertTrue(success)
        
        # Verify restored data
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT content FROM test_data ORDER BY id")
            contents = [row[0] for row in cursor.fetchall()]
            self.assertEqual(contents, ['test1', 'test2', 'test3', 'test4'])
    
    def test_failed_restore(self):
        """Test restore failure handling"""
        # Create corrupted backup
        backup_path = self.test_backup_dir / "corrupted.sqlite.gz"
        with gzip.open(backup_path, 'wb') as f:
            f.write(b"corrupted data")
        
        # Attempt restore
        with self.assertRaises(Exception):
            db_manager.restore_from_backup(backup_path)
        
        # Verify original data still intact
        with db_manager.get_cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM test_data")
            count = cursor.fetchone()[0]
            self.assertEqual(count, 3)
    
    def test_compression(self):
        """Test backup compression"""
        # Create both compressed and uncompressed backups
        compressed = db_manager.create_backup(compress=True)
        uncompressed = db_manager.create_backup(compress=False)
        
        # Verify file sizes
        self.assertLess(
            compressed.stat().st_size,
            uncompressed.stat().st_size
        )
        
        # Verify both are restorable
        self.assertTrue(db_manager.restore_from_backup(compressed))
        self.assertTrue(db_manager.restore_from_backup(uncompressed))
    
    def test_backup_scheduling(self):
        """Test backup scheduling and verification"""
        # Start backup manager
        backup_manager.start()
        
        # Force immediate backup
        backup_manager._perform_backup()
        
        # Stop backup manager before verification to prevent additional backups
        backup_manager.stop()
        
        # Verify backup occurred
        status = backup_manager.get_backup_status()
        self.assertIsNotNone(status['last_backup'])
        self.assertEqual(status['backup_count'], 1)
        
        # Verify backup is valid
        backup = status['recent_backups'][0]
        self.assertTrue(backup['verification']['is_valid'])
        
        # Verify no additional backups occurred
        final_status = backup_manager.get_backup_status()
        self.assertEqual(final_status['backup_count'], 1, "No additional backups should occur in test mode")

if __name__ == '__main__':
    unittest.main()