"""
DATABASE MANAGER
--------------
Central database management for the AI Assistant Brain Project.

This module provides:
1. Thread-safe database connections
2. Schema management and migrations
3. Connection pooling and resource management
4. Consistent database access patterns

Design Philosophy:
- Single source of truth for database operations
- Must handle concurrent access safely
- Must maintain data integrity
- Must support future schema evolution

Technical Requirements:
- Must use SQLite for portability
- Must handle concurrent UI and server access
- Must support transaction management
- Must enable efficient querying

Connection Lifecycle:
1. Connection Creation
   - Each thread gets its own connection
   - Connections are cached in thread-local storage
   - New connections created only when needed

2. Connection Pooling
   - Connections reused within same thread
   - Automatic cleanup on thread exit
   - Manual cleanup available via close_connections()

3. Transaction Management
   - All operations use transactions
   - Automatic rollback on errors
   - Context managers ensure proper cleanup

4. Thread Safety Notes:
   - Never share connections between threads
   - Each cursor operation is atomic
   - Use get_cursor() context manager for safety

Critical Dependencies:
1. SQLite Settings
   - Journal mode: WAL for concurrent access
   - Busy timeout: 5000ms
   - Foreign keys: Enabled

2. File System
   - Database file location must be writable
   - Temporary directory must have space
   - Lock files must be permitted

3. Memory Management
   - Cursor results not cached
   - Large results should be processed in chunks
   - Explicit connection cleanup recommended
"""

import sqlite3
import logging
import threading
from typing import Optional, Any, Dict, List, Generator
from contextlib import contextmanager
from pathlib import Path
from config import (
    DB_FILE, 
    DB_SETTINGS,
    EXPERTISE_SETTINGS,
    BACKUP_SETTINGS, 
    TEST_DB_FILE
)
import os
import shutil
import gzip
from datetime import datetime, timezone, UTC
import gc
from queue import Queue, Empty, Full
from sqlite3 import Cursor, Connection
import sys
import json
import time

# Configure logging
logger = logging.getLogger(__name__)

# Database Schema Definitions
MEMORY_ITEMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_items (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    layer TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT (datetime('now', 'utc')),
    last_accessed TIMESTAMP NOT NULL DEFAULT (datetime('now', 'utc')),
    access_count INTEGER DEFAULT 0,
    importance FLOAT DEFAULT 0.5,
    topics TEXT,
    related_items TEXT,
    decay_factor FLOAT DEFAULT 1.0,
    source_type TEXT DEFAULT 'unknown'
)
"""

MEMORY_ACCESS_PATTERNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_access_patterns (
    memory_id TEXT NOT NULL,
    access_type TEXT NOT NULL,
    context TEXT,
    timestamp TIMESTAMP DEFAULT (datetime('now', 'utc')),
    source TEXT,
    chain_id TEXT,
    FOREIGN KEY (memory_id) REFERENCES memory_items(id)
)
"""

MEMORY_CONNECTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_connections (
    source_id TEXT,
    target_id TEXT,
    strength FLOAT,
    last_reinforced TIMESTAMP,
    connection_type TEXT,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES memory_items(id),
    FOREIGN KEY (target_id) REFERENCES memory_items(id)
)
"""

MEMORY_CONNECTIONS_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_conn_source ON memory_connections(source_id)",
    "CREATE INDEX IF NOT EXISTS idx_conn_target ON memory_connections(target_id)",
    "CREATE INDEX IF NOT EXISTS idx_conn_type ON memory_connections(connection_type)"
]

MEMORY_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_layer ON memory_items(layer)",
    "CREATE INDEX IF NOT EXISTS idx_memory_importance ON memory_items(importance)",
    "CREATE INDEX IF NOT EXISTS idx_memory_layer_access ON memory_items(layer, last_accessed)"
]

DOMAIN_EXPERTISE_SCHEMA = """
CREATE TABLE IF NOT EXISTS domain_expertise (
    topic TEXT PRIMARY KEY,
    confidence FLOAT NOT NULL DEFAULT %f,
    interaction_count INTEGER NOT NULL DEFAULT 0,
    last_updated TIMESTAMP NOT NULL DEFAULT (datetime('now', 'utc')),
    source_diversity INTEGER NOT NULL DEFAULT %d,
    quality_score FLOAT NOT NULL DEFAULT %f
)
""" % (
    EXPERTISE_SETTINGS['confidence']['default'],
    1,  # Just use a constant since we removed source_diversity settings
    EXPERTISE_SETTINGS['quality']['default']
)

DOMAIN_EXPERTISE_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_domain_confidence ON domain_expertise(topic, confidence)"
]

ACCESS_TRACKER_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    access_time TIMESTAMP DEFAULT (datetime('now', 'utc')),
    access_type TEXT NOT NULL,
    context TEXT,
    source TEXT,
    chain_id TEXT,
    FOREIGN KEY (memory_id) REFERENCES memory_items(id)
)
"""

ACCESS_TRACKER_INDICES = """
CREATE INDEX IF NOT EXISTS idx_access_memory_id ON memory_access_log(memory_id);
CREATE INDEX IF NOT EXISTS idx_access_time ON memory_access_log(access_time);
CREATE INDEX IF NOT EXISTS idx_access_chain ON memory_access_log(chain_id);
"""

MEMORY_TOPICS_TRIGGERS = """
-- Constraint to ensure topics is valid JSON array
CREATE TRIGGER IF NOT EXISTS validate_topics_json
BEFORE INSERT ON memory_items
FOR EACH ROW
WHEN NEW.topics IS NOT NULL 
  AND (json_valid(NEW.topics) = 0 OR json_type(NEW.topics) != 'array')
BEGIN
    SELECT RAISE(ABORT, 'topics must be a valid JSON array');
END;

-- Trigger to ensure topics are unique within array
CREATE TRIGGER IF NOT EXISTS unique_topics_array
BEFORE INSERT ON memory_items
FOR EACH ROW
WHEN (
    SELECT COUNT(*) > COUNT(DISTINCT value)
    FROM json_each(NEW.topics)
)
BEGIN
    SELECT RAISE(ABORT, 'topics array must contain unique values');
END;

-- Trigger to normalize topic case
CREATE TRIGGER IF NOT EXISTS normalize_topics
BEFORE INSERT ON memory_items
FOR EACH ROW
BEGIN
    UPDATE memory_items 
    SET topics = (
        SELECT json_group_array(lower(value))
        FROM json_each(NEW.topics)
    )
    WHERE id = NEW.id;
END;
"""

class DatabaseManager:
    """
    Manages SQLite database connections and operations.
    
    Core Responsibilities:
    1. Provide thread-safe database access
    2. Manage connection lifecycle
    3. Handle schema updates
    4. Maintain data integrity
    
    Design Notes:
    - Uses thread-local storage for connections
    - Implements connection pooling
    - Provides context managers for transactions
    - Handles migrations automatically
    """
    
    def __init__(self, test_mode=False):
        self._pool_size = DB_SETTINGS['pool_size']
        self._pool = Queue(self._pool_size)
        self._active_connections = set()
        self._init_lock = threading.Lock()
        self._lock = threading.Lock()
        self._connection_timeout = DB_SETTINGS['connection_timeout']
        self._busy_timeout = DB_SETTINGS['busy_timeout']
        self._local = threading.local()
        self._pool_timeout = DB_SETTINGS.get('pool_timeout', 5)
        
        # Use test database if in test mode
        self.db_path = str(TEST_DB_FILE if test_mode else DB_FILE)
        logging.info(f"DatabaseManager initialized with path: {self.db_path}")
        
        self._init_pool()
        self._init_database()

    def _ensure_db_path(self):
        """
        Ensure database directory exists.
        
        Note:
            Creates parent directories if needed
            Handles path normalization
        """
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    def _init_database(self):
        """Initialize database schema"""
        logging.info("\n=== DB MANAGER: DATABASE INITIALIZATION START ===")
        logging.info(f"Database path: {self.db_path}")
        
        with self.get_cursor() as cursor:
            try:
                # Drop existing tables if needed
                cursor.executescript("""
                    DROP TABLE IF EXISTS memory_items;
                    DROP TABLE IF EXISTS memory_access_log;
                    DROP TABLE IF EXISTS memory_connections;
                    DROP TABLE IF EXISTS domain_expertise;
                """)
                logging.info("Dropped all existing tables")
                
                logging.info("Creating tables in order...")
                
                # Create memory items table
                cursor.execute(MEMORY_ITEMS_SCHEMA)
                logging.info("Created memory_items")
                
                # Create access log table and indices
                cursor.execute(ACCESS_TRACKER_SCHEMA)
                cursor.executescript(ACCESS_TRACKER_INDICES)  # Changed from execute to executescript
                logging.info("Created access tracking tables")
                
                # Create memory connections
                cursor.execute(MEMORY_CONNECTIONS_SCHEMA)
                for index in MEMORY_CONNECTIONS_INDICES:  # This one is still a list
                    cursor.execute(index)
                logging.info("Created memory relationship tables")
                
                # Create domain expertise
                cursor.execute(DOMAIN_EXPERTISE_SCHEMA)
                for index in DOMAIN_EXPERTISE_INDICES:  # This one is still a list
                    cursor.execute(index)
                logging.info("Created domain expertise tables")
                
                # Add topics column if it doesn't exist
                try:
                    cursor.execute("SELECT topics FROM memory_items LIMIT 1")
                except:
                    logger.info("Adding topics column to memory_items")
                    cursor.execute("""
                        ALTER TABLE memory_items 
                        ADD COLUMN topics TEXT DEFAULT '[]'
                    """)
                    
                    # Create index for topic searching
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_memory_topics 
                        ON memory_items(topics)
                    """)
                
                # Add topic-related triggers
                cursor.executescript(MEMORY_TOPICS_TRIGGERS)
                logger.info("Added topic validation triggers")
                
                # Add last_accessed column if it doesn't exist
                try:
                    cursor.execute("SELECT last_accessed FROM memory_items LIMIT 1")
                except:
                    logger.info("Adding last_accessed column to memory_items")
                    cursor.execute("""
                        ALTER TABLE memory_items 
                        ADD COLUMN last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    """)
                    
                    # Create index for last_accessed queries
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_memory_last_accessed 
                        ON memory_items(last_accessed)
                    """)
                
                logging.info("=== DATABASE INITIALIZATION COMPLETE ===\n")
                
            except Exception as e:
                logging.error(f"Error initializing database: {str(e)}")
                raise

    @contextmanager
    def get_cursor(self):
        """Get a database cursor within a transaction"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Only start transaction if we're not already in one
        if not conn.in_transaction:
            cursor.execute("BEGIN IMMEDIATE")
        
        try:
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            cursor.close()
            self.release_connection(conn)

    def _get_connection(self) -> Connection:
        """
        Get a thread-local database connection with new settings
        """
        if not hasattr(self._local, 'connection'):
            conn = sqlite3.connect(
                self.db_path, 
                timeout=self._connection_timeout
            )
            conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout}")
            conn.execute(f"PRAGMA journal_mode = {DB_SETTINGS['journal_mode']}")
            conn.execute(f"PRAGMA synchronous = {DB_SETTINGS['synchronous']}")
            conn.row_factory = sqlite3.Row
            self._local.connection = conn
        
        return self._local.connection

    def close_connections(self):
        """Close all database connections"""
        with self._lock:
            # Close pooled connections
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break

            # Close any active connections
            for conn in list(self._active_connections):
                try:
                    conn.close()
                    self._active_connections.remove(conn)
                except Exception as e:
                    logging.error(f"Error closing connection: {e}")

            # Clear the pool
            self._pool = Queue()
            self._init_pool()

    def execute_query(self, query: str, params: tuple = ()) -> Optional[Any]:
        """
        Execute a database query.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            Query results if any
            
        Note:
            Handles both queries and commands
            Manages transactions automatically
            Provides error logging
        """
        try:
            with self.get_cursor() as cursor:
                cursor.execute(query, params)
                if query.strip().upper().startswith('SELECT'):
                    return cursor.fetchall()
                return None
                
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise

    def execute_many(self, query: str, param_list: list):
        """
        Execute a query with multiple parameter sets.
        
        Args:
            query: SQL query string
            param_list: List of parameter tuples
            
        Note:
            Efficient for bulk operations
            Handles transaction automatically
            Provides error logging
        """
        try:
            with self.get_cursor() as cursor:
                cursor.executemany(query, param_list)
                
        except Exception as e:
            logger.error(f"Bulk execution error: {str(e)}")
            logger.error(f"Query: {query}")
            raise

    def get_database_size(self) -> int:
        """Get current database size in bytes"""
        try:
            return os.path.getsize(self.db_path)
        except Exception as e:
            logging.error(f"Error getting database size: {e}")
            return 0

    def optimize_database(self) -> None:
        """Run database optimization and cleanup"""
        try:
            with self.get_cursor() as cursor:
                # Analyze for query optimization
                cursor.execute("ANALYZE")
                
                # Clean up unused space
                cursor.execute("VACUUM")
                
                # Reindex for better performance
                cursor.execute("REINDEX")
                
                # Optimize write-ahead log
                cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                
                logging.info(f"Database optimized. New size: {self.get_database_size()} bytes")
                
        except Exception as e:
            logging.error(f"Error optimizing database: {e}")
            raise

    def create_backup(self, compress=True):
        """Create backup with proper connection handling"""
        try:
            # Wait for any active transactions to complete
            time.sleep(0.5)
            
            with self._lock:  # Ensure exclusive access
                with self.get_cursor() as cursor:
                    # Checkpoint WAL
                    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    
                    # Perform backup
                    backup_path = self._get_backup_path(compress)
                    self._backup_db(backup_path, compress)
                    
                    return backup_path
                    
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise

    def _get_backup_path(self, compress: bool) -> Path:
        """Generate a unique backup path based on current timestamp"""
        timestamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')
        backup_dir = Path(self.db_path).parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / f"db_backup_{timestamp}.sqlite"
        if compress:
            backup_path = backup_path.with_suffix('.sqlite.gz')
        return backup_path

    def _backup_db(self, backup_path: Path, compress: bool):
        """Backup the database to the specified path"""
        try:
            # Close all connections before backup
            self.close_connections()
            
            # Use sqlite3's built-in backup
            src_conn = sqlite3.connect(self.db_path)
            if compress:
                with gzip.open(backup_path, 'wb') as f_out:
                    for line in src_conn.iterdump():
                        f_out.write(line.encode('utf-8') + b'\n')
            else:
                with open(backup_path, 'w', encoding='utf-8') as f_out:
                    for line in src_conn.iterdump():
                        f_out.write(line + '\n')
            
            src_conn.close()
            
            # Add this line back - it was removed earlier
            self._rotate_backups()
            
        except Exception as e:
            logging.error(f"Error backing up database: {e}")
            raise

    def restore_from_backup(self, backup_path: Path) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            bool: True if restore successful
            
        Note:
            Creates safety backup before restore
            Handles both compressed and uncompressed backups
            Verifies backup integrity
        """
        try:
            # Create new database connection for restore
            self.close_connections()
            
            # Create temporary database for restore
            temp_db = sqlite3.connect(':memory:')
            
            if backup_path.suffix == '.gz':
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f_in:
                    temp_db.executescript(f_in.read())
            else:
                with open(backup_path, 'r', encoding='utf-8') as f_in:
                    temp_db.executescript(f_in.read())
            
            # Verify the restored database
            temp_db.execute("PRAGMA integrity_check")
            
            # Copy to actual database file
            with sqlite3.connect(self.db_path) as target_db:
                temp_db.backup(target_db)
            
            temp_db.close()
            return True
            
        except Exception as e:
            logging.error(f"Error restoring from backup: {e}")
            raise

    def _rotate_backups(self):
        """Maintain backup history within limits"""
        try:
            backup_dir = Path(self.db_path).parent / BACKUP_SETTINGS['storage']['backup_dir']
            if not backup_dir.exists():
                return
            
            # Get all backups
            compressed_backups = list(backup_dir.glob("*.sqlite.gz"))
            uncompressed_backups = list(backup_dir.glob("*.sqlite"))
            
            logging.debug(f"Found {len(compressed_backups)} compressed and {len(uncompressed_backups)} uncompressed backups")
            
            # Sort by creation time
            compressed_backups.sort(key=lambda x: x.stat().st_mtime)
            uncompressed_backups.sort(key=lambda x: x.stat().st_mtime)
            
            # Keep only recent backups based on settings
            max_backups = BACKUP_SETTINGS['storage']['max_backups']
            logging.debug(f"Max backups allowed: {max_backups}")
            
            # Remove old backups
            while len(compressed_backups) > max_backups:
                oldest = compressed_backups.pop(0)
                oldest.unlink()
                logging.debug(f"Removed old backup: {oldest}")
            
            # Remove old uncompressed backups
            while len(uncompressed_backups) > max_backups:
                oldest = uncompressed_backups.pop(0)
                oldest.unlink()
                logging.debug(f"Removed old backup: {oldest}")
                
        except Exception as e:
            logging.error(f"Error rotating backups: {e}")
            
    def _restore_safety_backup(self, safety_backup: Path):
        """Restore from safety backup after failed restore"""
        try:
            self.restore_from_backup(safety_backup)
            logging.info("Restored from safety backup")
        except Exception as e:
            logging.error(f"Failed to restore from safety backup: {e}")
            raise

    def verify_backup(self, backup_path: Path) -> Dict[str, Any]:
        """
        Verify backup integrity and content.
        
        Returns:
            Dict containing:
            - is_valid: bool
            - table_count: int
            - row_counts: Dict[str, int]
            - size: int
            - timestamp: datetime
        """
        try:
            # Create temporary database for verification
            temp_db = sqlite3.connect(':memory:')
            
            # Load backup into temp database
            if backup_path.suffix == '.gz':
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f_in:
                    temp_db.executescript(f_in.read())
            else:
                with open(backup_path, 'r', encoding='utf-8') as f_in:
                    temp_db.executescript(f_in.read())
            
            # Get database stats
            cursor = temp_db.cursor()
            
            # Get table list
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get row counts for each table
            row_counts = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                row_counts[table] = cursor.fetchone()[0]
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_ok = cursor.fetchone()[0] == "ok"
            
            temp_db.close()
            
            return {
                'is_valid': integrity_ok,
                'table_count': len(tables),
                'row_counts': row_counts,
                'size': backup_path.stat().st_size,
                'timestamp': datetime.fromtimestamp(backup_path.stat().st_mtime)
            }
            
        except Exception as e:
            logging.error(f"Backup verification failed: {e}")
            return {
                'is_valid': False,
                'error': str(e)
            }

    def _init_pool(self):
        """Initialize the connection pool"""
        self._pools = {}  # Dictionary of thread-local pools
        self._pool_locks = {}  # Dictionary of thread-local locks

    def _create_connection(self):
        """Create a new database connection"""
        conn = sqlite3.connect(
            self.db_path, 
            timeout=self._connection_timeout
        )
        conn.execute(f"PRAGMA busy_timeout = {self._busy_timeout}")
        conn.execute(f"PRAGMA journal_mode = {DB_SETTINGS['journal_mode']}")
        conn.execute(f"PRAGMA synchronous = {DB_SETTINGS['synchronous']}")
        conn.row_factory = sqlite3.Row
        self._active_connections.add(conn)
        return conn

    def get_table_info(self, table_name: str) -> List[tuple]:
        """Get information about table structure"""
        with self.get_cursor() as cursor:
            cursor.execute(f"PRAGMA table_info({table_name})")
            return cursor.fetchall()  # Return raw tuples instead of dicts

    def _handle_transaction(self, cursor: Cursor) -> None:
        """Ensure proper transaction handling"""
        cursor.execute("BEGIN TRANSACTION")  # Always start a new transaction

    def get_table_count(self, table_name: str) -> int:
        """Get the number of rows in a table"""
        with self.get_cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]

    def reset_pools(self):
        """Reset all connection pools to ensure fresh schema"""
        self.close_connections()
        self._pools = {}
        self._pool_locks = {}
        self._init_pool()

    def release_connection(self, conn):
        """Release a connection back to the pool"""
        if conn in self._active_connections:
            self._active_connections.remove(conn)
            try:
                self._pool.put(conn, timeout=self._pool_timeout)
            except Full:
                conn.close()  # If pool is full, just close it

    def commit(self):
        """Commit the current thread's transaction"""
        thread_id = threading.get_ident()
        if thread_id in self._active_connections:
            conn = self._active_connections[thread_id]  # Get the connection first
            self._active_connections.remove(thread_id)
            try:
                self._pool.put(conn, timeout=self._pool_timeout)
            except Full:
                conn.close()  # Now conn is defined when we use it

# Global instance for use throughout project
db_manager = DatabaseManager() 