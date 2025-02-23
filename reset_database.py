# Add this to a new file called reset_database.py in your project root
import os
import time
import logging
import sqlite3
from AI_Project_Brain.app import init_db, upgrade_tags_table
from config import DB_FILE
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.memory_connections import memory_connections

def reset_database():
    """Reset the entire database with proper connection handling"""
    logging.info("Starting database reset...")
    
    # Close all connections
    db_manager.close_connections()
    db_manager.reset_pools()
    
    # Drop and recreate all tables
    with db_manager.get_cursor() as cursor:
        cursor.execute("PRAGMA writable_schema = 1")
        cursor.execute("DELETE FROM sqlite_master WHERE type IN ('table', 'index', 'trigger')")
        cursor.execute("PRAGMA writable_schema = 0")
        cursor.execute("COMMIT")  # Commit before VACUUM
        
    # VACUUM needs its own connection
    with sqlite3.connect(str(DB_FILE)) as conn:
        conn.execute("VACUUM")
    
    # Create new database with correct schema
    logging.info("Initializing new database...")
    init_db()  # This will create all tables
    upgrade_tags_table()
    logging.info("Database reset complete!")

def reset_memory_connections():
    """Reset just the memory connections tables"""
    with db_manager.get_cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS memory_connections")
        cursor.execute("DROP TABLE IF EXISTS connection_metrics")
        # Reinitialize the tables
        memory_connections._init_database()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    reset_database()