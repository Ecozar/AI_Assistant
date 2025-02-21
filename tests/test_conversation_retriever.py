import sys
import os
import unittest
import sqlite3
import logging

# Ensure the project root is in sys.path so that the AI_Project_Brain package can be found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.conversation_retriever import get_conversation_context

# Define the production database path and the test database path.
original_db_path = os.path.abspath(os.path.join(project_root, "AI_Project_Brain", "files.db"))
TEST_DB_PATH = os.path.join(project_root, "AI_Project_Brain", "test_files.db")

def setup_test_database():
    """Creates the conversation_history table in the test database."""
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT,
            user_message TEXT,
            assistant_message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

def clear_test_database():
    """Clears all records from the conversation_history table in the test database."""
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversation_history;")
    conn.commit()
    conn.close()

def insert_test_interaction(conversation_id, user_message, assistant_message):
    """Inserts a test record into the conversation_history table in the test database."""
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversation_history (conversation_id, user_message, assistant_message)
        VALUES (?, ?, ?)
    """, (conversation_id, user_message, assistant_message))
    conn.commit()
    conn.close()

class TestConversationRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the test database once before all tests.
        setup_test_database()

    def setUp(self):
        # Clear the test database and insert some test interactions.
        clear_test_database()
        insert_test_interaction("test_session", "Test message 1", "Response 1")
        insert_test_interaction("test_session", "Test message 2", "Response 2")

    def test_get_conversation_context(self):
        """
        Test that get_conversation_context retrieves the correct number of records
        and that the records are in chronological order.
        """
        # Monkey-patch sqlite3.connect so that when the production database is requested,
        # we instead return a connection to the test database.
        original_connect = sqlite3.connect
        sqlite3.connect = lambda path: original_connect(TEST_DB_PATH) if os.path.abspath(path) == os.path.abspath(original_db_path) else original_connect(path)
        
        try:
            context = get_conversation_context("test_session", limit=5)
            # Expect 2 records in chronological order.
            self.assertEqual(len(context), 2)
            # Verify the first record.
            self.assertEqual(context[0][0], "Test message 1")  # user_message
            self.assertEqual(context[0][1], "Response 1")
            # Verify the second record.
            self.assertEqual(context[1][0], "Test message 2")
            self.assertEqual(context[1][1], "Response 2")
        finally:
            # Restore the original sqlite3.connect function.
            sqlite3.connect = original_connect

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    unittest.main()
