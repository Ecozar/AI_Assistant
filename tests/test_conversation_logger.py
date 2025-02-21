import sys
import os
import unittest
import sqlite3

# Ensure the project root is in sys.path so the AI_Project_Brain package can be found.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.conversation_logger import log_interaction

# Define the path for the production database and the test database.
original_db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AI_Project_Brain", "files.db"))
TEST_DB_PATH = os.path.join(project_root, "AI_Project_Brain", "test_files.db")

def setup_test_database():
    """Creates a temporary test database with the conversation_history table."""
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
    """Deletes all records from the conversation_history table in the test database."""
    conn = sqlite3.connect(TEST_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM conversation_history;")
    conn.commit()
    conn.close()

class TestConversationLogger(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the test database once before all tests.
        setup_test_database()

    def setUp(self):
        # Clear database before each test.
        clear_test_database()

    def test_log_interaction_inserts_record(self):
        # Monkey-patch sqlite3.connect to redirect connections to our test database.
        original_connect = sqlite3.connect
        
        # Use absolute paths for comparison.
        sqlite3.connect = lambda path: original_connect(TEST_DB_PATH) if os.path.abspath(path) == os.path.abspath(original_db_path) else original_connect(path)

        try:
            # Log a test interaction.
            log_interaction("test_session", "Test user message", "Test assistant response")
            # Verify that the record was inserted.
            conn = sqlite3.connect(TEST_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT conversation_id, user_message, assistant_message FROM conversation_history;")
            rows = cursor.fetchall()
            conn.close()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "test_session")
            self.assertEqual(rows[0][1], "Test user message")
            self.assertEqual(rows[0][2], "Test assistant response")
        finally:
            # Restore the original sqlite3.connect function.
            sqlite3.connect = original_connect

if __name__ == "__main__":
    unittest.main()
