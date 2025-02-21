import sys
import os
# Ensure the project root is in sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import logging
import sqlite3
from config import DB_FILE

"""
Module: conversation_retriever
Provides functions to retrieve conversation history from the conversation_history table.
"""

def get_conversation_context(conversation_id: str = "default", limit: int = 5):
    db_path = DB_FILE
    conn = None
    try:
        logging.debug("Connecting to database at %s for retrieval", db_path)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_message, assistant_message, timestamp
            FROM conversation_history
            WHERE conversation_id = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (conversation_id, limit)
        )
        rows = cursor.fetchall()
        logging.debug("Retrieved %d records from conversation_history", len(rows))
        return rows
    except sqlite3.Error as e:
        logging.error("SQLite error during get_conversation_context: %s", e)
        return []
    finally:
        if conn:
            conn.close()
            logging.debug("Database connection closed after retrieval")

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    context = get_conversation_context("test_session", limit=5)
    print("Retrieved conversation context:")
    for record in context:
        print(record)
