"""Test memory summarization functionality"""
import sys
import os
import logging
from datetime import datetime
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.memory_summarizer import memory_summarizer
from AI_Project_Brain.db_manager import db_manager

def test_summarization():
    print("Starting test_summarization...")
    
    # 1. Add some test conversations with current timestamp
    try:
        with db_manager.get_cursor() as cursor:
            # First clear any existing test data
            cursor.execute("DELETE FROM conversation_history WHERE conversation_id = 'test_convo'")
            
            test_conversations = [
                ("What is quantum physics?", "Quantum physics studies atomic behavior.", "scientific,educational"),
                ("Tell me about art history", "Art history spans many cultures...", "historical,art"),
                ("How do computers work?", "Computers process binary data...", "technical,educational")
            ]
            
            # Insert with current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for user_msg, assist_msg, tags in test_conversations:
                cursor.execute("""
                    INSERT INTO conversation_history 
                    (user_message, assistant_message, tags, conversation_id, timestamp)
                    VALUES (?, ?, ?, 'test_convo', ?)
                """, (user_msg, assist_msg, tags, current_time))
            
            print(f"Added {len(test_conversations)} test conversations")
    
        # 2. Generate summary
        print("\nGenerating summary...")
        summary = memory_summarizer.generate_period_summary('week')
        if summary is None:
            print("No summary generated - no conversations found in period")
            return
            
        print("\nGenerated Summary:")
        print(f"Summary Text: {summary['text']}")
        print(f"Topics: {summary['topics']}")
        print(f"Tags: {summary['tags']}")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")
        raise

if __name__ == "__main__":
    test_summarization() 