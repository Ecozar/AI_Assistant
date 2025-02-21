import unittest
from datetime import datetime
from AI_Project_Brain.memory_manager import memory_manager
from AI_Project_Brain.db_manager import db_manager

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        self.clean_test_data()
        
    def test_basic_operations(self):
        """Test basic memory operations"""
        memory_id = memory_manager.add_memory(
            "Test content",
            importance=0.5,
            topics=["test"]
        )
        
        self.assertIsNotNone(memory_id)
        
        metrics = memory_manager.get_memory_metrics(memory_id)
        self.assertEqual(metrics['access_count'], 0)
        self.assertEqual(metrics['importance']['value'], 0.5)
        
    def clean_test_data(self):
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_connections WHERE source_id LIKE 'test_%'") 