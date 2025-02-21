import sys
import os
import unittest
import logging
from datetime import datetime, timedelta
import json
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.memory_access_tracker import memory_access_tracker
from AI_Project_Brain.db_manager import db_manager
from config import MEMORY_SETTINGS

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class TestMemoryAccessTracker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        logging.info("Setting up test environment")
        
    def setUp(self):
        """Set up before each test"""
        logging.info("Starting new test")
        self.memory_id = "test_memory"
        self.clean_test_data()
        
    def test_record_access(self):
        """Test basic access recording"""
        memory_access_tracker.record_access(
            self.memory_id,
            "read",
            "test context",
            "test source"
        )
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT COUNT(*) as count 
                FROM memory_access_log
                WHERE memory_id = ?
            """, (self.memory_id,))
            count = cursor.fetchone()['count']
            self.assertEqual(count, 1)
            
    def test_pattern_analysis(self):
        """Test access pattern analysis"""
        # Create test pattern
        base_time = datetime.now().replace(hour=10, minute=0)
        for _ in range(5):
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                "test context",
                "test source",
                timestamp=base_time
            )
            
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        self.assertTrue(patterns['hourly_patterns'])
        
    def test_enhanced_access_patterns(self):
        """Test enhanced access pattern analysis with time-based patterns"""
        logging.info("=== Starting Enhanced Access Patterns Test ===")
        
        # Test data with both noise and significant patterns
        test_times = [
            # Significant 9 AM pattern (4 occurrences)
            "2024-02-01 09:00:00",  # Thursday 9 AM
            "2024-02-01 09:30:00",  # Thursday 9 AM
            "2024-02-02 09:15:00",  # Friday 9 AM
            "2024-02-02 09:45:00",  # Friday 9 AM
            
            # Noise patterns (single occurrences)
            "2024-02-01 14:00:00",  # Thursday 2 PM
            "2024-02-01 15:00:00",  # Thursday 3 PM
            "2024-02-01 16:00:00",  # Thursday 4 PM
        ]
        
        # Record test accesses
        for i, timestamp in enumerate(test_times):
            dt = datetime.fromisoformat(timestamp)
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read" if i % 2 == 0 else "reference",
                f"context_{i//2}",
                "test_source",
                timestamp=dt
            )
        
        # Get patterns
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        hour_counts = {p['hour']: p['count'] for p in patterns['hourly_patterns']}
        
        # Verify noise filtering
        min_occurrences = MEMORY_SETTINGS['access_tracking']['patterns']['min_occurrences']
        logging.info(f"Testing with min_occurrences = {min_occurrences}")
        
        # Check that significant patterns are included
        self.assertIn('09', hour_counts, "Should include hour with multiple occurrences")
        self.assertGreaterEqual(hour_counts['09'], 4)
        
        # Check that noise is filtered out
        noise_hours = ['14', '15', '16']
        for hour in noise_hours:
            self.assertNotIn(hour, hour_counts, 
                f"Should filter out hour {hour} with only one occurrence")

    def clean_test_data(self):
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%'")

    def tearDown(self):
        """Clean up after each test"""
        logging.info("Cleaning up after test")
        self.clean_test_data()

class TestAdvancedPatternRecognition(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.memory_id = "test_pattern_memory"
        self.clean_test_data()
        
        # Create test memory
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO memory_items 
                (id, content, layer, importance, created_at)
                VALUES (?, 'Test memory content', 'working', 0.5, CURRENT_TIMESTAMP)
            """, (self.memory_id,))

    def test_sequence_patterns(self):
        """Test enhanced sequence pattern detection with variable lengths and time analysis"""
        # Test Case 1: Basic repeating sequence with good timing
        sequence_times = [
            # First sequence block - tight timing
            datetime(2024, 2, 1, 9, 0),   # 9:00 AM
            datetime(2024, 2, 1, 9, 15),  # 9:15 AM
            datetime(2024, 2, 1, 9, 30),  # 9:30 AM
            
            # Second sequence block - tight timing
            datetime(2024, 2, 1, 10, 0),  # 10:00 AM
            datetime(2024, 2, 1, 10, 15), # 10:15 AM
            datetime(2024, 2, 1, 10, 30), # 10:30 AM
            
            # Third sequence block - wider gaps but within limit
            datetime(2024, 2, 1, 14, 0),  # 2:00 PM
            datetime(2024, 2, 1, 14, 45), # 2:45 PM
            datetime(2024, 2, 1, 15, 30)  # 3:30 PM
        ]
        
        sequence = [
            ("read", "learning"),
            ("reference", "study"),
            ("update", "practice")
        ] * 3  # Repeat sequence three times
        
        for (access_type, context), timestamp in zip(sequence, sequence_times):
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                access_type,
                context,
                "test_source",
                timestamp
            )
        
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        
        # Verify enhanced sequence detection
        self.assertTrue(patterns['sequence_patterns'], "Should detect sequence patterns")
        
        # Get the highest confidence pattern
        main_pattern = max(patterns['sequence_patterns'], key=lambda x: x['confidence'])
        
        # Basic pattern verification
        self.assertEqual(main_pattern['count'], 3, "Should find 3 occurrences of sequence")
        self.assertGreater(main_pattern['confidence'], 0.0, "Should have positive confidence")
        
        # Verify time gap analysis
        self.assertTrue(hasattr(main_pattern, 'avg_time_gap'), "Should include time gap analysis")
        
        # Verify complexity factor
        self.assertTrue(hasattr(main_pattern, 'complexity'), "Should include complexity factor")
        self.assertGreater(main_pattern['complexity'], 0.0, "Should have positive complexity")
        
        # Test Case 2: Variable length pattern detection
        # Add a shorter but frequent pattern
        short_sequence_times = [
            datetime(2024, 2, 1, 16, 0),  # 4:00 PM
            datetime(2024, 2, 1, 16, 15), # 4:15 PM
            datetime(2024, 2, 1, 17, 0),  # 5:00 PM
            datetime(2024, 2, 1, 17, 15), # 5:15 PM
        ]
        
        short_sequence = [
            ("read", "learning"),
            ("reference", "study")
        ] * 2  # Repeat shorter sequence twice
        
        for (access_type, context), timestamp in zip(short_sequence, short_sequence_times):
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                access_type,
                context,
                "test_source",
                timestamp
            )
        
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        
        # Verify multiple pattern lengths
        sequence_lengths = set(len(p['sequence']) for p in patterns['sequence_patterns'])
        self.assertGreater(len(sequence_lengths), 1, "Should detect patterns of different lengths")
        
        # Verify no duplicate subsequences
        pattern_sequences = [p['sequence'] for p in patterns['sequence_patterns']]
        for i, seq1 in enumerate(pattern_sequences):
            for seq2 in pattern_sequences[i+1:]:
                self.assertFalse(
                    memory_access_tracker._is_subsequence(seq1, seq2) or
                    memory_access_tracker._is_subsequence(seq2, seq1),
                    "Should not include duplicate subsequences"
                )

    def test_periodic_patterns(self):
        """Test detection of periodic access patterns"""
        # Create daily access pattern
        daily_times = [
            datetime(2024, 2, 1, 9, 0),  # Day 1
            datetime(2024, 2, 2, 9, 0),  # Day 2
            datetime(2024, 2, 3, 9, 0)   # Day 3
        ]
        
        for timestamp in daily_times:
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                "daily_review",
                "test_source",
                timestamp
            )
        
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        
        # Verify periodic pattern detection
        self.assertTrue(patterns['periodic_patterns'], "Should detect periodic patterns")
        periodic_pattern = patterns['periodic_patterns'][0]
        self.assertEqual(periodic_pattern['type'], 'daily', "Should identify daily pattern")
        self.assertEqual(periodic_pattern['count'], 2, "Should find 2 daily intervals")
        self.assertGreater(periodic_pattern['confidence'], 0.0, "Should have positive confidence")

    def test_contextual_patterns(self):
        """Test detection of context-based patterns"""
        # Create multiple accesses in different contexts
        contexts = [
            ("learning", datetime(2024, 2, 1, 9, 0)),
            ("learning", datetime(2024, 2, 1, 10, 0)),
            ("practice", datetime(2024, 2, 1, 11, 0)),
            ("learning", datetime(2024, 2, 1, 14, 0)),
            ("review", datetime(2024, 2, 1, 15, 0))
        ]
        
        for context, timestamp in contexts:
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                context,
                "test_source",
                timestamp
            )
        
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        
        # Verify context pattern detection
        self.assertTrue(patterns['contextual_patterns'], "Should detect contextual patterns")
        learning_pattern = next(p for p in patterns['contextual_patterns'] 
                              if p['context'] == 'learning')
        self.assertEqual(learning_pattern['count'], 3, "Should count 3 learning context accesses")
        self.assertGreater(learning_pattern['confidence'], 0.0, "Should have positive confidence")

    def test_pattern_persistence(self):
        """Test that patterns are properly stored and updated"""
        # Add detailed logging for debugging
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("=== Starting pattern persistence test ===")
        
        # Create multiple test accesses at 9 AM
        base_timestamp = datetime(2024, 2, 1, 9, 0)
        for i in range(3):  # Create 3 accesses to exceed min_occurrences threshold
            timestamp = base_timestamp + timedelta(days=i)
            logging.debug(f"Creating test access at {timestamp}")
            
            memory_access_tracker._test_mode_record_access(
                self.memory_id,
                "read",
                "test_context",
                "test_source",
                timestamp
            )
        
        # Analyze patterns
        logging.debug("Analyzing patterns...")
        patterns = memory_access_tracker.analyze_access_patterns(self.memory_id)
        logging.debug(f"Patterns returned: {patterns}")
        
        # Verify pattern storage
        with db_manager.get_cursor() as cursor:
            logging.debug("Checking database for stored patterns...")
            cursor.execute("""
                SELECT COUNT(*) as pattern_count, GROUP_CONCAT(pattern_type) as types
                FROM memory_access_patterns
                WHERE memory_id = ?
            """, (self.memory_id,))
            result = cursor.fetchone()
            count = result['pattern_count']
            types = result['types']
            logging.debug(f"Found {count} patterns of types: {types}")
            self.assertGreater(count, 0, "Should store patterns in database")

    def clean_test_data(self):
        """Clean up test data"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM memory_items WHERE id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_access_log WHERE memory_id LIKE 'test_%'")
            cursor.execute("DELETE FROM memory_access_patterns WHERE memory_id LIKE 'test_%'")

    def tearDown(self):
        """Clean up after each test"""
        self.clean_test_data()

if __name__ == '__main__':
    logging.info("Starting memory access tracker tests")
    unittest.main(verbosity=2) 