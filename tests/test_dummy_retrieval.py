import sys
import os
import unittest

# Add project root to sys.path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.dummy_retrieval import get_relevant_context

class TestDummyRetrieval(unittest.TestCase):
    def test_get_relevant_context_returns_list(self):
        """
        Test that get_relevant_context returns a list of strings.
        """
        result = get_relevant_context("any query", top_n=2)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)

if __name__ == "__main__":
    unittest.main()
