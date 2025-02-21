import sys
import os
# Add the project root (one directory up) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import unittest
from AI_Project_Brain.advanced_prompt_builder import build_prompt

class TestAdvancedPromptBuilder(unittest.TestCase):
    def test_build_prompt_with_all_components(self):
        """
        Test that build_prompt correctly integrates knowledge, conversation history,
        and the current query.
        """
        knowledge_chunks = [
            "The Earth orbits the Sun once every 365.25 days.",
            "Water boils at 100°C at sea level."
        ]
        conversation_context = [
            ("Hello, how are you?", "I am fine, thank you!", "2025-02-10 18:11:09"),
            ("What's the weather like?", "It's sunny today.", "2025-02-10 18:16:23")
        ]
        query = "Why does the Earth orbit the Sun?"
        prompt = build_prompt(query, knowledge_chunks, conversation_context)
        
        # Check that the prompt contains the expected sections.
        self.assertIn("Instruction: Use the context below to provide a detailed, coherent answer.", prompt)
        self.assertIn("Knowledge:", prompt)
        self.assertIn("Conversation History:", prompt)
        self.assertIn("Current Query: " + query, prompt)
    
    def test_build_prompt_with_no_conversation(self):
        """
        Test that build_prompt works when there is no conversation history provided.
        """
        knowledge_chunks = [
            "The Earth orbits the Sun once every 365.25 days."
        ]
        conversation_context = None
        query = "Explain the orbit of the Earth."
        prompt = build_prompt(query, knowledge_chunks, conversation_context)
        
        self.assertIn("Knowledge:", prompt)
        # Ensure that the conversation history section is omitted.
        self.assertNotIn("Conversation History:", prompt)
        self.assertIn("Current Query: " + query, prompt)
    
    def test_build_prompt_with_empty_knowledge(self):
        """
        Test that build_prompt works when no knowledge chunks are provided.
        """
        knowledge_chunks = []
        conversation_context = [
            ("Hello", "Hi there!", "2025-02-10 18:11:09")
        ]
        query = "Tell me something interesting."
        prompt = build_prompt(query, knowledge_chunks, conversation_context)
        
        # Ensure the knowledge section is not present.
        self.assertNotIn("Knowledge:", prompt)
        self.assertIn("Conversation History:", prompt)
        self.assertIn("Current Query: " + query, prompt)

if __name__ == "__main__":
    unittest.main()
