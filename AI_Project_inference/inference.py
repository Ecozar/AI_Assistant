import sys
import os
import json
import logging

# Ensure the project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.advanced_prompt_builder import build_prompt
from AI_Project_Brain.app import get_ui_settings
from AI_Project_Brain.retrieval_pipeline import get_unified_context
from AI_Project_Brain.retrieval_pipeline import get_conversation_context
from AI_Project_Brain.personality_state import personality_tracker
from AI_Project_Brain.auto_tagger import get_suggested_tags
from config import PERSONALITY_DEFAULTS
from AI_Project_Brain.sentiment_analyzer import sentiment_analyzer
from AI_Project_Brain.domain_expertise import domain_expertise
from AI_Project_Brain.text_utils import generate_embedding

"""
INFERENCE MODULE
----------------
This module generates responses (currently dummy responses) based on the assembled prompt.
Key points:
- It relies on the advanced prompt builder to incorporate the system prompt, knowledge chunks, 
  and conversation history.
- The assembled prompt must always reflect the latest system prompt from the central config.
- This module is designed to be LLM-agnostic; once integrated with a local LLM, ensure that 
  any modifications continue to adhere to SSOT principles.
- Any debugging or performance improvements should prioritize consistency in personality 
  and configuration.
"""

def generate_response(query: str, conversation_id: str = None) -> str:
    """
    Generate response using unified context retrieval
    """
    try:
        # Get settings first
        settings = get_ui_settings()
        
        # Get query topics
        query_topics = get_suggested_tags(query)  # Already returns a list
        
        # Check domain confidence
        domain_confidence = domain_expertise.get_domain_confidence(query_topics)
        
        # Get both knowledge and conversation context
        knowledge_chunks, conversation_context = get_unified_context(
            query,
            conversation_id
        )
        
        # Get tags for interaction tracking
        conversation_tags = get_suggested_tags(query)
        
        # Analyze interaction sentiment and style
        interaction_data = sentiment_analyzer.analyze_interaction(query)
        
        # Build prompt with confidence awareness
        prompt = build_prompt(
            query, 
            knowledge_chunks, 
            conversation_context,
            system_prompt=settings.get('system_prompt'),
            domain_confidence=domain_confidence,  # Add confidence to prompt
            topics=query_topics
        )
        
        # TODO: Replace with actual LLM response
        response = f"Dummy response for prompt: {prompt}"
        
        # Update personality state based on interaction
        personality_tracker.update_state(interaction_data, conversation_tags)
        
        # Update domain expertise
        if knowledge_chunks:
            domain_expertise.update_domain_knowledge(
                content=response,
                topics=query_topics,
                source_type='conversation',
                quality_score=0.8  # TODO: Implement quality scoring
            )
        
        return response

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Test the function
    sample_queries = [
        "Tell me something about artificial intelligence.",
        "What is machine learning?",
        "How does natural language processing work?"
    ]
    
    print("Testing inference.py with sample queries:\n")
    for query in sample_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        print(generate_response(query))
        print("-" * 50)
