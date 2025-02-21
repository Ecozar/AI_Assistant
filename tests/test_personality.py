"""
PERSONALITY SYSTEM TESTS
-----------------------
Tests the personality state tracking without LLM integration.

Tests:
1. State persistence and loading
2. Personality evolution
3. Mood transitions
4. Knowledge confidence tracking
5. Thread safety
"""

import sys
import os
import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import json
from typing import Dict, List

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from AI_Project_Brain.personality_state import PersonalityTracker, PersonalityState, personality_tracker
from AI_Project_Brain.db_manager import db_manager
from config import PERSONALITY_DEFAULTS
from AI_Project_Brain.sentiment_analyzer import sentiment_analyzer
from AI_Project_Brain.advanced_prompt_builder import build_prompt

class TestPersonalitySystem(unittest.TestCase):
    def setUp(self):
        """Reset personality state and database before each test"""
        # Clear database tables
        with db_manager.get_cursor() as cursor:
            cursor.execute("DELETE FROM emotional_contexts")
            cursor.execute("DELETE FROM emotional_patterns")
            cursor.execute("DELETE FROM personality_state")
            cursor.execute("DELETE FROM emotional_memory")
        
        # Reset personality tracker
        personality_tracker.reset_state()
    
    def test_initial_state(self):
        """Test that initial state matches config defaults"""
        state = personality_tracker._state
        self.assertEqual(state.openness, PERSONALITY_DEFAULTS['state']['openness'])
        self.assertEqual(state.mood, PERSONALITY_DEFAULTS['state']['default_mood'])
        self.assertEqual(state.energy, PERSONALITY_DEFAULTS['state']['energy'])
    
    def test_mood_transitions(self):
        """Test mood changes based on sentiment"""
        # Test positive interaction
        personality_tracker.update_state(
            {'sentiment': 0.5},  # Very positive
            ['technical']
        )
        self.assertEqual(personality_tracker._state.mood, "positive")
        
        # Test negative interaction
        personality_tracker.update_state(
            {'sentiment': -0.5},  # Very negative
            ['technical']
        )
        self.assertEqual(personality_tracker._state.mood, "concerned")
        
        # Test return to neutral
        personality_tracker.update_state(
            {'sentiment': 0.0},  # Neutral
            ['technical']
        )
        self.assertEqual(personality_tracker._state.mood, PERSONALITY_DEFAULTS['state']['default_mood'])
    
    def test_knowledge_confidence(self):
        """Test knowledge confidence evolution"""
        # Simulate multiple interactions about technical topics
        for _ in range(3):
            personality_tracker.update_state(
                {'sentiment': 0.1},
                ['technical', 'scientific']
            )
        
        state = personality_tracker.get_current_state()
        self.assertGreater(state['knowledge_confidence'].get('technical', 0), 0.5)
        self.assertGreater(state['knowledge_confidence'].get('scientific', 0), 0.5)
    
    def test_state_persistence(self):
        """Test personality state export/import"""
        # Update state
        personality_tracker.update_state(
            {'sentiment': 0.5, 'learning_value': 0.1},
            ['technical']
        )
        
        # Export state
        test_file = "test_personality_state.json"
        personality_tracker.export_state(test_file)
        
        # Create new instance and import
        new_personality = PersonalityTracker()
        new_personality.import_state(test_file)
        
        # Compare states
        original_state = personality_tracker.get_current_state()
        imported_state = new_personality.get_current_state()
        
        self.assertEqual(original_state['mood'], imported_state['mood'])
        self.assertEqual(original_state['energy'], imported_state['energy'])
        
        # Cleanup
        os.remove(test_file)
    
    def test_thread_safety(self):
        """Test concurrent personality updates"""
        def update_task(sentiment):
            personality_tracker.update_state(
                {'sentiment': sentiment},
                ['technical']
            )
            time.sleep(0.1)  # Simulate work
        
        # Run multiple updates concurrently
        with ThreadPoolExecutor(max_workers=4) as executor:
            sentiments = [0.5, -0.3, 0.2, -0.1]
            futures = [executor.submit(update_task, s) for s in sentiments]
            # Wait for all to complete
            for f in futures:
                f.result()
        
        # Verify state is consistent
        state = personality_tracker.get_current_state()
        self.assertIn(state['mood'], ['positive', 'concerned', 'neutral'])
        self.assertTrue(0 <= state['energy'] <= 1)
    
    def test_personality_evolution(self):
        """Test gradual personality trait evolution"""
        initial_openness = personality_tracker._state.openness
        
        # Simulate learning opportunities
        for _ in range(5):
            personality_tracker.update_state(
                {
                    'sentiment': 0.1,
                    'learning_value': 0.1
                },
                ['scientific', 'technical']
            )
        
        # Verify openness increased but not too dramatically
        final_openness = personality_tracker._state.openness
        self.assertGreater(final_openness, initial_openness)
        self.assertLess(final_openness - initial_openness, 0.1)  # Change should be subtle
    
    def test_formality_adaptation(self):
        """Test adaptation of formality level"""
        initial_formality = personality_tracker._state.formality
        
        # Simulate very formal interactions
        for _ in range(3):
            personality_tracker.update_state(
                {
                    'sentiment': 0.1,
                    'formality_level': 1.0
                },
                ['technical']
            )
        
        # Verify formality increased but smoothly
        self.assertGreater(personality_tracker._state.formality, initial_formality)

    def test_sentiment_analysis(self):
        """Test sentiment analysis integration"""
        test_queries = [
            ("I really appreciate your help!", 0.5),  # Positive
            ("This is terrible, I hate it.", -0.5),  # Negative
            ("Please explain this concept.", 0.1)     # Neutral but formal
        ]
        
        for query, expected_sentiment in test_queries:
            interaction_data = sentiment_analyzer.analyze_interaction(query)
            self.assertAlmostEqual(
                interaction_data['sentiment'],
                expected_sentiment,
                delta=0.3
            )

    def test_emotional_awareness(self):
        """Test that prompts include appropriate emotional awareness"""
        # Use global personality_tracker
        personality_tracker.update_state(
            {
                'sentiment': 0.8,
                'learning_value': 0.1,
                'formality_level': 0.3
            },
            ['technical', 'scientific']
        )
        
        print(f"\nAfter positive update:")
        print(f"Direct state access: {personality_tracker._state.mood}")
        current_state = personality_tracker.get_current_state()
        print(f"State from getter: {current_state}")
        
        # Verify positive state
        prompt = build_prompt(
            query="How are you feeling about technical topics?",
            knowledge_chunks=["Some technical context"],
            conversation_context=[],
        )
        self.assertIn("positive and optimistic state", prompt)
        
        # Test transition to concerned state with high formality
        negative_sentiment = PERSONALITY_DEFAULTS['sentiment']['negative_threshold'] - 0.1
        
        # Multiple updates to gradually increase formality
        for _ in range(3):  # Three updates should be enough to cross the threshold
            personality_tracker.update_state(
                {
                    'sentiment': negative_sentiment,
                    'learning_value': 0.05,
                    'formality_level': 0.9  # High formality
                },
                ['philosophy']
            )
            print(f"\nFormality after update: {personality_tracker._state.formality}")
        
        print(f"\nAfter negative update:")
        print(f"Direct state access: {personality_tracker._state.mood}")
        current_state = personality_tracker.get_current_state()
        print(f"State from getter: {current_state}")
        
        # Verify concerned state in prompt
        prompt = build_prompt(
            query="Are you comfortable discussing this?",
            knowledge_chunks=[],
            conversation_context=[],
        )
        
        print(f"\nGenerated prompt:\n{prompt}")
        self.assertIn("concerned or cautious", prompt)
        self.assertIn("formal and precise", prompt)

    def test_emotional_consistency(self):
        """Test that emotional state remains consistent across interactions"""
        # Start with positive state
        personality_tracker.update_state(
            {'sentiment': 0.8},
            ['technical']
        )
        
        initial_state = personality_tracker.get_current_state()
        
        # Generate multiple prompts
        prompts = []
        queries = [
            "How are you feeling?",
            "Tell me about technology",
            "Are you confident about this?"
        ]
        
        for query in queries:
            prompt = build_prompt(
                query=query,
                knowledge_chunks=[],
                conversation_context=[],
            )
            prompts.append(prompt)
        
        # Verify emotional consistency across prompts
        emotional_markers = []
        for prompt in prompts:
            if "positive" in prompt:
                emotional_markers.append("positive")
            elif "concerned" in prompt:
                emotional_markers.append("concerned")
            elif "neutral" in prompt:
                emotional_markers.append("neutral")
        
        # All prompts should show same emotional state
        self.assertEqual(len(set(emotional_markers)), 1, 
                        "Emotional state should be consistent across prompts")

    def test_emotional_memory(self):
        """Test emotional memory recording and retrieval"""
        # Record a series of emotional events
        events = [
            ("positive", "learning success", 0.8),
            ("concerned", "technical difficulty", 0.6),
            ("positive", "problem solved", 0.7)
        ]
        
        for mood, trigger, intensity in events:
            personality_tracker.record_emotional_event(mood, trigger, intensity)
        
        # Get recent emotions
        recent = personality_tracker.get_recent_emotions(hours=1)
        self.assertEqual(len(recent), 3)
        self.assertEqual(recent[0].mood, "positive")  # Most recent first
        
        # Check emotional summary
        summary = personality_tracker.get_emotional_summary()
        self.assertEqual(summary['dominant_mood'], "positive")
        self.assertGreater(summary['emotional_stability'], 0.5)
        self.assertIn("learning success", summary['recent_triggers'])

    def test_emotional_context_system(self):
        """Test the complete emotional context and pattern analysis system"""
        # Reset state
        personality_tracker.reset_state()
        
        # Store initial state for comparison
        initial_state = personality_tracker.get_current_state()
        
        # Simulate a series of interactions with different emotional contexts
        test_interactions = [
            {
                # Happy learning interaction
                'data': {
                    'sentiment': 0.8,
                    'learning_value': 0.9,
                    'formality_level': 0.6,
                    'trigger': 'successful_learning'
                },
                'tags': ['technical', 'educational']
            },
            {
                # Challenging technical discussion
                'data': {
                    'sentiment': -0.2,
                    'learning_value': 0.7,
                    'formality_level': 0.8,
                    'trigger': 'complex_topic'
                },
                'tags': ['technical', 'scientific']
            },
            {
                # Return to positive with understanding
                'data': {
                    'sentiment': 0.6,
                    'learning_value': 0.8,
                    'formality_level': 0.7,
                    'trigger': 'understanding_achieved'
                },
                'tags': ['technical', 'educational']
            }
        ]
        
        print("\nTesting Emotional Context System:")
        print("=================================")
        
        states_history = []  # Track state changes
        patterns_history = []  # Track pattern development
        
        # Process each interaction and verify state changes
        for i, interaction in enumerate(test_interactions):
            print(f"\nInteraction {i + 1}:")
            print(f"Input: {interaction['data']}")
            
            # Store pre-interaction state
            pre_state = personality_tracker.get_current_state()
            
            # Update emotional context
            personality_tracker.update_emotional_context(
                interaction['data'],
                interaction['tags']
            )
            
            # Get and store post-interaction state
            post_state = personality_tracker.get_current_state()
            states_history.append(post_state)
            
            # Verify basic state transitions
            self._verify_state_transition(
                pre_state, 
                post_state, 
                interaction['data'],
                f"Interaction {i+1}"
            )
            
            # Get and verify emotional patterns
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT pattern_type, pattern_data, confidence
                    FROM emotional_patterns
                    ORDER BY confidence DESC
                """)
                current_patterns = cursor.fetchall()
                patterns_history.append(current_patterns)
                
                # Verify pattern development
                self._verify_pattern_development(
                    current_patterns, 
                    i+1,
                    f"Interaction {i+1}"
                )
            
            print("\nState Verification Passed ✓")
        
        # Verify overall emotional evolution
        self._verify_emotional_evolution(states_history, patterns_history)
        print("\nEmotional Evolution Verified ✓")
        
        # Verify final emotional summary
        summary = personality_tracker.get_emotional_summary()
        self._verify_emotional_summary(summary, states_history)
        print("\nEmotional Summary Verified ✓")

    def _verify_state_transition(self, pre_state: Dict, post_state: Dict, 
                               interaction: Dict, context: str):
        """Verify individual state transitions"""
        sentiment = interaction['sentiment']
        
        # Verify mood transitions
        if sentiment > 0.5:
            self.assertEqual(post_state['mood'], 'positive', 
                            f"{context}: Should transition to positive mood")
        elif sentiment < -0.3:
            self.assertEqual(post_state['mood'], 'concerned', 
                            f"{context}: Should transition to concerned mood")
        
        # Verify energy levels
        self.assertTrue(0 <= post_state['energy'] <= 1, 
                       f"{context}: Energy should stay in valid range")
        if sentiment > 0:
            self.assertGreater(post_state['energy'], pre_state['energy'], 
                              f"{context}: Energy should increase with positive sentiment")
        
        # Verify formality adaptation
        if interaction['formality_level'] > pre_state['formality']:
            self.assertGreater(post_state['formality'], pre_state['formality'],
                              f"{context}: Formality should increase")
        
        # Verify knowledge confidence
        for tag in interaction.get('tags', []):
            self.assertIn(tag, post_state['knowledge_confidence'],
                         f"{context}: Should track confidence for tag {tag}")
            self.assertGreater(post_state['knowledge_confidence'][tag], 0,
                              f"{context}: Should have positive confidence for {tag}")

    def _verify_pattern_development(self, patterns: List, interaction_count: int, 
                                  context: str):
        """Verify pattern detection and confidence development"""
        if interaction_count >= 2:
            # Should have detected transitions after 2+ interactions
            transition_patterns = [p for p in patterns if p[0] == 'emotional_transitions']
            self.assertTrue(transition_patterns, 
                           f"{context}: Should detect emotional transitions")
            
            # Verify confidence scoring
            for _, pattern_data, confidence in patterns:
                self.assertTrue(0 <= confidence <= 1,
                               f"{context}: Confidence should be in valid range")
                
            if interaction_count >= 3:
                # Should detect cycles after 3+ interactions
                cycle_patterns = [p for p in patterns if p[0] == 'emotional_cycles']
                self.assertTrue(cycle_patterns,
                               f"{context}: Should detect emotional cycles")

    def _verify_emotional_evolution(self, states_history: List[Dict], 
                                  patterns_history: List[List]):
        """Verify overall emotional evolution"""
        # Verify personality trait development
        self.assertTrue(any(
            states_history[-1]['knowledge_confidence'][tag] > 
            states_history[0]['knowledge_confidence'].get(tag, 0)
            for tag in states_history[-1]['knowledge_confidence']
        ), "Knowledge confidence should show overall growth")
        
        # Verify pattern confidence growth
        for i in range(1, len(patterns_history)):
            current_conf = sum(p[2] for p in patterns_history[i])
            prev_conf = sum(p[2] for p in patterns_history[i-1])
            self.assertGreaterEqual(current_conf, prev_conf,
                                  "Pattern confidence should grow or maintain")

    def _verify_emotional_summary(self, summary: Dict, states_history: List[Dict]):
        """Verify emotional summary accuracy"""
        self.assertIn('dominant_mood', summary)
        self.assertIn('emotional_stability', summary)
        self.assertIn('recent_triggers', summary)
        
        # Verify stability calculation
        self.assertTrue(0 <= summary['emotional_stability'] <= 1,
                       "Emotional stability should be in valid range")
        
        # Verify trigger tracking
        self.assertTrue(len(summary['recent_triggers']) <= 3,
                       "Should track top 3 recent triggers")
        
        # Verify mood dominance calculation
        mood_counts = {}
        for state in states_history:
            mood_counts[state['mood']] = mood_counts.get(state['mood'], 0) + 1
        calculated_dominant = max(mood_counts.items(), key=lambda x: x[1])[0]
        self.assertEqual(summary['dominant_mood'], calculated_dominant,
                        "Summary should correctly identify dominant mood")

if __name__ == '__main__':
    unittest.main() 