"""
PERSONALITY VISUALIZER
---------------------
Visualizes personality state changes over time.
"""

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import matplotlib.pyplot as plt
import numpy as np
from AI_Project_Brain.personality_state import personality_tracker
from AI_Project_database.auto_tagging import get_suggested_tags
from config import PERSONALITY_DEFAULTS

def simulate_and_visualize(interactions=20):
    """Simulate interactions and visualize personality changes"""
    # Track all state changes
    moods = []
    energy = []
    formality = []
    openness = []
    timestamps = range(interactions)
    
    # Reset state for clean visualization
    personality_tracker.reset_state()
    
    # Simulate a series of varied interactions
    scenarios = [
        # Formal, positive interaction
        {
            'sentiment': 0.8,
            'learning_value': 0.1,
            'formality_level': 0.9
        },
        # Informal, concerned interaction
        {
            'sentiment': -0.4,
            'learning_value': 0.05,
            'formality_level': 0.3
        },
        # Neutral, moderately formal interaction
        {
            'sentiment': 0.1,
            'learning_value': 0.1,
            'formality_level': 0.6
        }
    ]
    
    # Simulate interactions
    for i in range(interactions):
        # Cycle through scenarios
        scenario = scenarios[i % len(scenarios)]
        
        # Update state
        personality_tracker.update_state(
            scenario,
            ['technical', 'scientific'] if i % 2 == 0 else ['philosophy']
        )
        
        # Record state
        state = personality_tracker._state
        moods.append(1 if state.mood == "positive" else -1 if state.mood == "concerned" else 0)
        energy.append(state.energy)
        formality.append(state.formality)
        openness.append(state.openness)
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Core Traits
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, energy, label='Energy', marker='o')
    plt.plot(timestamps, formality, label='Formality', marker='s')
    plt.plot(timestamps, openness, label='Openness', marker='^')
    plt.axhline(y=PERSONALITY_DEFAULTS['formality']['formal_threshold'], 
                color='r', linestyle='--', label='Formal Threshold')
    plt.legend()
    plt.title('Personality Trait Evolution')
    plt.grid(True)
    plt.ylabel('Trait Value')
    
    # Plot 2: Mood and State
    plt.subplot(2, 1, 2)
    plt.plot(timestamps, moods, label='Mood', marker='o')
    plt.fill_between(timestamps, moods, alpha=0.2)
    plt.yticks([-1, 0, 1], ['Concerned', 'Neutral', 'Positive'])
    plt.legend()
    plt.title('Emotional State Transitions')
    plt.grid(True)
    plt.xlabel('Interaction Number')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_and_visualize() 