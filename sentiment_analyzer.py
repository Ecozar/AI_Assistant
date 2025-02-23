"""
SENTIMENT ANALYZER
----------------
Analyzes interaction sentiment for personality state updates.
"""

import logging
from typing import Dict, Tuple
from textblob import TextBlob
import numpy as np

from config import PERSONALITY_DEFAULTS

class SentimentAnalyzer:
    """Analyzes interaction sentiment and formality"""
    
    def __init__(self):
        self.formal_indicators = {
            'please', 'thank you', 'would you', 'could you',
            'sincerely', 'regards', 'dear', 'appreciate'
        }
        
        self.learning_indicators = {
            'understand', 'learn', 'explain', 'how does',
            'what is', 'why does', 'teach', 'help me'
        }
    
    def analyze_interaction(self, text: str) -> Dict[str, float]:
        """
        Analyze interaction for sentiment, formality, and learning value.
        
        Returns:
            Dict containing sentiment, formality_level, and learning_value
        """
        # Use TextBlob for sentiment
        blob = TextBlob(text.lower())
        sentiment = blob.sentiment.polarity
        
        # Calculate formality
        words = set(text.lower().split())
        formality_score = len(words & self.formal_indicators) / len(words)
        formality_level = max(
            PERSONALITY_DEFAULTS['formality']['default_level'],
            formality_score
        )
        
        # Calculate learning value
        learning_indicators_present = len(words & self.learning_indicators)
        learning_value = min(
            1.0,
            learning_indicators_present * PERSONALITY_DEFAULTS['learning']['min_increment']
        )
        
        return {
            'sentiment': sentiment,
            'formality_level': formality_level,
            'learning_value': learning_value
        }

# Global instance
sentiment_analyzer = SentimentAnalyzer() 