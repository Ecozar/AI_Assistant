"""
ADVANCED PROMPT BUILDER
----------------------
Constructs context-aware prompts for LLM interaction in the AI Assistant Brain Project.

This module is a critical bridge between:
1. The retrieval system (which finds relevant context)
2. The conversation history (which maintains continuity)
3. The LLM interface (which will generate responses)

Design Philosophy:
- Prompts must be structured for consistent personality
- Context must be integrated naturally into prompts
- Conversation history must provide continuity without overwhelming context
- System must support future LLM integration without major refactoring

Technical Requirements:
- Must handle multiple types of context (docs, conversations, facts)
- Must respect token limits of future LLMs
- Must support both task-specific and general conversation
- Must maintain consistent assistant personality
"""

import logging
from typing import List, Optional, Dict, Tuple, Set, Any
from datetime import datetime

from .conversation_logger import ConversationLogger
from .model_manager import model_manager
from .app import get_ui_settings
from config import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOKEN_BUDGETS,
    DEFAULT_PERSONALITY_TRAITS
)
from .personality_state import personality_tracker

logger = logging.getLogger(__name__)

class PromptBuilder:
    """
    Builds structured prompts for LLM interaction.
    
    Core Responsibilities:
    1. Combine different types of context coherently
    2. Maintain consistent assistant personality
    3. Respect token budgets for different prompt sections
    4. Support different types of interactions
    
    The prompt structure is designed to:
    - Ground the LLM with relevant context
    - Maintain conversation continuity
    - Ensure consistent personality
    - Support future LLM-specific optimizations
    """
    
    def __init__(self):
        """
        Initialize the prompt builder.
        
        Sets up:
        - Access to conversation history
        - Default personality traits
        - Token budget management
        """
        self.conversation_logger = ConversationLogger()
        
        # Use constants from config
        self.personality_traits = DEFAULT_PERSONALITY_TRAITS
        self.token_budgets = DEFAULT_TOKEN_BUDGETS

    def build_prompt(self,
                    query: str,
                    context_chunks: List[str],
                    conversation_id: Optional[str] = None,
                    task_specific: bool = False,
                    system_prompt: Optional[str] = None) -> str:
        """
        Build a complete prompt for LLM interaction.
        
        This method orchestrates prompt construction by:
        1. Starting with system message (personality)
        2. Adding retrieved context
        3. Including relevant conversation history
        4. Appending the current query
        
        Args:
            query: The user's current query
            context_chunks: Retrieved relevant context
            conversation_id: Optional ID for conversation continuity
            task_specific: Whether to use task-specific formatting
            system_prompt: Optional custom system prompt from settings
            
        Returns:
            A structured prompt string ready for LLM
            
        Note:
            Format is optimized for future LLM integration
            Currently supports general chat but extensible for specific tasks
        """
        try:
            # Use provided system prompt or fall back to default
            settings = get_ui_settings()
            final_system_prompt = system_prompt or settings.get('system_prompt') or DEFAULT_SYSTEM_PROMPT
            
            # Start with system message (personality setup)
            prompt_parts = [f"System: {final_system_prompt}"]
            
            # Add context if available
            if context_chunks:
                prompt_parts.append(self._format_context(context_chunks))
            
            # Add conversation history if available
            if conversation_id:
                history = self._get_formatted_history(conversation_id)
                if history:
                    prompt_parts.append(history)
            
            # Add current query
            prompt_parts.append(f"User: {query}\n\nAssistant:")
            
            return "\n\n".join(prompt_parts)
            
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            # Fallback to basic prompt if something goes wrong
            return f"User: {query}\n\nAssistant:"

    def _build_system_message(self, task_specific: bool = False) -> str:
        """
        Construct the system message that defines assistant personality.
        
        Args:
            task_specific: Whether to include task-specific instructions
            
        Returns:
            Formatted system message string
            
        Note:
            This is crucial for maintaining consistent personality
            Will be optimized for specific LLM requirements
        """
        # Use the default system prompt from config
        base_message = DEFAULT_SYSTEM_PROMPT
        
        if task_specific:
            base_message += "\nFocus on the specific task at hand, "
            "using provided context to inform your response."
            
        return f"System: {base_message}"

    def _format_context(self, context_chunks: List[str]) -> str:
        """
        Format retrieved context chunks for the prompt.
        
        Args:
            context_chunks: List of relevant context pieces
            
        Returns:
            Formatted context string
            
        Note:
            Format optimized for LLM comprehension
            Maintains clear separation between different chunks
        """
        formatted_chunks = [
            f"Relevant Context {i+1}:\n{chunk}"
            for i, chunk in enumerate(context_chunks)
        ]
        
        return "Context Information:\n" + "\n\n".join(formatted_chunks)

    def _get_formatted_history(self, conversation_id: str) -> str:
        """
        Format relevant conversation history for the prompt.
        
        Args:
            conversation_id: ID of the current conversation
            
        Returns:
            Formatted conversation history string
            
        Note:
            Implements smart history truncation
            Maintains conversation flow while respecting token limits
        """
        history = self.conversation_logger.get_conversation_history(conversation_id)
        
        if not history:
            return ""
            
        formatted_exchanges = []
        for exchange in history[-3:]:  # Last 3 exchanges for context
            formatted_exchanges.extend([
                f"User: {exchange['user_message']}",
                f"Assistant: {exchange['assistant_message']}"
            ])
            
        return "Previous Conversation:\n" + "\n\n".join(formatted_exchanges)

    def _build_memory_context(self, memory_metrics: Dict[str, Any]) -> str:
        """Build context string based on memory metrics"""
        if not memory_metrics:
            return ""
        
        context_parts = []
        
        # Add usage patterns
        if memory_metrics['access_patterns']['total_accesses'] > 0:
            context_parts.append(
                f"This information has been accessed "
                f"{memory_metrics['access_patterns']['total_accesses']} times"
            )
        
        # Add connection context
        if memory_metrics['connections']['count'] > 0:
            context_parts.append(
                f"It connects to {memory_metrics['connections']['count']} other memories "
                f"with {memory_metrics['connections']['strength']:.2f} average strength"
            )
        
        # Add importance context
        context_parts.append(
            f"Current importance: {memory_metrics['importance']['current_value']:.2f}, "
            f"Usage value: {memory_metrics['importance']['usage_value']:.2f}"
        )
        
        return " | ".join(context_parts)

# Global instance for use throughout project
prompt_builder = PromptBuilder()

def _build_personality_context(personality_state: Dict) -> str:
    """Build detailed personality context for response generation"""
    mood = personality_state['mood']
    energy = personality_state['energy']
    formality = personality_state['formality']
    
    # Build emotional awareness cues
    emotional_cues = []
    if energy > 0.8:
        emotional_cues.append("You are feeling energetic and enthusiastic")
    elif energy < 0.4:
        emotional_cues.append("You are feeling reserved and thoughtful")
    
    # Map internal moods to more natural language expressions
    mood_expressions = {
        "positive": "You are in a positive and optimistic state",
        "concerned": "You are feeling somewhat concerned or cautious",
        "neutral": "You are in a balanced, neutral state"
    }
    
    emotional_cues.append(mood_expressions.get(mood, "You are in a balanced, neutral state"))
    
    # Add topic-specific confidence awareness
    confident_topics = [
        topic for topic, conf in personality_state['knowledge_confidence'].items() 
        if conf > 0.7
    ]
    if confident_topics:
        emotional_cues.append(f"You feel particularly confident discussing {', '.join(confident_topics)}")
    
    context = (
        f"\nYour Current State:"
        f"\n- Emotional State: {', '.join(emotional_cues)}"
        f"\n- Energy Level: {energy:.2f} (this affects how animated your responses should be)"
        f"\n- Communication Style: {'formal and precise' if formality > 0.7 else 'casual and approachable'}"
        f"\n\nResponse Guidelines:"
        f"\n- You should naturally express your current emotional state when relevant"
        f"\n- Your responses should reflect your current energy level"
        f"\n- Express confidence in topics you know well, and be honest about uncertainty in others"
        f"\n- Stay true to your current emotional state while remaining helpful"
    )
    
    return context

def build_prompt(
    query: str,
    knowledge_chunks: List[str],
    conversation_context: List[Tuple[str, str, str]],
    system_prompt: Optional[str] = None,
    domain_confidence: float = None,
    topics: Set[str] = None
) -> str:
    # Get fresh state here instead
    personality_state = personality_tracker.get_current_state()
    
    # Use provided system prompt or fall back to default
    settings = get_ui_settings()
    final_system_prompt = system_prompt or settings.get('system_prompt') or DEFAULT_SYSTEM_PROMPT
    
    # Add detailed personality context - pass the fresh state
    personality_context = _build_personality_context(personality_state)
    
    # Build the prompt with emotional awareness
    prompt_parts = [
        f"System: {final_system_prompt}",
        personality_context,
        "\nImportant Notes:",
        "- You can express your emotional state naturally when appropriate",
        "- If asked about your feelings, respond based on your current state",
        "- Be honest about your confidence levels in different topics",
        "- Your personality evolves through interactions, so stay consistent with your state",
        "",  # Empty line for separation
    ]
    
    # Add knowledge context if relevant to emotional state
    if knowledge_chunks:
        prompt_parts.append("\nRelevant Knowledge Context:")
        for chunk in knowledge_chunks:
            prompt_parts.append(f"- {chunk}")
    
    # Add the user query
    prompt_parts.append(f"\nUser: {query}")
    prompt_parts.append("\nAssistant (keeping in mind my current state):")
    
    return "\n".join(prompt_parts)
