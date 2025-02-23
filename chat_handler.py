"""
CHAT HANDLER
-----------
Handles incoming chat messages with consistent auto-tagging through memory_manager.
"""

import logging
from typing import Optional, Dict

from AI_Project_Brain.memory_manager import memory_manager
from AI_Project_Brain.db_manager import db_manager

logger = logging.getLogger(__name__)

def handle_chat_message(message: str, ui_instance) -> str:
    """Handle incoming chat messages using centralized memory storage"""
    try:
        # Store message using memory_manager's auto-tagging
        memory_id = memory_manager.store_memory(
            content=message,
            # Don't pass topics - let memory_manager handle auto-tagging
        )
        logger.debug(f"Stored chat message as memory {memory_id}")
        
        # Process message through UI
        response = ui_instance.process_message(message)
        
        return response

    except Exception as e:
        logger.error(f"Error in chat handler: {e}", exc_info=True)
        return f"I apologize, but I encountered an error: {str(e)}" 