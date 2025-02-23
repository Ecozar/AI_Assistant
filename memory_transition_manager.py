"""
MEMORY TRANSITION MANAGER
------------------------
Manages memory layer transitions based on:
- Access patterns (from memory_access_tracker)
- Domain expertise
- Memory relationships
"""

import logging
logger = logging.getLogger(__name__)
from typing import List, Set, Dict, Optional
from datetime import datetime

from .db_manager import db_manager
from .memory_layers import MemoryLayer
from .memory_access_tracker import memory_access_tracker
from .domain_expertise import domain_expertise
from config import MEMORY_SETTINGS
from .memory_manager import memory_manager

class MemoryTransitionManager:
    def check_transitions(self, memory_id: str):
        """Check and execute memory layer transitions"""
        logger.info(f"\n=== Starting transition check for memory {memory_id} ===")
        
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT layer, importance
                FROM memory_items
                WHERE id = ?
            """, (memory_id,))
            memory = cursor.fetchone()
            
            if not memory:
                logger.warning(f"Memory {memory_id} not found")
                return
                
            current_layer = memory['layer']
            logger.info(f"Current layer: {current_layer}, importance: {memory['importance']}")
            
            # 1. Check access patterns
            patterns = memory_access_tracker.analyze_access_patterns(memory_id)
            significant_patterns = [p for p in patterns['hourly_patterns'] 
                                 if p['count'] >= MEMORY_SETTINGS['access_tracking']['patterns']['min_occurrences']]
            logger.info(f"Found {len(significant_patterns)} significant patterns: {significant_patterns}")
            
            # 2. Get domain influence
            cursor.execute("""
                SELECT m.content, m.topics
                FROM memory_items m
                WHERE m.id = ?
            """, (memory_id,))
            memory_data = cursor.fetchone()
            logger.info(f"Memory topics: {memory_data['topics']}")
            
            # Find matching domains based on topics
            cursor.execute("""
                SELECT domain_id, confidence_level, topic_cluster
                FROM expertise_domains
                WHERE EXISTS (
                    SELECT 1 
                    FROM json_each(topic_cluster) t1
                    WHERE t1.value IN (
                        SELECT value 
                        FROM json_each(json(?))
                    )
                )
                ORDER BY confidence_level DESC
                LIMIT 1
            """, (memory_data['topics'],))
            domain = cursor.fetchone()
            logger.info(f"Matching domain: {domain['domain_id'] if domain else 'None'}")
            if domain:
                logger.info(f"Domain confidence: {domain['confidence_level']}")
            
            # 3. Get related memories
            cursor.execute("""
                SELECT target_id, connection_type, strength
                FROM memory_connections
                WHERE source_id = ? AND strength >= ?
            """, (
                memory_id,
                MEMORY_SETTINGS['transitions']['connection_strength_threshold']
            ))
            related_memories = cursor.fetchall()
            logger.info(f"Found {len(related_memories)} related memories above threshold")
            for rel in related_memories:
                logger.info(f"Related memory: {rel['target_id']}, strength: {rel['strength']}")
            
            # Determine if promotion is needed
            total_accesses = sum(p['count'] for p in significant_patterns)
            should_promote = (
                total_accesses >= MEMORY_SETTINGS['transitions']['min_significant_patterns'] or
                (domain and domain['confidence_level'] >= MEMORY_SETTINGS['transitions']['domain_confidence_threshold'])
            )
            logger.info(f"Should promote: {should_promote}")
            logger.info(f"Promotion criteria:")
            logger.info(f"- Significant patterns: {total_accesses} >= {MEMORY_SETTINGS['transitions']['min_significant_patterns']}")
            logger.info(f"- Domain confidence: {domain['confidence_level'] if domain else 'None'} >= {MEMORY_SETTINGS['transitions']['domain_confidence_threshold']}")
            
            if should_promote and current_layer == MemoryLayer.WORKING.value:
                # Promote to short-term
                logger.info(f"Promoting memory {memory_id} to short-term")
                self._promote_memory(cursor, memory_id, MemoryLayer.SHORT_TERM.value)
                
                # Promote strongly related memories
                for related in related_memories:
                    logger.info(f"Promoting related memory {related['target_id']}")
                    self._promote_memory(cursor, related['target_id'], MemoryLayer.SHORT_TERM.value)
                    
                logger.info(f"Promotion complete for memory {memory_id} and {len(related_memories)} related memories")
            else:
                logger.info(f"No promotion needed. should_promote: {should_promote}, current_layer: {current_layer}")
            
            logger.info("=== Transition check complete ===\n")
    
    def _promote_memory(self, cursor, memory_id: str, new_layer: str):
        """Promote a memory to a new layer"""
        logger.info(f"Promoting memory {memory_id} to {new_layer}")
        
        # Get current state
        cursor.execute("SELECT layer, importance FROM memory_items WHERE id = ?", (memory_id,))
        before = cursor.fetchone()
        logger.info(f"Before promotion - Layer: {before['layer']}, Importance: {before['importance']}")
        
        cursor.execute("""
            UPDATE memory_items
            SET layer = ?,
                importance = MIN(importance + ?, 1.0)
            WHERE id = ?
        """, (
            new_layer,
            MEMORY_SETTINGS['transitions']['importance_increment'],
            memory_id
        ))
        
        # Verify update
        cursor.execute("SELECT layer, importance FROM memory_items WHERE id = ?", (memory_id,))
        after = cursor.fetchone()
        logger.info(f"After promotion - Layer: {after['layer']}, Importance: {after['importance']}")

# Global instance
memory_transition_manager = MemoryTransitionManager() 