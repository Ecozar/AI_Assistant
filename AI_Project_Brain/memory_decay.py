"""
MEMORY DECAY
-----------
Handles gradual decay of memory importance and connections over time.
"""

import threading
import time
from datetime import datetime, timedelta
import logging
from typing import Dict, Set

from .db_manager import db_manager
from .memory_layers import MemoryLayer
from config import MEMORY_SETTINGS, EXPERTISE_SETTINGS

class MemoryDecayService:
    def __init__(self):
        self._stop_event = threading.Event()
        self._decay_thread = None
        self.logger = logging.getLogger('memory_decay')
        
    def start(self):
        """Start the decay service"""
        if self._decay_thread and self._decay_thread.is_alive():
            self.logger.warning("Decay service already running")
            return
            
        self._stop_event.clear()
        self._decay_thread = threading.Thread(target=self._decay_loop, daemon=True)
        self._decay_thread.start()
        self.logger.info("Memory decay service started")
    
    def stop(self):
        """Stop the decay service"""
        self.logger.info("Stopping decay service...")
        self._stop_event.set()
        if self._decay_thread:
            self._decay_thread.join(timeout=5.0)
            if self._decay_thread.is_alive():
                self.logger.warning("Decay thread did not stop cleanly")
            else:
                self.logger.info("Decay service stopped")
    
    def _decay_loop(self):
        """Periodically apply decay to memories"""
        while not self._stop_event.is_set():
            try:
                protected_ids = self._get_protected_memories()
                self._apply_decay(protected_ids)
                self.logger.debug(f"Applied decay, protected {len(protected_ids)} memories")
                
                # Sleep in small intervals to check stop event
                for _ in range(MEMORY_SETTINGS['decay']['check_interval']):
                    if self._stop_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Error in decay loop: {str(e)}", exc_info=True)
                time.sleep(60)  # Wait before retry
    
    def _get_protected_memories(self) -> Set[str]:
        """Get IDs of memories that should be protected from decay"""
        protected = set()
        try:
            with db_manager.get_cursor() as cursor:
                # Check connection strength protection
                cursor.execute("""
                    SELECT DISTINCT m.id 
                    FROM memory_items m
                    JOIN memory_connections c 
                    ON m.id = c.source_id OR m.id = c.target_id
                    GROUP BY m.id
                    HAVING COUNT(CASE WHEN c.strength >= ? THEN 1 END) >= ?
                """, (
                    MEMORY_SETTINGS['protection']['min_connection_strength'],
                    MEMORY_SETTINGS['protection']['min_strong_connections']
                ))
                protected.update(row[0] for row in cursor.fetchall())
                
                # Check access frequency and importance protection
                cursor.execute("""
                    SELECT id FROM memory_items
                    WHERE access_count >= ?
                    OR importance >= ?
                    OR (julianday('now') - julianday(last_accessed)) <= ?
                """, (
                    MEMORY_SETTINGS['protection']['min_access_count'],
                    MEMORY_SETTINGS['protection']['importance_threshold'],
                    MEMORY_SETTINGS['protection']['recent_access_window'] / (24 * 60 * 60)
                ))
                protected.update(row[0] for row in cursor.fetchall())
                
                self.logger.debug(f"Protected memories: {len(protected)}")
                
        except Exception as e:
            self.logger.error(f"Error getting protected memories: {str(e)}", exc_info=True)
        
        return protected
    
    def _apply_decay(self, protected_ids: Set[str]):
        """Apply decay to both memories and their connections"""
        try:
            with db_manager.get_cursor() as cursor:
                # First decay the memories themselves
                for layer in MemoryLayer:
                    if layer == MemoryLayer.PERMANENT:
                        continue
                        
                    decay_rate = MEMORY_SETTINGS['decay']['layers'][layer.value]['rate']
                    min_importance = MEMORY_SETTINGS['decay']['layers'][layer.value]['min_importance']
                    
                    # Apply memory decay with protection
                    if protected_ids:
                        cursor.execute("""
                            UPDATE memory_items
                            SET importance = MAX(?, importance * (1.0 - ?))
                            WHERE layer = ? AND id NOT IN ({})
                        """.format(','.join('?' * len(protected_ids))),
                        [min_importance, decay_rate, layer.value] + list(protected_ids))
                    else:
                        cursor.execute("""
                            UPDATE memory_items
                            SET importance = MAX(?, importance * (1.0 - ?))
                            WHERE layer = ?
                        """, (min_importance, decay_rate, layer.value))

                # Then decay the connections
                cursor.execute("""
                    UPDATE memory_connections
                    SET strength = strength * (1.0 - COALESCE(
                        (SELECT decay_rate FROM connection_metadata 
                         WHERE connection_metadata.source_id = memory_connections.source_id
                         AND connection_metadata.target_id = memory_connections.target_id),
                        0.1  -- Default decay rate if no metadata
                    ))
                    WHERE (source_id, target_id) NOT IN (
                        SELECT source_id, target_id 
                        FROM connection_metadata
                        WHERE julianday('now') - julianday(last_reinforced) < ?
                    )
                """, (MEMORY_SETTINGS['decay']['protected_period'] / 24.0,))  # Convert hours to days

                # Remove very weak connections
                cursor.execute("""
                    DELETE FROM memory_connections 
                    WHERE strength < ?
                """, (MEMORY_SETTINGS['connections']['min_strength'],))

        except Exception as e:
            self.logger.error(f"Error applying decay: {str(e)}", exc_info=True)

# Global instance
memory_decay = MemoryDecayService() 