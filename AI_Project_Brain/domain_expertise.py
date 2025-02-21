"""
DOMAIN EXPERTISE TRACKER
-----------------------
Tracks and evaluates the system's knowledge domains and capabilities.

This module enables honest self-awareness by:
1. Tracking knowledge accumulation in different domains
2. Measuring knowledge depth and breadth
3. Maintaining confidence scores based on evidence
4. Supporting honest capability assessment

Integration with Memory Access:
- Uses filtered access patterns to adjust confidence
- Prevents confidence inflation from random queries
- Requires repeated engagement to build expertise
- Supports honest self-assessment of capabilities
"""

import logging
logger = logging.getLogger(__name__)
from typing import Dict, List, Set, Optional, Tuple, Any
from datetime import datetime
import json
import uuid
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .db_manager import db_manager
from .text_utils import generate_embedding  # Import from text_utils
from .memory_summarizer import memory_summarizer
from config import MEMORY_SETTINGS, EXPERTISE_SETTINGS
from .memory_access_tracker import memory_access_tracker

class DomainExpertise:
    def __init__(self):
        self._init_database()
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Domain expertise system initialized")
        
    def _init_database(self):
        """Initialize expertise tracking tables"""
        with db_manager.get_cursor() as cursor:
            # Main expertise domains table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS expertise_domains (
                    domain_id TEXT PRIMARY KEY,
                    topic_cluster TEXT NOT NULL,  -- JSON array of topics
                    confidence_level FLOAT DEFAULT 0.0,
                    first_emergence TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    memory_count INTEGER DEFAULT 0,
                    avg_quality FLOAT DEFAULT 0.0,
                    depth_score FLOAT DEFAULT 0.0,
                    breadth_score FLOAT DEFAULT 0.0,
                    evidence_sources TEXT,  -- JSON array of source types
                    CONSTRAINT valid_confidence CHECK (confidence_level >= 0 AND confidence_level <= 1),
                    CONSTRAINT valid_quality CHECK (avg_quality >= 0 AND avg_quality <= 1)
                )
            """)

            # Link memories to domains
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS domain_memories (
                    domain_id TEXT,
                    memory_id TEXT,
                    contribution_score FLOAT DEFAULT 0.5,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (domain_id, memory_id),
                    FOREIGN KEY (domain_id) REFERENCES expertise_domains(domain_id),
                    FOREIGN KEY (memory_id) REFERENCES memory_items(id)
                )
            """)

            # Track domain evolution
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS domain_evolution (
                    domain_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence_delta FLOAT,
                    event_type TEXT,  -- 'memory_added', 'reinforcement', 'decay'
                    details TEXT,  -- JSON with event-specific details
                    PRIMARY KEY (domain_id, timestamp),
                    FOREIGN KEY (domain_id) REFERENCES expertise_domains(domain_id)
                )
            """)

            # Add topic embeddings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topic_embeddings (
                    topic TEXT PRIMARY KEY,
                    embedding TEXT NOT NULL,  -- JSON array of embedding values
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def update_domain_knowledge(self, content: str, topics: Set[str], source_type: str = None, quality_score: float = None) -> str:
        """Update domain knowledge with new content"""
        with db_manager.get_cursor() as cursor:
            # Check for existing memory with same content
            cursor.execute("""
                SELECT id FROM memory_items 
                WHERE content = ?
            """, (content,))
            existing = cursor.fetchone()
            
            if existing:
                memory_id = existing['id']
            else:
                # Create new memory
                memory_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO memory_items (id, content, layer)
                    VALUES (?, ?, 'working')
                """, (memory_id, content))

            # Get or create domain with current timestamp
            domain_id = self._get_or_create_domain(cursor, topics, datetime.now())

            # Link memory to domain if not already linked
            cursor.execute("""
                INSERT OR IGNORE INTO domain_memories (domain_id, memory_id)
                VALUES (?, ?)
            """, (domain_id, memory_id))

            # Update domain metrics
            self._update_domain_metrics(cursor, domain_id, topics, source_type, quality_score)

            return domain_id

    def _merge_domain_memories(self, cursor, source_domain_id: str, target_domain_id: str):
        """Transfer all memories from source domain to target domain"""
        cursor.execute("""
            INSERT OR IGNORE INTO domain_memories (domain_id, memory_id)
            SELECT ?, memory_id 
            FROM domain_memories 
            WHERE domain_id = ?
        """, (target_domain_id, source_domain_id))
        
        # Remove old associations
        cursor.execute("DELETE FROM domain_memories WHERE domain_id = ?", 
                      (source_domain_id,))

    def _get_or_create_domain(self, cursor, topics: Set[str], timestamp: datetime = None) -> str:
        """Get existing domain or create new one for topic cluster"""
        timestamp = timestamp or datetime.now()
        topics_list = sorted(topics)  # Sort for consistent matching
        topics_json = json.dumps(topics_list)
        
        # First try exact match
        cursor.execute("""
            SELECT domain_id, topic_cluster 
            FROM expertise_domains
            WHERE topic_cluster = ?
        """, (topics_json,))
        
        exact_match = cursor.fetchone()
        if exact_match:
            return exact_match['domain_id']
        
        # Then try overlap match
        cursor.execute("""
            SELECT domain_id, topic_cluster 
            FROM expertise_domains
            WHERE EXISTS (
                SELECT 1 FROM json_each(topic_cluster) as t
                WHERE t.value IN (SELECT value FROM json_each(?))
            )
            LIMIT 1
        """, (topics_json,))
        
        overlap_match = cursor.fetchone()
        if overlap_match:
            # Merge topics into existing domain
            merged_topics = set(json.loads(overlap_match['topic_cluster'])) | topics
            cursor.execute("""
                UPDATE expertise_domains 
                SET topic_cluster = ?
                WHERE domain_id = ?
            """, (json.dumps(sorted(merged_topics)), overlap_match['domain_id']))
            
            # Transfer memories from any other domains with these topics
            cursor.execute("""
                SELECT domain_id FROM expertise_domains
                WHERE domain_id != ? AND EXISTS (
                    SELECT 1 FROM json_each(topic_cluster) as t
                    WHERE t.value IN (SELECT value FROM json_each(?))
                )
            """, (overlap_match['domain_id'], topics_json))
            
            for row in cursor.fetchall():
                self._merge_domain_memories(cursor, row['domain_id'], overlap_match['domain_id'])
                # Delete the old domain after merging memories
                cursor.execute("DELETE FROM expertise_domains WHERE domain_id = ?", (row['domain_id'],))
            
            return overlap_match['domain_id']
        
        # Create new domain if no matches found
        domain_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO expertise_domains (
                domain_id, topic_cluster, confidence_level,
                first_emergence, last_activity,
                memory_count, avg_quality,
                evidence_sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            domain_id,
            topics_json,
            0.0,  # Initial confidence
            timestamp,
            timestamp,
            0,  # Initial memory count
            0.0,  # Initial quality
            '[]'  # Empty evidence sources
        ))
        
        return domain_id

    def get_domain_confidence(self, topics: Set[str]) -> float:
        """Get confidence level for a set of topics"""
        try:
            with db_manager.get_cursor() as cursor:
                # Simpler query without json_each
                cursor.execute("""
                    SELECT confidence_level, topic_cluster
                    FROM expertise_domains
                    WHERE topic_cluster IS NOT NULL
                """)
                
                domains = cursor.fetchall()
                if not domains:
                    return 0.0
                
                # Calculate overlap-weighted confidence
                max_confidence = 0.0
                for confidence, domain_topics in domains:
                    # Parse JSON manually since SQLite version may not support JSON functions
                    domain_topic_set = set(json.loads(domain_topics))
                    overlap = len(topics & domain_topic_set)
                    if overlap > 0:
                        weighted_conf = confidence * (overlap / len(topics))
                        max_confidence = max(max_confidence, weighted_conf)
                
                return max_confidence
                
        except Exception as e:
            logging.error(f"Error getting domain confidence: {e}")
            return 0.0

    def _cluster_topics(self, topics: Set[str]) -> List[Set[str]]:
        """Group related topics into clusters"""
        # TODO: Implement more sophisticated clustering
        # For now, use simple grouping based on co-occurrence
        return [topics]  # Return single cluster

    def _update_confidence(self, cursor, domain_id: str):
        """Update domain confidence based on memory count and quality"""
        cursor.execute("""
            UPDATE expertise_domains
            SET confidence_level = MIN(
                confidence_level + (
                    (SELECT COUNT(*) FROM domain_memories WHERE domain_id = ?) * 0.1 +
                    COALESCE(avg_quality, 0) * 0.2
                ),
                1.0  -- Max confidence
            )
            WHERE domain_id = ?
        """, (domain_id, domain_id))

    def _cluster_related_topics(self, topic: str) -> Set[str]:
        """Find related topics based on embedding similarity"""
        # For now, just return empty set since we handle relationships 
        # through domain merging
        return set()

    def track_domain_interaction(self, domain_id: str, interaction_data: Dict):
        """Track a domain interaction"""
        try:
            quality_score = interaction_data.get('quality_score', 
                                               EXPERTISE_SETTINGS['quality']['default'])
            source_type = interaction_data.get('source_type', 'unknown')
            topics = interaction_data.get('topics', set())
            
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    INSERT OR IGNORE INTO domain_expertise
                    (topic, confidence, quality_score)
                    VALUES (?, ?, ?)
                """, (
                    domain_id,
                    EXPERTISE_SETTINGS['confidence']['default'],
                    quality_score
                ))
                
                cursor.execute("""
                    UPDATE domain_expertise
                    SET 
                        confidence = MIN(?, confidence + ?),
                        interaction_count = COALESCE(interaction_count, 0) + 1,
                        quality_score = ?,
                        source_diversity = COALESCE(source_diversity, 0) + 1
                    WHERE topic = ?
                """, (
                    EXPERTISE_SETTINGS['confidence']['max'],
                    quality_score,
                    quality_score,
                    domain_id
                ))
        except Exception as e:
            logging.error(f"Error tracking domain interaction: {str(e)}")
            raise

    def _update_domain_scores(self, cursor, domain_id: str, topics: Set[str]) -> None:
        """Update domain depth and breadth scores"""
        # Get existing topic cluster
        cursor.execute("""
            SELECT topic_cluster
            FROM expertise_domains
            WHERE domain_id = ?
        """, (domain_id,))
        
        existing_topics = set(json.loads(cursor.fetchone()[0]))
        
        # Calculate new scores
        breadth_score = len(existing_topics.union(topics)) / EXPERTISE_SETTINGS['max_topics']
        depth_score = len(existing_topics.intersection(topics)) / len(existing_topics) if existing_topics else 0
        
        # Update scores
        cursor.execute("""
            UPDATE expertise_domains
            SET 
                depth_score = ?,
                breadth_score = ?,
                topic_cluster = ?
            WHERE domain_id = ?
        """, (
            depth_score,
            breadth_score,
            json.dumps(list(existing_topics.union(topics))),
            domain_id
        ))

    def update_domain_confidence(self, memory_id: str, domain_id: str):
        """Update domain confidence based on significant access patterns
        
        Integration Notes:
        - Only considers patterns that meet min_occurrences threshold
        - Confidence increases are based on repeated, meaningful engagement
        - Helps maintain accurate self-awareness of knowledge depth
        """
        patterns = memory_access_tracker.analyze_access_patterns(memory_id)
        significant_patterns = [p for p in patterns['hourly_patterns'] 
                              if p['count'] >= MEMORY_SETTINGS['access_tracking']['patterns']['min_occurrences']]
        
        if significant_patterns:
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    UPDATE expertise_domains
                    SET confidence_level = MIN(confidence_level + ?, ?)
                    WHERE domain_id = ?
                """, (
                    EXPERTISE_SETTINGS['confidence']['increment_per_pattern'] * len(significant_patterns),
                    EXPERTISE_SETTINGS['confidence']['max_level'],
                    domain_id
                ))
                
                logger.debug(f"Updated confidence for domain {domain_id} based on {len(significant_patterns)} patterns")

    def _update_domain_metrics(self, cursor, domain_id: str, topics: Set[str], source_type: str = None, quality_score: float = None):
        """Update domain metrics including memory count, quality, and evidence"""
        # Get current memory count first
        cursor.execute("""
            SELECT COUNT(DISTINCT memory_id) as count 
            FROM domain_memories 
            WHERE domain_id = ?
        """, (domain_id,))
        memory_count = cursor.fetchone()['count']

        # Update metrics
        cursor.execute("""
            UPDATE expertise_domains SET
                memory_count = ?,
                avg_quality = COALESCE(?, avg_quality),
                last_activity = CURRENT_TIMESTAMP,
                evidence_sources = json_insert(
                    COALESCE(evidence_sources, '[]'),
                    '$[#]', 
                    ?
                )
            WHERE domain_id = ?
        """, (memory_count, quality_score, source_type, domain_id))
        
        # Update confidence
        self._update_confidence(cursor, domain_id)

# Global instance
domain_expertise = DomainExpertise() 