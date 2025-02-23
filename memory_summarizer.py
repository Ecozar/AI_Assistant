"""
MEMORY SUMMARIZER
----------------
Processes conversation history into long-term memory summaries.

Core Responsibilities:
1. Periodically analyze conversation batches
2. Generate meaningful summaries
3. Store summaries with importance weights
4. Maintain summary history for personality development

Design Philosophy:
- Summaries must be retrievable by topic/time
- Must maintain SSOT with existing tag system
- Must support future LLM integration
- Must handle periodic updates efficiently
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import math

from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.auto_tagger import load_approved_tags
from config import MEMORY_SETTINGS

class MemorySummarizer:
    def __init__(self):
        self._init_summary_table()
        self.approved_tags = load_approved_tags()  # This is all we need
    
    def _init_summary_table(self):
        """Initialize the summary storage table"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    id INTEGER PRIMARY KEY,
                    time_period TEXT,        -- e.g., "2025-02-W06" for week 6
                    summary TEXT,
                    key_topics TEXT,         -- JSON list of main topics
                    importance FLOAT,
                    tags TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_summaries_period 
                ON memory_summaries(time_period)
            """)

    def generate_period_summary(self, period: str = MEMORY_SETTINGS['summary_period']) -> Optional[Dict]:
        """Generate or update summary for specified time period"""
        try:
            with db_manager.get_cursor() as cursor:
                period_id = self._get_period_id(period)
                
                # Get all unsummarized conversations
                cursor.execute("""
                    SELECT user_message, assistant_message, tags
                    FROM conversation_history 
                    WHERE id NOT IN (
                        SELECT DISTINCT conversation_id 
                        FROM summarized_conversations
                    )
                    ORDER BY timestamp DESC
                """)
                
                conversations = cursor.fetchall()
                if not conversations:
                    return None
                    
                # Check if we already have a summary for today
                cursor.execute("""
                    SELECT id FROM memory_summaries 
                    WHERE time_period = ? 
                    AND DATE(created_at) = DATE('now')
                """, (period_id,))
                
                if cursor.fetchone():
                    logging.info(f"Summary already exists for period {period_id} today")
                    return None
                    
                # Create new summary
                summary = self._create_summary(conversations)
                
                # Record which conversations were summarized
                cursor.execute("""
                    INSERT INTO summarized_conversations (conversation_id, summary_id)
                    SELECT id, ? FROM conversation_history 
                    WHERE id NOT IN (
                        SELECT conversation_id FROM summarized_conversations
                    )
                """, (period_id,))
                
                # Create the summary
                cursor.execute("""
                    INSERT INTO memory_summaries 
                    (time_period, summary, key_topics, importance, tags)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    period_id,
                    summary['text'],
                    json.dumps(summary['topics']),
                    summary['importance'],
                    summary['tags']
                ))
                
                logging.info(f"Created new summary for {len(conversations)} conversations")
                return summary
                
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return None
    
    def get_relevant_summaries(self, 
                             topic: Optional[str] = None,
                             tags: Optional[List[str]] = None,
                             limit: int = 5) -> List[Dict]:
        """Retrieve relevant summaries for context building"""
        try:
            with db_manager.get_cursor() as cursor:
                query = """
                    SELECT time_period, summary, key_topics, importance, tags
                    FROM memory_summaries
                    WHERE 1=1
                """
                params = []
                
                if topic:
                    query += " AND key_topics LIKE ?"
                    params.append(f"%{topic}%")
                    
                if tags:
                    query += " AND tags LIKE ?"
                    params.append(f"%{','.join(tags)}%")
                    
                query += " ORDER BY importance DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            logging.error(f"Error retrieving summaries: {e}")
            return []

    def _get_period_id(self, period: str) -> str:
        """Generate standardized period identifier"""
        now = datetime.now()
        if period == 'week':
            return now.strftime("%Y-%m-W%V")
        elif period == 'month':
            return now.strftime("%Y-%m")
        return now.strftime("%Y-%m-%d")

    def _create_summary(self, conversations: List[tuple]) -> Dict:
        try:
            time_span = self._get_conversation_timespan(conversations)
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT timestamp 
                    FROM conversation_history
                    WHERE id NOT IN (SELECT conversation_id FROM summarized_conversations)
                    ORDER BY timestamp DESC
                """)
                # Use timestamp helper
                timestamps = [self._parse_timestamp(row[0]) for row in cursor.fetchall()]
            
            if not timestamps:
                return {
                    'text': "No conversations to summarize",
                    'topics': [],
                    'importance': 0.1,
                    'tags': ""
                }
            
            # Process tags and build topic counts
            topic_counts = {}
            topic_first_seen = {}
            all_topics = []
            
            for user_msg, asst_msg, tag_string in conversations:
                tags = self._parse_tags(tag_string)
                all_topics.extend(tags)
                for tag in tags:
                    topic_counts[tag] = topic_counts.get(tag, 0) + 1
                    if tag not in topic_first_seen:
                        topic_first_seen[tag] = timestamps[-1]  # Earliest timestamp
            
            # Calculate importance scores
            density_score = self._calculate_density_score(len(conversations), time_span)
            topic_scores = self._calculate_topic_scores(topic_counts, topic_first_seen, timestamps)
            
            # Update memory and get final importance
            # Use timestamp helper
            timestamp_str = self._format_timestamp(timestamps[0])
            self._update_topic_memory(topic_scores, timestamp_str)
            importance = self._calculate_final_importance(density_score, topic_scores)
            
            # Create summary text
            topics = sorted(topic_counts, key=topic_counts.get, reverse=True)[:5]
            topic_str = ", ".join(topics) if topics else "various topics"
            summary_text = f"Summary of {len(conversations)} conversations about {topic_str}. "
            summary_text += "Topics discussed include: " + topic_str
            
            return {
                'text': summary_text,
                'topics': topics,
                'importance': importance,
                'tags': ','.join(topics[:MEMORY_SETTINGS['max_common_tags']])
            }
            
        except Exception as e:
            logging.error(f"Error creating summary: {e}")
            return {
                'text': "Error generating summary",
                'topics': [],
                'importance': 0.1,
                'tags': ""
            }

    def _get_conversation_timespan(self, conversations: List[tuple]) -> timedelta:
        """Calculate time between first and last conversation"""
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT MAX(timestamp), MIN(timestamp)
                    FROM conversation_history
                    WHERE id IN (
                        SELECT id FROM conversation_history 
                        WHERE id NOT IN (SELECT conversation_id FROM summarized_conversations)
                    )
                """)
                max_time, min_time = cursor.fetchone()
                if not max_time or not min_time:
                    return timedelta(0)
                # Use timestamp helpers
                return self._parse_timestamp(max_time) - self._parse_timestamp(min_time)
        except Exception as e:
            logging.error(f"Error calculating timespan: {e}")
            return timedelta(0)

    def _extract_common_tags(self, conversations: List[tuple]) -> List[str]:
        """Extract most common tags from conversations"""
        tag_counts = {}
        for _, _, tags in conversations:
            if tags:
                for tag in tags.split(','):
                    tag = tag.strip()
                    if tag in self.approved_tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return sorted(
            tag_counts, 
            key=tag_counts.get, 
            reverse=True
        )[:MEMORY_SETTINGS['max_common_tags']]

    def get_period_summaries(self, start_period: str = None, end_period: str = None, limit: int = 10) -> List[Dict]:
        """Retrieve summaries for a date range"""
        try:
            with db_manager.get_cursor() as cursor:
                query = """
                    SELECT time_period, summary, key_topics, importance, tags, created_at
                    FROM memory_summaries
                    WHERE 1=1
                """
                params = []
                
                if start_period:
                    query += " AND time_period >= ?"
                    params.append(start_period)
                if end_period:
                    query += " AND time_period <= ?"
                    params.append(end_period)
                    
                query += " ORDER BY time_period DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                return cursor.fetchall()
                
        except Exception as e:
            logging.error(f"Error retrieving period summaries: {e}")
            return []

    def _calculate_importance(self, conversations: List[tuple], time_span: timedelta) -> float:
        """Calculate importance using time decay and topic patterns"""
        try:
            # Time decay calculation
            time_now = datetime.now()
            decay_factor = 0.95  # Daily decay rate
            
            # Get conversation timestamps
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT timestamp 
                    FROM conversation_history
                    WHERE id IN (
                        SELECT id FROM conversation_history 
                        WHERE id NOT IN (SELECT conversation_id FROM summarized_conversations)
                    )
                    ORDER BY timestamp DESC
                """)
                # Use timestamp helper
                timestamps = [self._parse_timestamp(row[0]) for row in cursor.fetchall()]
            
            # Calculate weighted count with decay
            weighted_count = sum(
                decay_factor ** ((time_now - conv_time).days) 
                for conv_time in timestamps
            )
            
            # Density score with decay
            hours = time_span.total_seconds() / 3600
            density_score = min(1.0, weighted_count / (hours + 1))
            
            # Topic importance with persistence check
            topic_counts = {}
            topic_first_seen = {}
            
            for _, _, tag_string in conversations:
                tags = self._parse_tags(tag_string)  # Use new safe parser
                for tag in tags:
                    topic_counts[tag] = topic_counts.get(tag, 0) + 1
                    if tag not in topic_first_seen:
                        topic_first_seen[tag] = timestamps[-1]
            
            # Calculate topic persistence
            topic_scores = {}
            for tag, count in topic_counts.items():
                time_present = (timestamps[0] - topic_first_seen[tag]).total_seconds() / 3600
                topic_scores[tag] = (count / len(conversations)) * (1 - math.exp(-time_present/24))
            
            topic_score = max(topic_scores.values()) if topic_scores else 0
            
            # Update topic memory layers
            timestamp_str = self._format_timestamp(timestamps[0])
            self._update_topic_memory(topic_scores, timestamp_str)
            
            # Dynamic weight based on topic persistence
            persistent_topics = len([s for s in topic_scores.values() if s > 0.3])
            weight_factor = 0.5 + min(0.3, persistent_topics * 0.1)
            
            # Combine scores
            importance = (weight_factor * density_score) + ((1 - weight_factor) * topic_score)
            return min(1.0, importance)
            
        except Exception as e:
            logging.error(f"Error calculating importance: {e}")
            return 0.1

    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic string for consistent matching"""
        return topic.strip().lower()

    def _parse_tags(self, tag_string: str) -> List[str]:
        """Safely parse and validate tags"""
        if not tag_string or not isinstance(tag_string, str):
            return []
        
        try:
            # Clean and split tags
            tags = [
                self._normalize_topic(tag)
                for tag in tag_string.split(',')
                if tag.strip()
            ]
            
            # Only return valid normalized tags
            return [
                tag for tag in tags 
                if tag in [t.lower() for t in self.approved_tags]
            ]
            
        except Exception as e:
            logging.error(f"Error parsing tags: {e}")
            return []

    def _update_topic_memory(self, topics: Dict[str, float], timestamp: str):
        """Update topic memory layers based on persistence"""
        try:
            with db_manager.get_cursor() as cursor:
                # First, log all existing topics for comparison
                cursor.execute("SELECT topic, memory_layer, mention_count FROM topic_memory")
                existing_topics = cursor.fetchall()
                logging.info(f"=== Current Topic Memory State ===")
                logging.info(f"Existing topics in DB: {existing_topics}")
                
                for topic, importance in topics.items():
                    normalized_topic = self._normalize_topic(topic)
                    logging.info(f"\nProcessing topic: {topic}")
                    logging.info(f"Normalized to: {normalized_topic}")
                    
                    # Get existing memory entry with more detail
                    cursor.execute("""
                        SELECT memory_layer, first_seen, mention_count, importance, topic
                        FROM topic_memory 
                        WHERE topic = ?
                    """, (normalized_topic,))
                    existing = cursor.fetchone()
                    logging.info(f"Found existing record: {existing}")
                    
                    if existing:
                        logging.info(f"Updating existing record for {normalized_topic}")
                        layer, first_seen, count, old_importance, existing_topic = existing
                        
                        # Use timestamp helpers for calculation
                        time_diff = self._parse_timestamp(timestamp) - self._parse_timestamp(first_seen)
                        
                        # Promote to next layer if criteria met
                        new_layer = layer
                        if layer == 'short' and time_diff.days >= 7 and count >= 10:
                            new_layer = 'medium'
                        elif layer == 'medium' and time_diff.days >= 30 and count >= 30:
                            new_layer = 'long'
                        
                        # Update with rolling average importance
                        new_importance = (old_importance * count + importance) / (count + 1)
                        
                        cursor.execute("""
                            UPDATE topic_memory
                            SET memory_layer = ?,
                                last_seen = ?,
                                mention_count = mention_count + 1,
                                importance = ?
                            WHERE topic = ?
                        """, (new_layer, timestamp, new_importance, normalized_topic))
                    else:
                        logging.info(f"Creating new record for {normalized_topic}")
                        cursor.execute("""
                            INSERT INTO topic_memory 
                            (topic, memory_layer, first_seen, last_seen, mention_count, importance)
                            VALUES (?, 'short', ?, ?, 1, ?)
                        """, (normalized_topic, timestamp, timestamp, importance))
                
        except Exception as e:
            logging.error(f"Error updating topic memory: {e}")

    def _calculate_density_score(self, conversation_count: int, time_span: timedelta) -> float:
        """Calculate conversation density score with decay"""
        try:
            hours = time_span.total_seconds() / 3600
            if hours == 0:
                return 1.0
            return min(1.0, conversation_count / (hours + 1))
        except Exception as e:
            logging.error(f"Error calculating density score: {e}")
            return 0.1

    def _calculate_topic_scores(self, topic_counts: Dict[str, int], 
                              topic_first_seen: Dict[str, datetime],
                              timestamps: List[datetime]) -> Dict[str, float]:
        """Calculate topic importance scores with persistence"""
        try:
            topic_scores = {}
            epsilon = 0.1  # Ensure non-zero scores for new topics
            
            logging.info("=== Topic Score Calculation ===")
            logging.info(f"Input topic_counts: {topic_counts}")
            
            for tag, count in topic_counts.items():
                time_present = (timestamps[0] - topic_first_seen[tag]).total_seconds() / 3600
                # Add epsilon to avoid zero scores
                score = (count / len(timestamps)) * (1 - math.exp(-(time_present + epsilon)/24))
                topic_scores[tag] = score
                logging.info(f"Calculated for {tag}: count={count}, time_present={time_present}h, score={score}")
            
            return topic_scores
        except Exception as e:
            logging.error(f"Error calculating topic scores: {e}", exc_info=True)
            return {}

    def _calculate_final_importance(self, density_score: float, 
                                  topic_scores: Dict[str, float]) -> float:
        """Calculate final importance score"""
        try:
            persistent_topics = len([s for s in topic_scores.values() if s > 0.3])
            weight_factor = 0.5 + min(0.3, persistent_topics * 0.1)
            return min(1.0, (weight_factor * density_score) + 
                           ((1 - weight_factor) * max(topic_scores.values(), default=0)))
        except Exception as e:
            logging.error(f"Error calculating final importance: {e}")
            return 0.1

    def _format_timestamp(self, dt: datetime) -> str:
        """Ensure consistent timestamp formatting"""
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def _parse_timestamp(self, ts: str) -> datetime:
        """Safely parse timestamp string"""
        return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')

# Global instance
memory_summarizer = MemorySummarizer() 