"""
MEMORY ACCESS TRACKER
-------------------
Core component for understanding memory usage patterns while filtering noise.

Integration Points:
- Domain Expertise: Provides filtered patterns to help adjust domain confidence
- Memory Optimizer: Informs storage decisions based on access frequency
- Personality System: Helps identify truly important topics vs passing interests

Noise Filtering:
- Uses min_occurrences threshold to identify significant patterns
- Prevents one-off queries from affecting the AI's knowledge model
- Matches human memory formation (repeated exposure matters)
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import threading
import json
import uuid

from .db_manager import (
    db_manager,
    ACCESS_TRACKER_SCHEMA,
    ACCESS_TRACKER_INDICES
)
from config import MEMORY_SETTINGS

logger = logging.getLogger(__name__)

class Pattern:
    """Class to represent memory access patterns with attribute and dictionary access"""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def to_dict(self):
        """Convert pattern to dictionary for storage"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
    def __getattr__(self, name):
        """Fallback for attribute access"""
        return self.__dict__.get(name)
        
    def __getitem__(self, key):
        """Support dictionary-style access"""
        return self.__dict__[key]
        
    def __setitem__(self, key, value):
        """Support dictionary-style assignment"""
        self.__dict__[key] = value
        
    def get(self, key, default=None):
        """Dictionary-style get with default"""
        return self.__dict__.get(key, default)
        
    def __contains__(self, key):
        """Support 'in' operator"""
        return key in self.__dict__
        
    def __bool__(self):
        """Support truth value testing"""
        return bool(self.__dict__)

class MemoryAccessTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._init_database()
        logger.info("Memory access tracker initialized")

    def _init_database(self):
        """Initialize database tables"""
        with db_manager.get_cursor() as cursor:
            # Create access log table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    access_type TEXT NOT NULL,
                    context TEXT,
                    source TEXT,
                    access_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    chain_id TEXT
                )
            """)
            
            # Create patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory_access_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    confidence FLOAT NOT NULL,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    occurrence_count INTEGER DEFAULT 1,
                    FOREIGN KEY (memory_id) REFERENCES memory_items(id)
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_memory_id 
                ON memory_access_log(memory_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_time 
                ON memory_access_log(access_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern_memory 
                ON memory_access_patterns(memory_id)
            """)

    def record_access(self, memory_id: str, access_type: str, context: str, source: str, 
                     chain_id: Optional[str] = None, access_time: Optional[datetime] = None):
        """Record a memory access event"""
        try:
            with db_manager.get_cursor() as cursor:
                # Record in access log
                cursor.execute("""
                    INSERT INTO memory_access_log
                    (memory_id, access_type, context, source, chain_id, access_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (memory_id, access_type, context, source, chain_id, 
                     access_time.strftime('%Y-%m-%d %H:%M:%S') if access_time else None))
                
                # Update last_accessed timestamp
                cursor.execute("""
                    UPDATE memory_items 
                    SET last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (memory_id,))
                
                logger.debug(f"Recorded access - Memory: {memory_id}, Type: {access_type}")
        except Exception as e:
            logger.error(f"Failed to record access: {e}")
            raise

    def _cleanup_old_records(self, cursor, memory_id: str, settings: Dict):
        """Clean up old access records to maintain size limits"""
        try:
            # Get count of records for this memory
            cursor.execute("""
                SELECT COUNT(*) FROM memory_access_log
                WHERE memory_id = ?
            """, (memory_id,))
            count = cursor.fetchone()[0]
            
            if count > settings['max_history_per_memory']:
                # Delete oldest records keeping max_history_per_memory most recent
                cursor.execute("""
                    DELETE FROM memory_access_log
                    WHERE memory_id = ? AND id NOT IN (
                        SELECT id FROM memory_access_log
                        WHERE memory_id = ?
                        ORDER BY access_time DESC
                        LIMIT ?
                    )
                """, (memory_id, memory_id, settings['max_history_per_memory']))
                
                deleted = cursor.rowcount
                logger.debug(f"Cleaned up {deleted} old access records for memory {memory_id}")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old access records: {e}")
            raise

    def analyze_access_patterns(self, memory_id: str) -> Dict:
        """Analyze access patterns for a memory with pattern aliases"""
        logger.debug(f"Starting pattern analysis for memory: {memory_id}")
        try:
            with db_manager.get_cursor() as cursor:
                # First verify the memory exists
                cursor.execute("SELECT id FROM memory_items WHERE id = ?", (memory_id,))
                result = cursor.fetchone()
                logger.debug(f"Memory existence check result: {result}")
                if not result:
                    raise KeyError("No item with that key")

                # Get all access records for this memory
                cursor.execute("""
                    SELECT access_type, context, access_time, chain_id, source
                    FROM memory_access_log
                    WHERE memory_id = ?
                    ORDER BY access_time ASC
                """, (memory_id,))
                
                accesses = cursor.fetchall()
                logger.debug(f"Found {len(accesses)} access records")
                logger.debug(f"Access records: {accesses}")
                
                # Analyze patterns using existing methods
                logger.debug("Starting pattern analysis...")
                hourly = self._analyze_hourly_patterns(accesses)
                logger.debug(f"Hourly patterns: {hourly}")
                periodic = self._analyze_periodicities(accesses)
                logger.debug(f"Periodic patterns: {periodic}")
                sequence = self._analyze_sequences(accesses)
                logger.debug(f"Sequence patterns: {sequence}")
                contextual = self._analyze_contexts(accesses)
                logger.debug(f"Contextual patterns: {contextual}")
                
                patterns = {
                    'hourly_patterns': hourly,
                    'frequency_patterns': hourly,  # Alias
                    'periodic_patterns': periodic,
                    'time_patterns': periodic,     # Alias
                    'sequence_patterns': sequence,
                    'chain_patterns': sequence,    # Alias
                    'contextual_patterns': contextual
                }
                
                logger.debug(f"Final pattern dictionary: {patterns}")
                return patterns
                
        except Exception as e:
            logger.error(f"Error analyzing access patterns: {e}", exc_info=True)
            return {
                'hourly_patterns': [], 'frequency_patterns': [],
                'periodic_patterns': [], 'time_patterns': [],
                'sequence_patterns': [], 'chain_patterns': [],
                'contextual_patterns': []
            }

    def _analyze_hourly_patterns(self, records: List[Dict]) -> List[Dict]:
        """Analyze hourly access patterns"""
        try:
            # Group accesses by hour
            hour_counts = {}
            for record in records:
                hour = datetime.fromisoformat(record['access_time']).strftime('%H')
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            
            # Get pattern settings
            pattern_settings = MEMORY_SETTINGS['access_tracking']['patterns']
            min_occurrences = pattern_settings['min_occurrences']
            max_occurrences = pattern_settings.get('max_occurrences', pattern_settings['max_sequence_occurrences'])
            
            # Create patterns only for hours that meet minimum occurrence threshold
            patterns = []
            for hour, count in hour_counts.items():
                if count >= min_occurrences:  # Only include if meets minimum threshold
                    pattern = Pattern(
                        hour=hour,
                        count=count,
                        confidence=min(1.0, count / max_occurrences),
                        pattern_type='hourly_patterns'
                    )
                    patterns.append(pattern)
                    logger.debug(f"Found significant hourly pattern: hour={hour}, count={count}")
                else:
                    logger.debug(f"Filtered out noise pattern: hour={hour}, count={count}")
            
            # Sort by count descending
            return sorted(patterns, key=lambda x: (-x['count'], x['hour']))
        except Exception as e:
            logger.error(f"Error in hourly pattern analysis: {e}")
            return []

    def _analyze_sequences(self, records: List[Dict]) -> List[Dict]:
        """Analyze for repeated access sequences with variable lengths and time considerations"""
        try:
            sequences = []
            min_window = 2  # Minimum sequence length
            max_window = MEMORY_SETTINGS['access_tracking']['patterns']['sequence_window']
            max_time_gap = MEMORY_SETTINGS['access_tracking']['patterns'].get('max_sequence_time_gap', 3600)  # 1 hour default
            
            # Convert records to sequence of (type, context, timestamp) tuples
            access_sequence = [(
                r['access_type'],
                r['context'],
                datetime.fromisoformat(r['access_time'])
            ) for r in records]
            
            # Analyze different window sizes
            for window_size in range(min_window, max_window + 1):
                # Find repeated subsequences
                for i in range(len(access_sequence) - window_size + 1):
                    # Check time gap within sequence
                    current_seq = access_sequence[i:i + window_size]
                    if any((current_seq[j+1][2] - current_seq[j][2]).total_seconds() > max_time_gap
                           for j in range(len(current_seq)-1)):
                        continue
                    
                    # Create pattern without timestamps
                    pattern = tuple((t, c) for t, c, _ in current_seq)
                    
                    # Count occurrences considering time gaps
                    count = 0
                    for j in range(len(access_sequence) - window_size + 1):
                        check_seq = access_sequence[j:j + window_size]
                        check_pattern = tuple((t, c) for t, c, _ in check_seq)
                        
                        # Verify pattern match and time constraints
                        if (pattern == check_pattern and
                            all((check_seq[k+1][2] - check_seq[k][2]).total_seconds() <= max_time_gap
                                for k in range(len(check_seq)-1))):
                            count += 1
                    
                    if count >= MEMORY_SETTINGS['access_tracking']['patterns']['min_sequence_occurrences']:
                        # Calculate time-based confidence boost
                        avg_time_gap = sum(
                            (current_seq[j+1][2] - current_seq[j][2]).total_seconds()
                            for j in range(len(current_seq)-1)
                        ) / (len(current_seq) - 1)
                        time_factor = max(0.0, 1.0 - (avg_time_gap / max_time_gap))
                        
                        # Calculate sequence complexity factor
                        unique_elements = len(set(pattern))
                        complexity_factor = unique_elements / len(pattern)
                        
                        # Combined confidence score
                        base_confidence = min(1.0, count / MEMORY_SETTINGS['access_tracking']['patterns']['max_sequence_occurrences'])
                        confidence = base_confidence * (1 + time_factor + complexity_factor) / 3
                        
                        # Create a Pattern object with all required attributes
                        pattern_obj = Pattern(
                            sequence=pattern,
                            count=count,
                            confidence=confidence,
                            avg_time_gap=avg_time_gap,
                            complexity=complexity_factor,
                            pattern_type='sequence_patterns',  # Match the pattern_type used in analyze_access_patterns
                            time_factor=time_factor,
                            unique_elements=unique_elements,
                            window_size=window_size
                        )
                        
                        # Only add if it's a significant pattern
                        if count >= MEMORY_SETTINGS['access_tracking']['patterns']['min_sequence_occurrences']:
                            sequences.append(pattern_obj)
                            logger.debug(f"Found sequence pattern: {pattern} with count {count}")
            
            # Filter out subsequences and return the most significant patterns
            return self._filter_subsequences(sequences)
            
        except Exception as e:
            logger.error(f"Error in sequence pattern analysis: {e}")
            return []
    
    def _normalize_sequence(self, seq: Any) -> str:
        """Normalize a sequence for comparison"""
        if isinstance(seq, (tuple, list)):
            return str(tuple(str(x) for x in seq))
        return str(seq)

    def _is_subsequence(self, seq1: Any, seq2: Any) -> bool:
        """Helper method to check if one sequence is contained within another"""
        if not seq1 or not seq2:
            return False
            
        # Normalize sequences for comparison
        norm1 = self._normalize_sequence(seq1)
        norm2 = self._normalize_sequence(seq2)
        
        # Check both directions to handle variable length patterns
        return norm1 in norm2 or norm2 in norm1

    def _filter_subsequences(self, sequences: List[Pattern]) -> List[Pattern]:
        """Filter out subsequences while keeping the most significant patterns"""
        if not sequences:
            return []
            
        # Sort by significance (confidence, count, complexity)
        sequences.sort(key=lambda x: (-x['confidence'], -x['count'], -x['complexity']))
        
        # Keep track of patterns we want to keep
        kept_patterns = []
        for pattern in sequences:
            # Check if this pattern is a subsequence of any higher-ranked pattern
            is_subsequence = any(
                self._is_subsequence(pattern['sequence'], kept['sequence'])
                for kept in kept_patterns
            )
            
            if not is_subsequence:
                kept_patterns.append(pattern)
                logger.debug(f"Keeping pattern: {pattern['sequence']} (confidence: {pattern['confidence']})")
            else:
                logger.debug(f"Filtering out subsequence: {pattern['sequence']}")
        
        return kept_patterns

    def _analyze_periodicities(self, records: List[Dict]) -> List[Dict]:
        """Analyze periodic/time-based access patterns"""
        try:
            # Get settings
            pattern_settings = MEMORY_SETTINGS['access_tracking']['patterns']
            min_occurrences = pattern_settings['min_occurrences']
            
            # Group accesses by hour and weekday
            hour_patterns = {}
            weekday_patterns = {}
            
            for record in records:
                dt = datetime.fromisoformat(record['access_time'])
                hour = dt.strftime('%H')
                weekday = dt.strftime('%A')
                
                # Track hourly patterns
                if hour not in hour_patterns:
                    hour_patterns[hour] = {'count': 0, 'contexts': set()}
                hour_patterns[hour]['count'] += 1
                hour_patterns[hour]['contexts'].add(record['context'])
                
                # Track weekday patterns
                if weekday not in weekday_patterns:
                    weekday_patterns[weekday] = {'count': 0, 'contexts': set()}
                weekday_patterns[weekday]['count'] += 1
                weekday_patterns[weekday]['contexts'].add(record['context'])
            
            patterns = []
            
            # Create patterns for significant periodicities
            for hour, data in hour_patterns.items():
                if data['count'] >= min_occurrences:
                    patterns.append({
                        'type': 'hourly',
                        'hour': hour,
                        'count': data['count'],
                        'contexts': list(data['contexts']),
                        'confidence': min(1.0, data['count'] / pattern_settings.get('max_occurrences', 10))
                    })
            
            for weekday, data in weekday_patterns.items():
                if data['count'] >= min_occurrences:
                    patterns.append({
                        'type': 'weekday',
                        'weekday': weekday,
                        'count': data['count'],
                        'contexts': list(data['contexts']),
                        'confidence': min(1.0, data['count'] / pattern_settings.get('max_occurrences', 10))
                    })
            
            logger.debug(f"Found {len(patterns)} periodic patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error in periodicity analysis: {e}")
            return []

    def _analyze_contexts(self, records: List[Dict]) -> List[Dict]:
        """Analyze for contextual access patterns"""
        contexts = {}
        
        for record in records:
            context = record['context']
            source = record['source']
            
            if context not in contexts:
                contexts[context] = {
                    'count': 0,
                    'sources': set(),
                    'first_seen': record['access_time'],
                    'last_seen': record['access_time']
                }
            
            contexts[context]['count'] += 1
            contexts[context]['sources'].add(source)
            contexts[context]['last_seen'] = record['access_time']

        # Convert to list of significant patterns
        return [{
            'context': context,
            'count': data['count'],
            'source_diversity': len(data['sources']),
            'duration': (datetime.fromisoformat(data['last_seen']) - 
                        datetime.fromisoformat(data['first_seen'])).total_seconds() / 3600,
            'confidence': min(1.0, data['count'] / 
                MEMORY_SETTINGS['access_tracking']['patterns']['min_occurrences'])
        } for context, data in contexts.items()
        if data['count'] >= MEMORY_SETTINGS['access_tracking']['patterns']['min_occurrences']]

    def _store_patterns(self, memory_id: str, patterns: Dict[str, List[Dict]]):
        """Store significant patterns for future reference with type-specific handling"""
        with db_manager.get_cursor() as cursor:
            for pattern_type, pattern_list in patterns.items():
                for pattern in pattern_list:
                    # Skip empty patterns
                    if not pattern:
                        continue

                    try:
                        # Get pattern data based on object type
                        if isinstance(pattern, Pattern):
                            pattern_dict = pattern.to_dict()
                            # Extract the key identifying content based on pattern type
                            if pattern_type == 'sequence_patterns':
                                content = str(pattern_dict['sequence'])
                                logger.debug(f"Sequence pattern content: {content}")
                            elif pattern_type == 'hourly_patterns':
                                content = str(pattern_dict['hour'])
                            elif pattern_type == 'contextual_patterns':
                                content = str(pattern_dict['context'])
                            else:
                                content = str(pattern_dict.get('content', ''))
                            
                            confidence = pattern_dict['confidence']
                        else:
                            pattern_dict = pattern if isinstance(pattern, dict) else {'data': pattern}
                            content = str(pattern_dict.get('sequence', pattern_dict.get('hour', pattern_dict.get('context', ''))))
                            confidence = pattern_dict.get('confidence', 0.0)

                        # Create a deterministic pattern ID
                        pattern_content = {
                            'type': pattern_type,
                            'memory_id': memory_id,
                            'content': content
                        }
                        # Sort the content to ensure consistent IDs
                        pattern_id = str(uuid.uuid5(uuid.NAMESPACE_DNS,
                            json.dumps(pattern_content, sort_keys=True)))

                        # Ensure all required attributes are present
                        pattern_data = {
                            'pattern_type': pattern_type,
                            'confidence': confidence,
                            'count': pattern_dict.get('count', 1)
                        }

                        # Add pattern-specific attributes
                        if pattern_type == 'sequence_patterns':
                            pattern_data.update({
                                'avg_time_gap': pattern_dict.get('avg_time_gap', 0),
                                'complexity': pattern_dict.get('complexity', 0),
                                'time_factor': pattern_dict.get('time_factor', 0),
                                'unique_elements': pattern_dict.get('unique_elements', 0),
                                'window_size': pattern_dict.get('window_size', 0)
                            })
                        
                        # Merge remaining pattern-specific data
                        pattern_data.update(pattern_dict)
                        
                        logger.debug(f"Storing {pattern_type} pattern {pattern_id} with content: {content}")
                        
                        # Store the pattern in the database
                        cursor.execute("""
                            INSERT INTO memory_access_patterns
                            (pattern_id, memory_id, pattern_type, pattern_data, confidence)
                            VALUES (?, ?, ?, ?, ?)
                            ON CONFLICT(pattern_id) DO UPDATE SET
                            pattern_data = ?,
                            confidence = ?,
                            last_seen = CURRENT_TIMESTAMP,
                            occurrence_count = occurrence_count + 1
                        """, (
                            pattern_id,
                            memory_id,
                            pattern_type,
                            json.dumps(pattern_data),
                            confidence,
                            json.dumps(pattern_data),
                            confidence
                        ))
                        
                        if cursor.rowcount > 0:
                            logger.debug(f"Successfully stored/updated pattern {pattern_id}")
                            
                    except Exception as e:
                        logger.error(f"Failed to store pattern: {e}")
                        continue
                    

    def _test_mode_record_access(self, memory_id: str, access_type: str, context: str, 
                               source: str, timestamp: datetime, chain_id: Optional[str] = None):
        """Test helper to record access with specific timestamp"""
        try:
            with db_manager.get_cursor() as cursor:
                # Update last_accessed first
                cursor.execute("""
                    UPDATE memory_items
                    SET last_accessed = ?
                    WHERE id = ?
                """, (timestamp.strftime('%Y-%m-%d %H:%M:%S'), memory_id))
                
                # Then record the access
                cursor.execute("""
                    INSERT INTO memory_access_log
                    (memory_id, access_type, context, source, access_time, chain_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    memory_id,
                    access_type,
                    context,
                    source,
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    chain_id
                ))
                logger.debug(f"Test mode recorded access at: {timestamp}")
        except Exception as e:
            logger.error(f"Failed to record test access: {e}")
            raise

    def clear_access_logs(self, older_than: Optional[datetime] = None):
        """Clear access logs, optionally only those older than specified date"""
        try:
            with db_manager.get_cursor() as cursor:
                if older_than:
                    cursor.execute("""
                        DELETE FROM memory_access_log 
                        WHERE access_time < ?
                    """, (older_than,))
                else:
                    cursor.execute("DELETE FROM memory_access_log")
                
                logger.info(f"Cleared access logs: {cursor.rowcount} records deleted")
        except Exception as e:
            logger.error(f"Failed to clear access logs: {e}")
            raise

    def record_deduplication(self, kept_id: str, removed_id: str, similarity: float, cursor=None):
        """Record when memories are deduplicated"""
        try:
            # Use provided cursor or get a new one
            if cursor:
                cursor.execute("""
                    INSERT INTO memory_access_log
                    (memory_id, access_type, context, source)
                    VALUES (?, 'deduplication', ?, ?)
                """, (
                    kept_id,
                    f"Merged with {removed_id}",
                    f"similarity={similarity:.3f}"
                ))
            else:
                with db_manager.get_cursor() as new_cursor:
                    new_cursor.execute("""
                        INSERT INTO memory_access_log
                        (memory_id, access_type, context, source)
                        VALUES (?, 'deduplication', ?, ?)
                    """, (
                        kept_id,
                        f"Merged with {removed_id}",
                        f"similarity={similarity:.3f}"
                    ))
        except Exception as e:
            logger.error(f"Failed to record deduplication: {e}")
            raise

# Global instance
memory_access_tracker = MemoryAccessTracker() 