"""
PERSONALITY STATE TRACKER
------------------------
Manages the assistant's personality development and state over time.

Core Responsibilities:
1. Track personality traits and their evolution
2. Maintain emotional state and conversation tone
3. Develop consistent behavioral patterns
4. Integrate with memory system for context-aware responses

Design Philosophy:
- Personality must evolve naturally based on interactions
- State changes must be gradual and consistent
- Must maintain core personality traits while allowing growth
- Must integrate with existing memory and tag systems
"""

import logging
from typing import Dict, List, Optional, Tuple, NamedTuple, Any
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field
import threading

from .db_manager import db_manager
from config import DEFAULT_PERSONALITY_TRAITS, PERSONALITY_DEFAULTS
from AI_Project_Brain.auto_tagger import get_suggested_tags

@dataclass
class PersonalityState:
    """Represents the current personality state"""
    # Core personality dimensions (0-1 scale)
    openness: float = PERSONALITY_DEFAULTS['state']['openness']
    conscientiousness: float = PERSONALITY_DEFAULTS['state']['conscientiousness']
    extraversion: float = PERSONALITY_DEFAULTS['state']['extraversion']
    agreeableness: float = PERSONALITY_DEFAULTS['state']['agreeableness']
    stability: float = PERSONALITY_DEFAULTS['state']['stability']
    
    # Current emotional state
    mood: str = PERSONALITY_DEFAULTS['state']['default_mood']
    energy: float = PERSONALITY_DEFAULTS['state']['energy']
    
    # Interaction style
    formality: float = PERSONALITY_DEFAULTS['state']['formality']
    empathy: float = PERSONALITY_DEFAULTS['state']['empathy']
    
    # Learning and adaptation
    knowledge_confidence: Dict[str, float] = None

    def __post_init__(self):
        if self.knowledge_confidence is None:
            self.knowledge_confidence = {}

class EmotionalMemory(NamedTuple):
    """Represents a single emotional memory"""
    mood: str
    trigger: str
    intensity: float
    timestamp: datetime

@dataclass
class EmotionalAssociation:
    """Tracks emotional associations with topics over time"""
    topic: str
    impact: float
    valence: float  # Positive/negative association
    confidence: float
    last_updated: datetime = field(default_factory=datetime.now)
    trigger_history: List[str] = field(default_factory=list)

@dataclass
class EmotionalContext:
    """Rich emotional context with temporal awareness"""
    primary_emotion: str
    secondary_emotions: List[str]
    intensity: float
    triggers: Dict[str, float]  # trigger -> impact strength
    associated_topics: Dict[str, EmotionalAssociation]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def merge_with(self, other: 'EmotionalContext', decay_factor: float = 0.9) -> 'EmotionalContext':
        """Merge two emotional contexts with temporal decay"""
        time_diff = (self.timestamp - other.timestamp).total_seconds()
        decay = decay_factor ** (time_diff / 3600)  # Hourly decay
        
        # Merge triggers with decay
        merged_triggers = self.triggers.copy()
        for trigger, impact in other.triggers.items():
            if trigger in merged_triggers:
                merged_triggers[trigger] = max(impact * decay, merged_triggers[trigger])
            else:
                merged_triggers[trigger] = impact * decay
                
        return EmotionalContext(
            primary_emotion=self.primary_emotion,
            secondary_emotions=list(set(self.secondary_emotions + other.secondary_emotions)),
            intensity=max(self.intensity, other.intensity * decay),
            triggers=merged_triggers,
            associated_topics=self._merge_associations(other.associated_topics, decay),
            timestamp=self.timestamp
        )

    def _merge_associations(self, 
                          other_associations: Dict[str, EmotionalAssociation],
                          decay: float) -> Dict[str, EmotionalAssociation]:
        """Merge topic associations with temporal decay"""
        merged = self.associated_topics.copy()
        for topic, assoc in other_associations.items():
            if topic in merged:
                current = merged[topic]
                merged[topic] = EmotionalAssociation(
                    topic=topic,
                    impact=max(current.impact, assoc.impact * decay),
                    valence=(current.valence + assoc.valence * decay) / 2,
                    confidence=max(current.confidence, assoc.confidence * decay),
                    last_updated=max(current.last_updated, assoc.last_updated),
                    trigger_history=list(set(current.trigger_history + assoc.trigger_history))
                )
            else:
                merged[topic] = assoc
        return merged

class PersonalityTracker:
    """
    Manages and evolves the assistant's personality state.
    Implements Singleton pattern to ensure SSOT.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._state = PersonalityState()
            self._lock = threading.Lock()
            self._init_database()
            self._init_emotional_memory()
            self._load_state()
            self._state_is_fresh = True  # Add flag to track state freshness
            self.__class__._initialized = True
    
    def _init_database(self):
        """Initialize database tables for personality tracking"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_state (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT NOT NULL,
                    impact_scores TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add new table for emotional contexts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotional_contexts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    primary_emotion TEXT NOT NULL,
                    secondary_emotions TEXT NOT NULL,  -- JSON array
                    intensity FLOAT NOT NULL,
                    triggers TEXT NOT NULL,  -- JSON dict
                    associated_topics TEXT NOT NULL,  -- JSON dict
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add table for emotional patterns
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotional_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,  -- 'transition', 'topic_association', etc.
                    pattern_data TEXT NOT NULL,  -- JSON
                    confidence FLOAT NOT NULL,
                    observation_count INTEGER DEFAULT 1,
                    last_observed DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _init_emotional_memory(self):
        """Initialize emotional memory storage"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS emotional_memory (
                    id INTEGER PRIMARY KEY,
                    mood TEXT NOT NULL,
                    trigger TEXT,
                    intensity FLOAT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _load_state(self):
        """Load the most recent personality state"""
        try:
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT state_data FROM personality_state 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                result = cursor.fetchone()
                
                if result:
                    state_data = json.loads(result[0])
                    print(f"DEBUG - Loading state data: {state_data}")
                    self._state = PersonalityState(**state_data)
                    print(f"DEBUG - Loaded state with mood: {self._state.mood}")
                    self._state_is_fresh = True
                    logging.info("Loaded existing personality state")
                else:
                    self._save_state()
                    logging.info("Initialized new personality state")
                    
        except Exception as e:
            logging.error(f"Error loading personality state: {e}")
    
    def _save_state(self):
        """Save current personality state"""
        try:
            state_data = {
                'openness': self._state.openness,
                'conscientiousness': self._state.conscientiousness,
                'extraversion': self._state.extraversion,
                'agreeableness': self._state.agreeableness,
                'stability': self._state.stability,
                'mood': self._state.mood,
                'energy': self._state.energy,
                'formality': self._state.formality,
                'empathy': self._state.empathy,
                'knowledge_confidence': self._state.knowledge_confidence
            }
            
            print(f"DEBUG - Saving state with mood: {self._state.mood}")  # Debug
            
            with db_manager.get_cursor() as cursor:
                cursor.execute(
                    "INSERT INTO personality_state (state_data) VALUES (?)",
                    (json.dumps(state_data),)
                )
                
        except Exception as e:
            logging.error(f"Error saving personality state: {e}")
    
    def update_state(self, 
                    interaction_data: Dict[str, Any],
                    conversation_tags: List[str]) -> None:
        """
        Update personality state based on interaction.
        
        Args:
            interaction_data: Details about the interaction
            conversation_tags: Tags from the conversation
        """
        with self._lock:
            # Update knowledge confidence
            for tag in conversation_tags:
                current = self._state.knowledge_confidence.get(tag, 0.5)
                self._state.knowledge_confidence[tag] = min(1.0, current + 0.01)
            
            # Update mood based on interaction
            sentiment = interaction_data.get('sentiment', 0)
            self._update_mood(sentiment)
            
            # Gradual personality evolution
            self._evolve_personality(interaction_data)
            
            # Save the updated state and mark as fresh
            self._save_state()
            self._state_is_fresh = True
    
    def _update_mood(self, sentiment: float):
        """Update mood based on interaction sentiment"""
        # More responsive energy changes
        current_energy = self._state.energy
        mood_impact = PERSONALITY_DEFAULTS['evolution']['mood_impact']
        
        if sentiment > PERSONALITY_DEFAULTS['sentiment']['positive_threshold']:
            self._state.mood = "positive"
            self._state.energy = min(1.0, current_energy + mood_impact)
        elif sentiment < PERSONALITY_DEFAULTS['sentiment']['negative_threshold']:
            self._state.mood = "concerned"
            self._state.energy = max(0.4, current_energy - mood_impact)
        else:
            self._state.mood = PERSONALITY_DEFAULTS['state']['default_mood']
            self._state.energy = PERSONALITY_DEFAULTS['state']['baseline_energy']
    
    def _evolve_personality(self, interaction_data: Dict):
        """Gradually evolve personality traits based on interactions"""
        # Get evolution settings
        formality_weight = PERSONALITY_DEFAULTS['evolution']['formality_weight']
        old_weight = 1 - formality_weight
        
        # More responsive personality changes
        learning_opportunity = interaction_data.get('learning_value', 0)
        self._state.openness = min(1.0, self._state.openness + 
                                 learning_opportunity * PERSONALITY_DEFAULTS['evolution']['learning_rate'])
        
        # More responsive formality adjustment
        formality_cue = interaction_data.get('formality_level', 
                                           PERSONALITY_DEFAULTS['formality']['default_level'])
        self._state.formality = (old_weight * self._state.formality + 
                                formality_weight * formality_cue)
    
    def get_current_state(self) -> Dict:
        """Get current personality state for response generation"""
        with self._lock:
            # Only reload if state isn't fresh
            if not self._state_is_fresh:
                self._load_state()
            
            # Add debug logging
            logging.debug(f"Current personality state - Mood: {self._state.mood}, Energy: {self._state.energy:.2f}")
            
            return {
                'mood': self._state.mood,
                'energy': self._state.energy,
                'formality': self._state.formality,
                'knowledge_confidence': self._state.knowledge_confidence.copy()
            }
    
    def get_state_summary(self) -> str:
        """Get a human-readable summary of current personality state"""
        with self._lock:
            return (
                f"Current State:\n"
                f"Mood: {self._state.mood}\n"
                f"Energy: {self._state.energy:.2f}\n"
                f"Formality: {self._state.formality:.2f}\n"
                f"Top knowledge areas: {sorted(self._state.knowledge_confidence.items(), key=lambda x: x[1], reverse=True)[:3]}"
            )

    def export_state(self, filepath: str) -> None:
        """Export current personality state to file"""
        try:
            state_data = {
                'core_traits': {
                    'openness': self._state.openness,
                    'conscientiousness': self._state.conscientiousness,
                    'extraversion': self._state.extraversion,
                    'agreeableness': self._state.agreeableness,
                    'stability': self._state.stability
                },
                'current_state': {
                    'mood': self._state.mood,
                    'energy': self._state.energy,
                    'formality': self._state.formality,
                    'empathy': self._state.empathy
                },
                'knowledge': self._state.knowledge_confidence,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"Error exporting personality state: {e}")
            raise

    def import_state(self, filepath: str) -> None:
        """Import personality state from file"""
        try:
            with open(filepath, 'r') as f:
                state_data = json.load(f)
                
            # Validate state data
            required_keys = {'core_traits', 'current_state', 'knowledge'}
            if not all(k in state_data for k in required_keys):
                raise ValueError("Invalid state file format")
                
            # Create new state with imported data
            self._state = PersonalityState(
                openness=state_data['core_traits']['openness'],
                conscientiousness=state_data['core_traits']['conscientiousness'],
                extraversion=state_data['core_traits']['extraversion'],
                agreeableness=state_data['core_traits']['agreeableness'],
                stability=state_data['core_traits']['stability'],
                mood=state_data['current_state']['mood'],
                energy=state_data['current_state']['energy'],
                formality=state_data['current_state']['formality'],
                empathy=state_data['current_state']['empathy']
            )
            self._state.knowledge_confidence = state_data['knowledge']
            
            # Save imported state
            self._save_state()
            
        except Exception as e:
            logging.error(f"Error importing personality state: {e}")
            raise

    def reset_state(self):
        """Reset state to defaults (primarily for testing)"""
        with self._lock:
            self._state = PersonalityState()
            self._save_state()
            self._state_is_fresh = True
            self._initialized = True

    def record_emotional_event(self, mood: str, trigger: str, intensity: float):
        """Record a significant emotional event"""
        with self._lock:
            if intensity >= PERSONALITY_DEFAULTS['memory']['trigger_threshold']:
                with db_manager.get_cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO emotional_memory (mood, trigger, intensity)
                        VALUES (?, ?, ?)
                    """, (mood, trigger, intensity))
                    
                logging.debug(f"Recorded emotional event: {mood} ({intensity:.2f}) due to {trigger}")

    def get_recent_emotions(self, hours: int = 24) -> List[EmotionalMemory]:
        """Get emotional memories from recent history"""
        with self._lock:
            with db_manager.get_cursor() as cursor:
                cursor.execute("""
                    SELECT mood, trigger, intensity, timestamp
                    FROM emotional_memory
                    WHERE timestamp > datetime('now', ?)
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (f'-{hours} hours', PERSONALITY_DEFAULTS['memory']['max_emotional_history']))
                
                return [EmotionalMemory(
                    mood=row[0],
                    trigger=row[1],
                    intensity=row[2],
                    timestamp=datetime.fromisoformat(row[3])
                ) for row in cursor.fetchall()]

    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get a summary of recent emotional state"""
        recent_emotions = self.get_recent_emotions()
        if not recent_emotions:
            return {
                'dominant_mood': 'neutral',
                'emotional_stability': 1.0,
                'recent_triggers': []
            }

        # Calculate emotional statistics
        mood_counts = {}
        triggers = []
        total_intensity = 0

        for emotion in recent_emotions:
            mood_counts[emotion.mood] = mood_counts.get(emotion.mood, 0) + 1
            if emotion.trigger not in triggers:
                triggers.append(emotion.trigger)
            total_intensity += emotion.intensity

        dominant_mood = max(mood_counts.items(), key=lambda x: x[1])[0]
        stability = 1.0 - (len(mood_counts) / len(recent_emotions))

        return {
            'dominant_mood': dominant_mood,
            'emotional_stability': stability,
            'recent_triggers': triggers[:3],  # Top 3 triggers
            'average_intensity': total_intensity / len(recent_emotions)
        }

    def update_emotional_context(self, 
                               interaction_data: Dict,
                               conversation_tags: List[str]) -> None:
        """Update emotional context based on new interaction"""
        with self._lock:
            try:
                # Create new emotional context
                new_context = EmotionalContext(
                    primary_emotion=self._state.mood,
                    secondary_emotions=self._derive_secondary_emotions(interaction_data),
                    intensity=interaction_data.get('sentiment', 0.5),
                    triggers=self._extract_triggers(interaction_data),
                    associated_topics={
                        tag: EmotionalAssociation(
                            topic=tag,
                            impact=self._calculate_topic_impact(tag, interaction_data),
                            valence=interaction_data.get('sentiment', 0),
                            confidence=self._calculate_topic_confidence(tag),
                            trigger_history=[interaction_data.get('trigger', 'general')]
                        )
                        for tag in conversation_tags
                    },
                    timestamp=datetime.now()
                )
                
                # Store context
                self._store_emotional_context(new_context)
                
                # Analyze patterns with error handling
                self._analyze_patterns_safely(new_context)
                
            except Exception as e:
                logging.error(f"Error updating emotional context: {e}")
                # Ensure basic state update still occurs
                if hasattr(self, '_state'):
                    self._update_core_traits(EmotionalContext(
                        primary_emotion=self._state.mood,
                        secondary_emotions=[],
                        intensity=0.5,
                        triggers={},
                        associated_topics={},
                        timestamp=datetime.now()
                    ), {})

    def _derive_secondary_emotions(self, interaction_data: Dict) -> List[str]:
        """Derive secondary emotions based on interaction data"""
        secondary = []
        sentiment = interaction_data.get('sentiment', 0)
        learning_value = interaction_data.get('learning_value', 0)
        
        if learning_value > 0.5:
            secondary.append('curiosity')
        if abs(sentiment) > 0.7:
            secondary.append('intensity')
        if interaction_data.get('formality_level', 0) > 0.8:
            secondary.append('respect')
            
        return secondary

    def _extract_triggers(self, interaction_data: Dict) -> Dict[str, float]:
        """Extract emotional triggers with their impact strengths"""
        triggers = {}
        
        # Direct triggers from interaction
        if 'trigger' in interaction_data:
            triggers[interaction_data['trigger']] = interaction_data.get('sentiment', 0.5)
            
        # Topic-based triggers
        for topic, confidence in self._state.knowledge_confidence.items():
            if confidence > 0.7:  # High confidence topics can be triggers
                triggers[f"topic_{topic}"] = confidence * 0.5
                
        return triggers

    def _calculate_topic_impact(self, tag: str, interaction_data: Dict) -> float:
        """Calculate the emotional impact of a topic"""
        base_impact = abs(interaction_data.get('sentiment', 0))
        knowledge_confidence = self._state.knowledge_confidence.get(tag, 0.5)
        return min(1.0, base_impact * (1 + knowledge_confidence))

    def _calculate_topic_confidence(self, tag: str) -> float:
        """Calculate confidence in topic-emotion association"""
        return self._state.knowledge_confidence.get(tag, 0.5)

    def _store_emotional_context(self, context: EmotionalContext) -> None:
        """Store emotional context in database"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                INSERT INTO emotional_contexts 
                (primary_emotion, secondary_emotions, intensity, triggers, 
                 associated_topics, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                context.primary_emotion,
                json.dumps(context.secondary_emotions),
                context.intensity,
                json.dumps(context.triggers),
                json.dumps({
                    topic: {
                        'impact': assoc.impact,
                        'valence': assoc.valence,
                        'confidence': assoc.confidence,
                        'trigger_history': assoc.trigger_history
                    }
                    for topic, assoc in context.associated_topics.items()
                }),
                context.timestamp
            ))

    def _analyze_patterns_safely(self, new_context: EmotionalContext) -> None:
        """Wrapper for pattern analysis with error handling"""
        try:
            self._analyze_emotional_patterns(new_context)
        except Exception as e:
            logging.error(f"Error in pattern analysis: {e}")
            # Ensure state still gets updated even if pattern analysis fails
            self._update_state_from_context(new_context)

    def _get_recent_contexts(self, hours: int = 24) -> List[EmotionalContext]:
        """Retrieve recent emotional contexts from database"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT primary_emotion, secondary_emotions, intensity, 
                       triggers, associated_topics, timestamp
                FROM emotional_contexts
                WHERE timestamp > datetime('now', ?)
                ORDER BY timestamp DESC
            """, (f'-{hours} hours',))
            
            return [EmotionalContext(
                primary_emotion=row[0],
                secondary_emotions=json.loads(row[1]),
                intensity=row[2],
                triggers=json.loads(row[3]),
                associated_topics=self._deserialize_associations(json.loads(row[4])),
                timestamp=datetime.fromisoformat(row[5])
            ) for row in cursor.fetchall()]

    def _deserialize_associations(self, assoc_data: Dict) -> Dict[str, EmotionalAssociation]:
        """Deserialize emotional associations from JSON storage"""
        associations = {}
        for topic, data in assoc_data.items():
            associations[topic] = EmotionalAssociation(
                topic=topic,
                impact=data['impact'],
                valence=data['valence'],
                confidence=data['confidence'],
                last_updated=datetime.fromisoformat(data['last_updated']) 
                    if 'last_updated' in data 
                    else datetime.now(),
                trigger_history=data.get('trigger_history', [])
            )
        return associations

    def _analyze_emotional_transitions(self, contexts: List[EmotionalContext]) -> Dict:
        """Analyze patterns in emotional state transitions"""
        transitions = {}
        for i in range(len(contexts) - 1):
            current, next_context = contexts[i], contexts[i + 1]
            transition_key = f"{current.primary_emotion}->{next_context.primary_emotion}"
            
            if transition_key not in transitions:
                transitions[transition_key] = {
                    'count': 0,
                    'avg_intensity': 0,
                    'triggers': set(),
                    'confidence': 0
                }
                
            t = transitions[transition_key]
            t['count'] += 1
            t['avg_intensity'] = (t['avg_intensity'] * (t['count'] - 1) + 
                                next_context.intensity) / t['count']
            t['triggers'].update(next_context.triggers.keys())
            t['confidence'] = min(1.0, t['count'] / 10)  # Confidence grows with observations
        
        return transitions

    def _analyze_topic_correlations(self, contexts: List[EmotionalContext]) -> Dict:
        """Analyze correlations between topics and emotional states"""
        correlations = {}
        for context in contexts:
            for topic, assoc in context.associated_topics.items():
                if topic not in correlations:
                    correlations[topic] = {
                        'emotions': {},
                        'total_occurrences': 0,
                        'avg_impact': 0
                    }
                
                c = correlations[topic]
                c['total_occurrences'] += 1
                c['avg_impact'] = (c['avg_impact'] * (c['total_occurrences'] - 1) + 
                                 assoc.impact) / c['total_occurrences']
                
                if context.primary_emotion not in c['emotions']:
                    c['emotions'][context.primary_emotion] = 0
                c['emotions'][context.primary_emotion] += 1
        
        return correlations

    def _analyze_trigger_chains(self, contexts: List[EmotionalContext], 
                              new_context: EmotionalContext) -> Dict:
        """Analyze chains of triggers that lead to emotional states"""
        chains = {}
        window_size = 3  # Look for trigger chains up to 3 steps
        
        for i in range(len(contexts) - window_size + 1):
            window = contexts[i:i + window_size]
            chain = []
            
            for context in window:
                # Get strongest trigger
                if context.triggers:
                    strongest = max(context.triggers.items(), key=lambda x: x[1])
                    chain.append(strongest[0])
            
            if len(chain) == window_size:
                chain_key = '->'.join(chain)
                if chain_key not in chains:
                    chains[chain_key] = {
                        'count': 0,
                        'resulting_emotions': {},
                        'avg_intensity': 0
                    }
                
                c = chains[chain_key]
                c['count'] += 1
                if window[-1].primary_emotion not in c['resulting_emotions']:
                    c['resulting_emotions'][window[-1].primary_emotion] = 0
                c['resulting_emotions'][window[-1].primary_emotion] += 1
                c['avg_intensity'] = (c['avg_intensity'] * (c['count'] - 1) + 
                                    window[-1].intensity) / c['count']
        
        return chains

    def _analyze_emotional_cycles(self, contexts: List[EmotionalContext]) -> Dict:
        """
        Analyze cyclical patterns in emotional states.
        Detects recurring emotional sequences and their timing patterns.
        """
        cycles = {}
        min_cycle_length = 2
        max_cycle_length = 5  # Look for cycles up to 5 emotions long
        
        # Extract emotion sequence
        emotion_sequence = [ctx.primary_emotion for ctx in contexts]
        
        # Look for cycles of different lengths
        for cycle_length in range(min_cycle_length, max_cycle_length + 1):
            for start_idx in range(len(contexts) - cycle_length):
                # Get potential cycle
                cycle = emotion_sequence[start_idx:start_idx + cycle_length]
                cycle_key = '->'.join(cycle)
                
                # Look for repetitions of this cycle
                repetitions = []
                current_idx = start_idx
                while current_idx < len(emotion_sequence) - cycle_length:
                    next_sequence = emotion_sequence[current_idx:current_idx + cycle_length]
                    if next_sequence == cycle:
                        # Found a repetition
                        cycle_instance = {
                            'start_time': contexts[current_idx].timestamp,
                            'end_time': contexts[current_idx + cycle_length - 1].timestamp,
                            'avg_intensity': sum(ctx.intensity for ctx in 
                                contexts[current_idx:current_idx + cycle_length]) / cycle_length
                        }
                        repetitions.append(cycle_instance)
                        current_idx += cycle_length
                    else:
                        current_idx += 1
                
                # If we found repetitions, record the cycle
                if len(repetitions) > 1:
                    if cycle_key not in cycles:
                        cycles[cycle_key] = {
                            'length': cycle_length,
                            'occurrences': [],
                            'avg_duration': timedelta(0),
                            'stability': 0.0,
                            'confidence': 0.0
                        }
                    
                    # Update cycle statistics
                    c = cycles[cycle_key]
                    c['occurrences'].extend(repetitions)
                    
                    # Calculate average duration between repetitions
                    durations = [
                        (rep2['start_time'] - rep1['end_time'])
                        for rep1, rep2 in zip(repetitions[:-1], repetitions[1:])
                    ]
                    if durations:
                        c['avg_duration'] = sum(durations, timedelta(0)) / len(durations)
                    
                    # Calculate cycle stability (consistency of timing)
                    if len(durations) > 1:
                        avg_duration = sum(d.total_seconds() for d in durations) / len(durations)
                        variance = sum((d.total_seconds() - avg_duration) ** 2 
                                     for d in durations) / len(durations)
                        c['stability'] = 1.0 / (1.0 + (variance / avg_duration))
                    
                    # Update confidence based on number of repetitions and stability
                    c['confidence'] = min(1.0, len(repetitions) * c['stability'] / 5)
        
        return cycles

    def _store_patterns(self, patterns: Dict) -> None:
        """Store analyzed emotional patterns in database"""
        with db_manager.get_cursor() as cursor:
            for pattern_type, pattern_data in patterns.items():
                # Convert sets to lists for JSON serialization
                serializable_data = self._prepare_pattern_for_storage(pattern_data)
                
                # Calculate overall confidence for pattern type
                confidence = self._calculate_pattern_confidence(pattern_type, pattern_data)
                
                cursor.execute("""
                    INSERT INTO emotional_patterns 
                    (pattern_type, pattern_data, confidence, observation_count)
                    VALUES (?, ?, ?, ?)
                """, (
                    pattern_type,
                    json.dumps(serializable_data),
                    confidence,
                    len(pattern_data) if isinstance(pattern_data, dict) else 1
                ))

    def _prepare_pattern_for_storage(self, pattern_data: Dict) -> Dict:
        """Prepare pattern data for JSON serialization"""
        if isinstance(pattern_data, dict):
            result = {}
            for key, value in pattern_data.items():
                if isinstance(value, (set, frozenset)):
                    result[key] = list(value)
                elif isinstance(value, dict):
                    result[key] = self._prepare_pattern_for_storage(value)
                elif isinstance(value, timedelta):
                    result[key] = value.total_seconds()
                else:
                    result[key] = value
            return result
        return pattern_data

    def _calculate_pattern_confidence(self, pattern_type: str, pattern_data: Dict) -> float:
        """Calculate overall confidence score for a pattern type"""
        if pattern_type == 'emotional_transitions':
            return sum(t['confidence'] for t in pattern_data.values()) / len(pattern_data)
        elif pattern_type == 'emotional_cycles':
            return max((c['confidence'] for c in pattern_data.values()), default=0.0)
        elif pattern_type == 'topic_correlations':
            return sum(t['total_occurrences'] for t in pattern_data.values()) / 100
        elif pattern_type == 'trigger_chains':
            return sum(c['count'] for c in pattern_data.values()) / 50
        return 0.0

    def _update_state_from_context(self, new_context: EmotionalContext) -> None:
        """
        Update personality state based on emotional context analysis.
        Uses pattern recognition to make informed state adjustments.
        """
        with self._lock:
            # Get recent patterns for informed decisions
            recent_patterns = self._get_recent_patterns()
            
            # Update core personality traits based on emotional patterns
            self._update_core_traits(new_context, recent_patterns)
            
            # Update interaction style
            self._update_interaction_style(new_context, recent_patterns)
            
            # Update knowledge confidence
            self._update_knowledge_confidence(new_context)
            
            # Mark state as needing save
            self._state_is_fresh = False
            self._save_state()

    def _get_recent_patterns(self) -> Dict[str, Any]:
        """Retrieve recent emotional patterns with high confidence"""
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                SELECT pattern_type, pattern_data, confidence
                FROM emotional_patterns
                WHERE confidence > 0.5
                AND last_observed > datetime('now', '-1 day')
                ORDER BY confidence DESC
            """)
            
            patterns = {
                'transitions': {},
                'cycles': {},
                'correlations': {},
                'chains': {}
            }
            
            for row in cursor.fetchall():
                pattern_type, pattern_data, confidence = row
                pattern_data = json.loads(pattern_data)
                patterns[pattern_type] = {
                    'data': pattern_data,
                    'confidence': confidence
                }
            
            return patterns

    def _update_core_traits(self, context: EmotionalContext, patterns: Dict) -> None:
        """Update core personality traits based on emotional context and patterns"""
        # Update openness based on emotional variety
        if len(context.secondary_emotions) > 2:
            self._state.openness = min(1.0, self._state.openness + 0.01)
        
        # Update stability based on emotional patterns
        if 'cycles' in patterns and patterns['cycles']['confidence'] > 0.7:
            # High confidence cycles indicate predictable emotional patterns
            self._state.stability = min(1.0, self._state.stability + 0.02)
        
        # Update extraversion based on emotional intensity
        if context.intensity > 0.7:
            self._state.extraversion = min(1.0, self._state.extraversion + 0.01)
        elif context.intensity < 0.3:
            self._state.extraversion = max(0.0, self._state.extraversion - 0.01)
        
        # Update agreeableness based on positive interactions
        if any(assoc.valence > 0.7 for assoc in context.associated_topics.values()):
            self._state.agreeableness = min(1.0, self._state.agreeableness + 0.01)
        
        # Update conscientiousness based on pattern recognition
        if patterns.get('trigger_chains', {}).get('confidence', 0) > 0.8:
            self._state.conscientiousness = min(1.0, self._state.conscientiousness + 0.02)

    def _update_interaction_style(self, context: EmotionalContext, patterns: Dict) -> None:
        """Update interaction style based on emotional context and patterns"""
        # Update formality based on topic confidence
        avg_confidence = sum(assoc.confidence for assoc in context.associated_topics.values())
        if context.associated_topics:
            avg_confidence /= len(context.associated_topics)
            if avg_confidence > 0.8:
                # High confidence leads to slightly more formal interactions
                self._state.formality = min(1.0, self._state.formality + 0.01)
        
        # Update empathy based on emotional understanding
        if patterns.get('transitions', {}).get('confidence', 0) > 0.7:
            # Good understanding of emotional transitions increases empathy
            self._state.empathy = min(1.0, self._state.empathy + 0.02)
        
        # Update energy based on emotional intensity and stability
        target_energy = context.intensity
        if patterns.get('cycles', {}).get('confidence', 0) > 0.6:
            # Stable cycles moderate energy changes
            target_energy = (self._state.energy + target_energy) / 2
        
        self._state.energy = target_energy

    def _update_knowledge_confidence(self, context: EmotionalContext) -> None:
        """Update knowledge confidence based on emotional associations"""
        for topic, assoc in context.associated_topics.items():
            current_confidence = self._state.knowledge_confidence.get(topic, 0.5)
            
            # Confidence increases more with positive emotional associations
            confidence_change = 0.01 * (1 + max(0, assoc.valence))
            
            # Higher impact interactions have more effect on confidence
            confidence_change *= assoc.impact
            
            # Update confidence with limits
            self._state.knowledge_confidence[topic] = min(1.0, 
                current_confidence + confidence_change)

    def _analyze_emotional_patterns(self, new_context: EmotionalContext) -> None:
        """Analyze and store emotional patterns from context"""
        with self._lock:
            # Get recent contexts for pattern analysis
            recent_contexts = self._get_recent_contexts(hours=24)
            
            # Analyze different pattern types
            patterns = {
                'emotional_transitions': self._analyze_emotional_transitions(recent_contexts),
                'topic_correlations': self._analyze_topic_correlations(recent_contexts),
                'trigger_chains': self._analyze_trigger_chains(recent_contexts, new_context),
                'emotional_cycles': self._analyze_emotional_cycles(recent_contexts)
            }
            
            # Store new patterns with confidence scores
            self._store_patterns(patterns)

# Global instance
personality_tracker = PersonalityTracker() 