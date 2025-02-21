"""
CONFIGURATION
------------
Central configuration for the AI Assistant Brain Project.

This module serves as the Single Source of Truth (SSOT) for all configuration settings.
Settings are organized by component and include validation ranges where applicable.

Categories:
1. Project Paths
2. Database Settings
3. Memory Management
4. Model Configuration
5. UI Settings
6. Personality Defaults
7. Cleanup & Maintenance
"""

import os
import re
import logging
import pathlib

# ===============================
# Project Paths & Basic Settings
# ===============================
PROJECT_ROOT = pathlib.Path(__file__).parent.absolute()
DB_FILE = PROJECT_ROOT / "AI_Project_Brain" / "files.db"
UPLOAD_FOLDER = PROJECT_ROOT / "uploads"
MODEL_CACHE_DIR = PROJECT_ROOT / "AI_Project_Brain" / "models"

# Test-specific settings
TEST_DB_FILE = PROJECT_ROOT / "AI_Project_Brain" / "test_files.db"

# =================
# Memory Management
# =================
MEMORY_SETTINGS = {
    'max_common_tags': 3,
    'max_conversations': 10,
    'test_conversation_id': 'test_convo',
    'summary_period': 'day',
    'update_existing': True,
    'layers': {
        'working': {
            'max_age': 24 * 60 * 60,  # 24 hours in seconds
            'promotion_threshold': 0.8,  # Set this lower than our test value of 0.9
            'importance_threshold': 0.5,
            'deletion_threshold': 0.2
        },
        'short_term': {
            'max_age': 7 * 24 * 60 * 60,  # 7 days
            'promotion_threshold': 5,
            'importance_threshold': 0.7,
            'demotion_threshold': 0.4   # Below this, memory is demoted to working
        },
        'long_term': {
            'cleanup_age': 30 * 24 * 60 * 60,  # 30 days
            'min_importance': 0.8,
            'allow_permanent': False,
            'permanent_threshold': 0.95,
            'demotion_threshold': 0.6   # Below this, memory is demoted to short_term
        }
    },
    'decay': {
        'check_interval': 3600,  # 1 hour in seconds
        'layers': {
            'working': {
                'rate': 0.1,
                'min_importance': 0.1
            },
            'short_term': {
                'rate': 0.05,
                'min_importance': 0.2
            },
            'long_term': {
                'rate': 0.01,
                'min_importance': 0.3
            }
        },
        'access_boost': 0.2,  # Importance boost on access
        'connection_boost': 0.1,  # Boost for related items
        'promotion_decay_multiplier': 0.8,  # How much slower decay becomes after promotion
        'rate': 0.1,  # Base decay rate per period
        'min_importance': 0.1,  # Minimum importance value
        'protected_period': 24  # Hours before decay starts
    },
    'demotion': {
        'importance_penalty': 0.8,  # Multiply importance by this when demoting
        'access_count_reset': True  # Reset access count on demotion
    },
    'optimization': {
        'merge_threshold': 0.95,  # Similarity threshold for merging
        'consolidation_threshold': 0.8,  # Connection strength for consolidation
        'optimization_threshold': 1000,  # Memory count to trigger optimization
        'min_cluster_size': 3,  # Minimum memories to consolidate
        'compression': {
            'enabled': True,
            'min_size': 1000,  # Minimum content size to compress
            'target_ratio': 0.5  # Target compression ratio
        }
    },
    'metrics': {
        'max_access_norm': 100,  # Normalize access counts against this value
        'max_connection_norm': 20,  # Normalize connection counts against this value
        'access_weight': 0.4,
        'connection_weight': 0.3,
        'importance_weight': 0.3,
        'min_access_threshold': 5,  # Minimum accesses to consider pattern significant
        'temporal_window_hours': 24  # Hours to look back for temporal patterns
    },
    'access_tracking': {
        'enabled': True,
        'max_history_per_memory': 1000,
        'cleanup_interval': 24 * 60 * 60,
        'track_sources': True,
        'min_time_between_records': 1,
        'patterns': {
            'time_window': 7 * 24 * 60 * 60,
            'min_pattern_occurrences': 3,
            'chain_timeout': 300,
            'min_occurrences': 2,
            'min_chain_occurrences': 2,
            'max_chain_length': 10,
            'min_occurrences': 2,
            'sequence_window': 3,  # Length of sequences to look for
            'min_sequence_occurrences': 2,  # How many times a sequence must repeat
            'max_sequence_occurrences': 5,  # For confidence normalization
            'min_periodic_occurrences': 2,  # How many periods to establish pattern
            'max_periodic_occurrences': 7,  # For confidence normalization
        }
    },
    'protection': {
        'min_connection_strength': 0.7,
        'min_strong_connections': 2,
        'min_access_count': 5,
        'importance_threshold': 0.8,
        'recent_access_window': 7 * 24 * 60 * 60  # 7 days in seconds
    },
    'promotion': {
        'threshold': 0.8  # Importance threshold for promotion
    },
    'deduplication': {
        'similarity_threshold': 0.85,  # Minimum similarity to consider duplicates
        'preserve_higher_importance': True,  # Keep the more important memory
        'merge_topics': True,  # Combine topics from both memories
        'min_age_hours': 1,  # Don't deduplicate very recent memories
        'batch_size': 100  # Number of memories to check at once
    },
    'transitions': {
        'connection_strength_threshold': 0.8,  # Minimum strength for related memory transitions
        'min_significant_patterns': 2,         # Number of patterns needed for promotion
        'domain_confidence_threshold': 0.8,    # Domain confidence needed for faster promotion
        'importance_increment': 0.1,           # How much importance increases on promotion
    },
    'connections': {
        'min_strength': 0.1,  # Connections below this are removed
        'initial_strength': 0.5,
        'reinforcement_boost': 0.2,
        'cluster_threshold': 0.7,  # Minimum strength for cluster formation
    },
    'auto_tag': {
        'threshold': 0.157,  # Base threshold for tag suggestions
        'tag_weight': 0.7,   # Weight for tag name matches
        'description_weight': 0.3,  # Weight for description matches
        'max_confidence': 1.0,  # Maximum confidence score
        'min_confidence': 0.0,  # Minimum confidence score
    }
}

# ===================
# Cleanup & Maintenance
# ===================
CLEANUP_SETTINGS = {
    # Database cleanup
    'pattern_age_limit': 30,  # days
    'min_confidence_keep': 0.7,
    'max_patterns_per_type': 1000,
    'max_contexts_stored': 5000,
    
    # Maintenance intervals
    'cleanup_interval': 24 * 60 * 60,  # seconds (daily)
    'optimization_interval': 7 * 24 * 60 * 60,  # seconds (weekly)
    'backup_interval': 24 * 60 * 60,  # seconds (daily)
    
    # Thresholds
    'db_size_warning': 1024 * 1024 * 100,  # 100MB
    'max_memory_usage': 1024 * 1024 * 512,  # 512MB
    
    # Retention policies
    'keep_high_confidence_patterns': True,
    'compress_old_conversations': True,
    'min_pattern_observations': 3
}

# =================
# Database Settings
# =================
DB_SETTINGS = {
    'pool_size': 5,
    'pool_timeout': 1,  # seconds
    'max_connections': 20,
    'connection_timeout': 5,  # seconds
    'busy_timeout': 5000,  # milliseconds
    'journal_mode': 'WAL',
    'synchronous': 'NORMAL',
    'backup': {
        'max_backups': 5,
        'compress_by_default': True,
        'backup_dir': 'backups',
        'backup_on_shutdown': True,
        'min_backup_interval': 24 * 60 * 60,  # 24 hours in seconds
        'verify_restore': True
    }
}

# =================
# Model Settings
# =================
MODEL_SETTINGS = {
    'default_model': os.environ.get('AI_DEFAULT_MODEL', 'all-MiniLM-L6-v2'),
    'embedding_batch_size': 32,
    'max_sequence_length': 512,
    'cache_embeddings': True,
    'offline_mode_allowed': True
}

# =================
# Text Processing
# =================
TEXT_SETTINGS = {
    'chunk_size': 500,
    'chunk_overlap': 100,
    'min_chunk_size': 50,
    'max_chunk_size': 1000,
    'allowed_extensions': {'.txt', '.pdf'},
    'embedding_similarity_threshold': 0.2
}

# =================
# Logging Settings
# =================
LOG_SETTINGS = {
    'format': '%(asctime)s [%(levelname)s] %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'level': os.environ.get('AI_LOG_LEVEL', 'INFO'),
    'file_rotation': '1 day',
    'max_log_files': 7,
    'log_dir': PROJECT_ROOT / "logs"
}

# =================
# UI Settings
# =================
UI_SETTINGS = {
    'window_size': "500x600",
    'colors': {
        'bg': '#2b2b2b',
        'fg': '#ffffff',
        'button': '#404040',
        'entry': '#333333',
        'text': '#333333'
    },
    'font': {
        'family': 'Arial',
        'size': 10,
        'heading_size': 12
    },
    'refresh_rate': 1000,  # milliseconds
    'max_output_length': 10000
}

# =================
# Personality Settings
# =================
PERSONALITY_DEFAULTS = {
    'sentiment': {
        'default': 0.1,
        'positive_threshold': 0.3,
        'negative_threshold': -0.3
    },
    'learning': {
        'default_value': 0.05,
        'min_increment': 0.01
    },
    'formality': {
        'default_level': 0.6,
        'formal_threshold': 0.7
    },
    'state': {
        'openness': 0.7,
        'conscientiousness': 0.8,
        'extraversion': 0.6,
        'agreeableness': 0.75,
        'stability': 0.9,
        'energy': 0.8,
        'formality': 0.6,
        'empathy': 0.7,
        'default_mood': 'neutral',
        'baseline_energy': 0.8
    },
    'evolution': {
        'formality_weight': 0.3,
        'mood_impact': 0.3,
        'learning_rate': 0.05
    },
    'memory': {
        'max_emotional_history': 50,
        'emotion_decay_rate': 0.1,
        'trigger_threshold': 0.3
    }
}

# Approved tags with descriptions
APPROVED_TAGS = {
    "factual": "Accurate and objective content",
    "technical": "Technical details and precise information",
    "creative": "Imaginative and artistic content",
    "historical": "Pertaining to history or past events",
    "scientific": "Related to science and scientific research",
    "opinion": "Personal views or subjective commentary",
    "personal": "Relating to personal experiences or information",
    "fantasy": "Imaginary or fantastical elements",
    "literary": "Pertaining to literature or creative writing",
    "music": "Related to musical content or composition",
    "philosophy": "Philosophical thoughts or ideas",
    "miscellaneous": "Other topics that do not fit into a specific category"
}

# Add to existing config.py
DEFAULT_POOL_SIZE = 5
DEFAULT_POOL_TIMEOUT = 1  # seconds

# Add text processing constants
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 1000

# Add model constants
DEFAULT_MODEL = os.environ.get('AI_DEFAULT_MODEL', 'all-MiniLM-L6-v2')

# Add logging constants
LOG_FORMAT = '%(asctime)s [%(levelname)s] %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = os.environ.get('AI_LOG_LEVEL', 'INFO')

# Add prompt builder constants
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant with local memory that is helpful, direct, and knowledgeable. 
You operate offline with local context only and maintain conversation history and knowledge base."""

DEFAULT_TOKEN_BUDGETS = {
    "system_message": 500,
    "context": 2000,
    "conversation": 1000,
    "user_query": 500
}

DEFAULT_PERSONALITY_TRAITS = {
    "role": "AI assistant with local memory",
    "style": "helpful, direct, and knowledgeable",
    "limitations": "operates offline with local context only",
    "memory": "maintains conversation history and knowledge base"
}

def save_config_tags(new_tags):
    """Update APPROVED_TAGS in config.py with a dictionary of tags."""
    config_path = os.path.abspath(__file__)
    with open(config_path, 'r') as file:
        content = file.read()
    # Format the dictionary as a Python literal.
    tags_str = "{\n    " + ",\n    ".join(f'"{tag}": "{desc}"' for tag, desc in new_tags.items()) + "\n}"
    # Replace the existing APPROVED_TAGS definition.
    pattern = r"APPROVED_TAGS\s*=\s*\{([\s\S]*?)\}"
    new_content = re.sub(pattern, f"APPROVED_TAGS = {tags_str}", content)
    with open(config_path, 'w') as file:
        file.write(new_content)

BACKUP_SETTINGS = {
    'scheduler': {
        'history_limit': 100,  # Number of backup records to keep
        'status_limit': 5,     # Number of recent backups to show in status
        'retry_delay': 300,    # 5 minutes between retry attempts
        'max_retries': 3       # Maximum retry attempts per backup
    },
    'storage': {
        'max_backups': 5,
        'compress_by_default': True,
        'backup_dir': 'backups',
        'backup_on_shutdown': True,
        'min_backup_interval': 24 * 60 * 60,  # 24 hours
        'verify_after_backup': True
    },
    'verification': {
        'required_tables': ['conversation_history', 'personality_state'],
        'min_row_counts': {
            'conversation_history': 0,
            'personality_state': 1
        },
        'integrity_check': True,
        'size_warning_threshold': 1024 * 1024 * 100  # 100MB
    }
}

EXPERTISE_SETTINGS = {
    'confidence': {
        'memory_weight': 0.4,
        'quality_weight': 0.3,
        'time_weight': 0.2,
        'diversity_weight': 0.1,   # Used for source diversity weighting
        'default': 0.0,
        'min': 0.0,
        'max': 1.0
    },
    'clustering': {
        'similarity_threshold': 0.7,
        'min_cluster_size': 2
    },
    'quality': {
        'default': 0.5,
        'min': 0.0,
        'max': 1.0
    },
    'thresholds': {
        'min_memories': 1,
        'min_quality': 0.0,
        'max_quality': 1.0
    },
    'default_limit': 5  # Add this for get_strongest_domains
}
