# AI Assistant Project Overview

## Core Architecture

This is a sophisticated AI system built with several key architectural principles:

### 1. Single Source of Truth (SSOT)
- Configuration: All settings flow from config.py
- Database: Centralized through db_manager
- Models: Managed through model_manager
- Tags: Controlled set defined in central config

### 2. Memory Management System
The system implements a multi-layer memory architecture:

#### Memory Layers
- Working Memory: Active conversation context (24 hour lifetime)
- Short-term Memory: Recently important information (7 day lifetime)
- Long-term Memory: Stable, foundational knowledge (30 day cleanup)
- Permanent Memory: Critical information that doesn't decay

#### Memory Features
- Decay System: Each layer has its own decay rates and thresholds
- Importance Tracking: Memories have importance scores that affect their lifetime
- Connection System: Memories can be linked with weighted relationships
- Auto-tagging: Automatic categorization of memories
- Deduplication: Prevents redundant memories with similarity detection
- Protection: Important memories are protected from cleanup

### 3. Thread Safety Design
- UI runs in main thread
- Flask runs in daemon thread
- Database uses connection pooling
- Model manager ensures thread-safe model access
- All shared resources use proper locking

### 4. Database Management
- Uses SQLite with WAL journal mode
- Connection pooling for thread safety
- Automatic backups with compression
- Database optimization and maintenance
- Transaction management

### 5. Model Integration
- Default model: all-MiniLM-L6-v2
- Embedding generation and caching
- Batch processing capabilities
- Offline mode support
- Configurable sequence lengths

### 6. Text Processing
- Chunking with configurable sizes
- Overlap for context preservation
- Support for multiple file types
- Embedding-based similarity detection

### 7. Personality System
- Tracks multiple personality dimensions
- Adaptive learning capabilities
- Mood and energy management
- Formal/informal style switching
- Empathy and stability tracking

### 8. Maintenance & Cleanup
- Automated database optimization
- Regular memory cleanup
- Backup scheduling
- Pattern cleanup
- Resource usage monitoring

## Key Components

### AI_Project_Brain/
Core functionality including:
- Memory management (multi-layer memory system with decay)
- Conversation handling (logging, retrieval, context management)
- Model management (thread-safe model access, embedding generation)
- Text processing (chunking, embedding generation)
- Auto-tagging (content categorization)
- Backup management (automated backups, compression)
- Memory optimization (deduplication, consolidation)
- Memory metrics tracking (access patterns, importance)

### AI_Project_database/
Database operations including:
- Tag management (CRUD operations for approved tags)
- Auto-tagging (content analysis and tag suggestion)
- Database schema management (versioning, migrations)
- Connection pooling (thread-safe access)
- Transaction management (ACID compliance)

### AI_Project_inference/
Inference capabilities:
- Model inference (currently using dummy_retrieval.py)
- Pattern recognition (temporal and semantic patterns)
- Knowledge application (context-aware responses)
- Future LLM integration points

### AI_Project_ui/
Comprehensive desktop interface with:

1. Core UI Components:
   - Dark theme Tkinter interface
   - Responsive layout management
   - Error handling and user feedback
   - Multi-window support
   - Standardized styling system

2. Chat Interface:
   - Real-time message processing
   - Context-aware responses
   - Auto-tagging support
   - Memory ID tracking
   - Message history display

3. Memory Management:
   - Conversation history viewer
   - Record editing capabilities
   - Bulk tag operations
   - Memory importance tracking
   - Access pattern analysis

4. Document Processing:
   - File upload and processing
   - Text extraction and chunking
   - Automatic tagging
   - Content preview
   - Progress tracking

5. Settings Management:
   - Configuration editing
   - Tag management system
   - System prompt customization
   - Threshold adjustments
   - UI preferences

6. Backup Management:
   - Manual and automated backups
   - Backup verification
   - Restore capabilities
   - Backup rotation
   - Status monitoring

## Configuration System

The system uses a hierarchical configuration:
1. Default Config (in config.py)
2. User Settings (ui_settings.json)
3. Runtime Overrides
4. Per-Session Settings

## Future Considerations

1. Scaling Strategy:
- Potential migration to vector DB
- Multiple LLM support
- Distributed processing
- Enhanced caching

2. Security:
- Input validation
- Content filtering
- Access controls
- Data privacy

3. Maintenance:
- Regular database cleanup
- Model updates
- Configuration backups
- Log rotation

## Development Guidelines

1. Code Modification:
- Use db_manager's context managers
- Maintain thread safety
- Access models through model_manager
- Follow error handling hierarchy

2. Integration Points:
- UI ↔ Flask: Through shared db_manager
- Text Processing ↔ Models: Through model_manager
- Auto-tagging ↔ UI: Through tag_manager
- Memory ↔ Retrieval: Through conversation_logger

3. Testing Requirements:
- Database integrity tests
- Thread safety verification
- Memory management tests
- Offline operation testing