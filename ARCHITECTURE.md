# AI Assistant Brain Architecture

## Core Design Principles

1. **Single Source of Truth (SSOT)**
   - Configuration: All settings flow from ui_settings.json
   - Database: All persistence through db_manager
   - Models: All embeddings through model_manager
   - Tags: Approved tags defined in central config

2. **State Management Hierarchy**
   ```
   UI Layer (Tkinter)
   ↓
   Application Layer (Flask)
   ↓
   Processing Layer (Text/Embeddings)
   ↓
   Persistence Layer (SQLite)
   ```

3. **Thread Safety Design**
   - UI runs in main thread
   - Flask runs in daemon thread
   - Database uses connection per thread
   - Model manager ensures thread-safe model access
   - All shared resources use proper locking

4. **Memory Management Strategy**
   ```
   Short-term: Conversation context (last N turns)
   Medium-term: Document chunks with embeddings
   Long-term: Tagged and categorized knowledge base
   ```

5. **Future LLM Integration Points**
   ```
   dummy_retrieval.py → Local LLM
   text_embeddings → Potential vector DB
   prompt_builder → LLM-specific prompting
   ```

## Critical Dependencies

1. **Database Schema Evolution**
   - All schema changes must be versioned
   - Upgrade functions in db_manager
   - Maintain backward compatibility
   - Handle migration failures gracefully

2. **Configuration Hierarchy**
   ```
   Default Config
   ↓
   User Settings (ui_settings.json)
   ↓
   Runtime Overrides
   ↓
   Per-Session Settings
   ```

3. **Error Recovery Strategy**
   ```
   UI Layer: Show user-friendly messages
   App Layer: Log and handle gracefully
   DB Layer: Transaction rollback
   Model Layer: Fallback to defaults
   ```

## Integration Testing Requirements

1. **Database Integrity**
   - Foreign key constraints
   - Transaction boundaries
   - Concurrent access patterns
   - Data consistency checks

2. **Thread Safety Verification**
   - UI responsiveness
   - Background processing
   - Resource contention
   - Deadlock prevention

3. **Memory Management**
   - Model caching behavior
   - Connection pool efficiency
   - Large file handling
   - Resource cleanup

4. **Offline Operation**
   - Model availability
   - Cached embeddings
   - Graceful degradation
   - Error handling

## Future Considerations

1. **Scaling Strategy**
   - Potential migration to vector DB
   - Multiple LLM support
   - Distributed processing
   - Enhanced caching

2. **Security Considerations**
   - Input validation
   - Content filtering
   - Access controls
   - Data privacy

3. **Maintenance Requirements**
   - Regular database cleanup
   - Model updates
   - Configuration backups
   - Log rotation

## For AI Assistants

1. **Code Modification Guidelines**
   - All database operations must use db_manager's context managers
   - Thread safety must be maintained in UI operations
   - Model access must go through model_manager
   - Error handling must follow the established hierarchy
   ```python
   try:
       # Operation-specific code
   except SpecificError as e:
       logging.error(f"Context: {str(e)}")
       # Layer-appropriate error handling
   ```

2. **Critical Integration Points**
   ```
   UI ↔ Flask: Through shared db_manager
   Text Processing ↔ Models: Through model_manager
   Auto-tagging ↔ UI: Through tag_manager
   Memory ↔ Retrieval: Through conversation_logger
   ```

3. **State Management Rules**
   - UI state belongs in desktop_ui.py
   - Database state managed by db_manager
   - Configuration state flows from ui_settings.json
   - Runtime state handled by respective managers

4. **Documentation Requirements**
   - All public methods must have docstrings
   - Complex operations need inline comments
   - Error handling must be documented
   - Thread safety considerations must be noted

5. **Testing Implications**
   - Changes to db_manager require transaction tests
   - UI changes need thread safety verification
   - Model changes must work in offline mode
   - New features need integration tests

6. **Common Pitfalls**
   - Avoid direct database access; use db_manager
   - Don't create new model instances; use model_manager
   - Never store sensitive data in logs
   - Always handle thread safety in UI operations
   - Maintain SSOT principles for all changes

7. **Project Evolution Notes**
   - dummy_retrieval.py will be replaced with real LLM
   - Embedding storage may move to vector database
   - UI may expand to web interface
   - Security features will be enhanced

8. **Code Generation Rules**
   - Follow existing error handling patterns
   - Maintain consistent naming conventions
   - Include type hints for all new code
   - Document thread safety considerations
   - Add appropriate logging statements
   ```python
   # Example pattern for new features
   def new_feature():
       """
       Feature description.
       
       Thread Safety:
       - Note any threading considerations
       
       Error Handling:
       - Specify error handling strategy
       
       State Management:
       - Document state changes
       """
       try:
           with db_manager.get_cursor() as cursor:
               # Implementation
       except Exception as e:
           logging.error(f"Feature failed: {str(e)}")
           # Appropriate error handling
   ``` 