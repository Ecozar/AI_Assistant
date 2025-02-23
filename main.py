"""
MAIN ENTRY POINT
---------------
Coordinates startup sequence according to SSOT principles.
"""

import logging
from threading import Thread, Timer, Lock
from datetime import datetime, timedelta
import atexit
import signal
import sys
from waitress import serve

from AI_Project_Brain.logging_config import configure_logging
from AI_Project_Brain.db_manager import db_manager
from AI_Project_Brain.model_manager import model_manager
from AI_Project_ui.desktop_ui import start_ui
from AI_Project_Brain.app import app, init_db, upgrade_tags_table
from AI_Project_Brain.memory_summarizer import memory_summarizer
from config import (
    LOG_SETTINGS,
    DB_SETTINGS,
    CLEANUP_SETTINGS,
    MODEL_SETTINGS
)
from AI_Project_Brain.backup_manager import backup_manager
from AI_Project_Brain.memory_optimizer import memory_optimizer
from AI_Project_Brain.memory_manager import memory_manager

# Global cleanup flag
shutdown_in_progress = False

def cleanup_handler(signum=None, frame=None):
    """Handle cleanup on shutdown"""
    global shutdown_in_progress
    if shutdown_in_progress:
        return
    shutdown_in_progress = True
    
    logging.info("Starting cleanup...")
    
    # Stop backup manager
    backup_manager.stop()
    
    # Stop memory decay service
    memory_manager.stop_decay_service()
    
    # Stop memory summarization
    if hasattr(main, 'memory_timer') and main.memory_timer:
        main.memory_timer.cancel()
    
    # Stop memory optimizer
    memory_optimizer.stop()
    
    # Database cleanup
    try:
        if not db_manager._pool.empty():
            db_manager.optimize_database()
        db_manager.close_all_connections()
        logging.info("Database connections closed")
    except Exception as e:
        logging.error(f"Error during database cleanup: {e}")
    
    # Model cleanup
    try:
        model_manager.clear_model()
        logging.info("Model resources cleared")
    except Exception as e:
        logging.error(f"Error during model cleanup: {e}")
    
    logging.info("Cleanup completed")
    sys.exit(0)

def schedule_maintenance():
    """Schedule periodic maintenance tasks"""
    if shutdown_in_progress:
        return
        
    try:
        # Database optimization
        if db_manager.get_database_size() > CLEANUP_SETTINGS['db_size_warning']:
            db_manager.optimize_database()
        
        # Pattern cleanup
        with db_manager.get_cursor() as cursor:
            cursor.execute("""
                DELETE FROM emotional_patterns 
                WHERE last_observed < datetime('now', ?) 
                AND confidence < ?
            """, (
                f"-{CLEANUP_SETTINGS['pattern_age_limit']} days",
                CLEANUP_SETTINGS['min_confidence_keep']
            ))
        
        # Schedule next maintenance
        maintenance_timer = Timer(
            CLEANUP_SETTINGS['cleanup_interval'], 
            schedule_maintenance
        )
        maintenance_timer.daemon = True
        maintenance_timer.start()
        
    except Exception as e:
        logging.error(f"Error during maintenance: {e}")

def schedule_backup():
    """Schedule periodic database backups"""
    if shutdown_in_progress:
        return
        
    try:
        # Create backup
        backup_path = db_manager.create_backup(
            compress=DB_SETTINGS['backup']['compress_by_default']
        )
        logging.info(f"Scheduled backup created: {backup_path}")
        
        # Schedule next backup
        backup_timer = Timer(
            DB_SETTINGS['backup']['min_backup_interval'],
            schedule_backup
        )
        backup_timer.daemon = True
        backup_timer.start()
        
    except Exception as e:
        logging.error(f"Error during scheduled backup: {e}")

def main():
    """Initialize all components in correct order following SSOT"""
    try:
        # 1. Configure logging first
        configure_logging(
            log_file=LOG_SETTINGS['log_dir'] / "app.log",
            level=LOG_SETTINGS['level']
        )
        logging.info("Logging configured")
        
        # 2. Set up signal handlers
        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        atexit.register(cleanup_handler)
        
        # 3. Initialize database
        init_db()
        upgrade_tags_table()
        logging.info("Database initialized")
        
        # 4. Initialize model with new settings
        model = model_manager.get_model(MODEL_SETTINGS['default_model'])
        if model:
            logging.info("Model loaded successfully")
        else:
            logging.error("Failed to load model")
            return
        
        # 5. Schedule maintenance
        schedule_maintenance()
        logging.info("Maintenance scheduled")
        
        # 6. Schedule backups
        if DB_SETTINGS['backup']['backup_on_shutdown']:
            atexit.register(lambda: [
                db_manager.close_connections(),
                db_manager.create_backup(compress=True)
            ])
        schedule_backup()
        logging.info("Database backups scheduled")
        
        # 7. Start Waitress in background (replaces Flask)
        flask_thread = Thread(
            target=lambda: serve(
                app,
                host='127.0.0.1',  # Only allow local connections
                port=5000,
                threads=4,         # Handle concurrent uploads better
                url_scheme='http'  # Or 'https' if adding SSL
            ),
            daemon=True
        )
        flask_thread.start()
        logging.info("Waitress server started")
        
        # 8. Start UI (blocking)
        start_ui()
        
        # Initialize and start backup manager
        backup_manager.start()
        logging.info("Backup manager started")
        
        # Initialize and start memory optimizer
        memory_optimizer.start()
        logging.info("Memory optimizer started")
        
    except Exception as e:
        logging.error(f"Error during startup: {e}")
        cleanup_handler()
        raise

if __name__ == "__main__":
    main()
