"""
BACKUP SCHEDULER
--------------
Manages scheduled database backups and verification.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import threading
from typing import Optional, Dict, List

from .db_manager import db_manager
from config import DB_SETTINGS

class BackupScheduler:
    def __init__(self):
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()
        self._last_backup: Optional[datetime] = None
        self._backup_history: List[Dict] = []
        
    def start(self):
        """Start the backup scheduler"""
        with self._lock:
            self._schedule_next_backup()
            logging.info("Backup scheduler started")
    
    def stop(self):
        """Stop the backup scheduler"""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            logging.info("Backup scheduler stopped")
    
    def _schedule_next_backup(self):
        """Schedule the next backup"""
        if self._timer:
            self._timer.cancel()
        
        # Calculate next backup time
        interval = DB_SETTINGS['backup']['min_backup_interval']
        self._timer = threading.Timer(interval, self._perform_backup)
        self._timer.daemon = True
        self._timer.start()
    
    def _perform_backup(self):
        """Perform backup and verification"""
        try:
            # Create backup
            backup_path = db_manager.create_backup(
                compress=DB_SETTINGS['backup']['compress_by_default']
            )
            
            # Verify backup
            verification = db_manager.verify_backup(backup_path)
            
            # Store backup info
            backup_info = {
                'path': backup_path,
                'timestamp': datetime.now(),
                'verification': verification
            }
            self._backup_history.append(backup_info)
            
            # Trim history
            while len(self._backup_history) > 100:  # Keep last 100 entries
                self._backup_history.pop(0)
            
            self._last_backup = datetime.now()
            logging.info(f"Scheduled backup completed: {backup_path}")
            
        except Exception as e:
            logging.error(f"Scheduled backup failed: {e}")
        
        finally:
            # Schedule next backup
            self._schedule_next_backup()
    
    def get_backup_status(self) -> Dict:
        """Get current backup status"""
        with self._lock:
            return {
                'last_backup': self._last_backup,
                'next_backup': (self._last_backup + 
                    timedelta(seconds=DB_SETTINGS['backup']['min_backup_interval'])
                    if self._last_backup else None),
                'backup_count': len(self._backup_history),
                'recent_backups': self._backup_history[-5:]  # Last 5 backups
            }

# Global instance
backup_scheduler = BackupScheduler() 