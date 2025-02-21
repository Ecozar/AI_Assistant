"""
BACKUP MANAGER
-------------
Central management of database backups following SSOT principles.
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import threading
from typing import Optional, Dict, List

from .db_manager import db_manager
from config import BACKUP_SETTINGS

class BackupManager:
    """
    Manages all backup operations following SSOT principles.
    
    Core Responsibilities:
    1. Scheduling backups
    2. Verifying backups
    3. Managing backup history
    4. Providing backup status
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, test_mode: bool = False):
        """Initialize backup manager"""
        if not self._initialized:
            self._timer: Optional[threading.Timer] = None
            self._lock = threading.Lock()
            self._last_backup: Optional[datetime] = None
            self._backup_history: List[Dict] = []
            self._retry_count = 0
            self._test_mode = test_mode  # Add test mode flag
            self.__class__._initialized = True
    
    def start(self):
        """Start the backup manager"""
        with self._lock:
            self._schedule_next_backup()
            logging.info("Backup manager started")
    
    def stop(self):
        """Stop the backup manager"""
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            # Only do shutdown backup if not in test mode
            if not self._test_mode and BACKUP_SETTINGS['storage']['backup_on_shutdown']:
                self._perform_backup()
            logging.info("Backup manager stopped")
    
    def verify_backup(self, backup_path: Path) -> Dict:
        """Verify backup against configuration requirements"""
        verification = db_manager.verify_backup(backup_path)
        
        if not verification['is_valid']:
            return verification
            
        # Check required tables
        missing_tables = set(BACKUP_SETTINGS['verification']['required_tables']) - set(verification['row_counts'].keys())
        if missing_tables:
            return {
                'is_valid': False,
                'error': f"Missing required tables: {missing_tables}"
            }
            
        # Check minimum row counts
        for table, min_count in BACKUP_SETTINGS['verification']['min_row_counts'].items():
            if verification['row_counts'].get(table, 0) < min_count:
                return {
                    'is_valid': False,
                    'error': f"Table {table} has insufficient rows"
                }
                
        # Check size
        if verification['size'] > BACKUP_SETTINGS['verification']['size_warning_threshold']:
            logging.warning(f"Backup size ({verification['size']} bytes) exceeds threshold")
            
        return verification
    
    def get_backup_status(self) -> Dict:
        """Get current backup status following SSOT"""
        with self._lock:  # Use consistent context manager pattern
            last_backup = self._last_backup
            backup_count = len(self._backup_history)
            recent_backups = self._backup_history[-BACKUP_SETTINGS['scheduler']['status_limit']:] if self._backup_history else []
        
        # Process data outside lock
        return {
            'last_backup': last_backup,
            'next_backup': (last_backup + 
                timedelta(seconds=BACKUP_SETTINGS['storage']['min_backup_interval'])
                if last_backup else None),
            'backup_count': backup_count,
            'recent_backups': recent_backups
        }
    
    def _schedule_next_backup(self):
        """Schedule next backup following configuration"""
        if self._timer:
            self._timer.cancel()
        
        if not self._test_mode:  # Only schedule if not in test mode
            interval = BACKUP_SETTINGS['storage']['min_backup_interval']
            self._timer = threading.Timer(interval, self._perform_backup)
            self._timer.daemon = True
            self._timer.start()
    
    def _perform_backup(self):
        """Perform backup with retry logic"""
        try:
            logging.debug("Starting backup process")
            backup_path = db_manager.create_backup(
                compress=BACKUP_SETTINGS['storage']['compress_by_default']
            )
            logging.debug(f"Backup created at: {backup_path}")
            
            verification = None
            if BACKUP_SETTINGS['storage']['verify_after_backup']:
                logging.debug("Verifying backup")
                verification = self.verify_backup(backup_path)
                if not verification['is_valid']:
                    raise ValueError(f"Backup verification failed: {verification.get('error')}")
                logging.debug("Backup verified successfully")
            
            # Create backup info outside lock
            backup_info = {
                'path': backup_path,
                'timestamp': datetime.now(),
                'verification': verification
            }
            
            # Single update with all information
            self._update_backup_history(backup_info)
            
            self._retry_count = 0
            logging.info(f"Backup completed successfully: {backup_path}")
            
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            if not self._test_mode and self._retry_count < BACKUP_SETTINGS['scheduler']['max_retries']:
                self._retry_count += 1
                logging.debug(f"Scheduling retry {self._retry_count}")
                self._timer = threading.Timer(
                    BACKUP_SETTINGS['scheduler']['retry_delay'], 
                    self._perform_backup
                )
                self._timer.daemon = True
                self._timer.start()
                return
            raise  # Re-raise in test mode
        
        finally:
            # Only schedule next backup if not in test mode and no retries pending
            if not self._test_mode and self._retry_count == 0:
                self._schedule_next_backup()
    
    def _update_backup_history(self, backup_info: Dict):
        """Update backup history following configuration limits"""
        with self._lock:  # Use consistent context manager pattern
            self._backup_history.append(backup_info)
            while len(self._backup_history) > BACKUP_SETTINGS['scheduler']['history_limit']:
                self._backup_history.pop(0)
            self._last_backup = backup_info['timestamp']  # Update last_backup inside the same lock
        logging.debug("Backup history updated")

# Global instance following SSOT
backup_manager = BackupManager() 