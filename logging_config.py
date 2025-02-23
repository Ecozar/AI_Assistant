"""
LOGGING CONFIGURATION
--------------------
Central configuration for all logging in the project.
"""

import logging
import logging.handlers  # Add this for TimedRotatingFileHandler
from typing import Optional
from pathlib import Path
from config import LOG_SETTINGS

def configure_logging(
    log_file: Optional[str] = None,
    level: int = LOG_SETTINGS['level'],
    format_str: str = LOG_SETTINGS['format']
) -> None:
    """Configure logging with new settings"""
    # Create logs directory if it doesn't exist
    LOG_SETTINGS['log_dir'].mkdir(parents=True, exist_ok=True)
    
    # Clear existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set up handlers with rotation
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='D',
            interval=1,
            backupCount=LOG_SETTINGS['max_log_files']
        ))
    
    # Configure with new settings
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=LOG_SETTINGS['date_format'],
        handlers=handlers
    ) 