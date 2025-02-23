"""
MEMORY LAYERS
------------
Defines memory layer structure and items.
"""

import sys
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
from enum import Enum

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

class MemoryLayer(Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    PERMANENT = "permanent"  # Could add this for non-decaying memories

@dataclass
class MemoryItem:
    content: str
    layer: MemoryLayer
    created_at: datetime
    last_accessed: datetime
    access_count: int
    importance: float
    topics: List[str]
    related_items: List[str]  # IDs of related memories 