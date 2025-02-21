from typing import Dict
from .memory_access_tracker import memory_access_tracker
from .memory_connections import memory_connections

def get_memory_metrics(memory_id: str) -> Dict:
    """Get comprehensive metrics for a memory"""
    # Get raw patterns (which will now include aliases)
    raw_patterns = memory_access_tracker.analyze_access_patterns(memory_id)
    
    print("[DEBUG] Original patterns:", raw_patterns)
    
    # Simply pass through all patterns - aliases are already included
    patterns = raw_patterns
    
    print("[DEBUG] After mapping patterns:", patterns)
    
    access_metrics = memory_access_tracker.get_access_metrics(memory_id)
    connection_metrics = get_connection_metrics(memory_id)

    metrics = {
        'access': access_metrics,
        'patterns': patterns,
        'connections': connection_metrics
    }
    
    print("[DEBUG] Final metrics:", metrics)
    
    return metrics

def get_connection_metrics(memory_id: str) -> Dict:
    """Get metrics about memory connections"""
    connections = memory_connections.get_connected_memories(memory_id)
    return {
        'count': len(connections),
        'average_strength': sum(c['strength'] for c in connections) / len(connections) if connections else 0.0
    } 