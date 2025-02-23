"""
KNOWLEDGE DEPTH ANALYZER
-----------------------
Analyzes the depth and quality of knowledge in different domains.

Measures:
1. Vertical depth (how detailed the knowledge is)
2. Horizontal breadth (how it connects to other concepts)
3. Application capability (can it be used to solve problems)
4. Understanding verification (consistency checks)
"""

import logging
from typing import Dict, List, Set, Tuple, Any
import json
from datetime import datetime
import numpy as np
from collections import defaultdict

from .db_manager import db_manager
from .text_utils import generate_embedding
from config import (
    EXPERTISE_SETTINGS, 
    KNOWLEDGE_DEPTH_SETTINGS,
    DOMAIN_EXPERTISE_SETTINGS
)

class KnowledgeDepthAnalyzer:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        """Configure detailed logging for depth analysis"""
        self.logger = logging.getLogger('knowledge_depth')
        self.logger.setLevel(logging.DEBUG)
        
        # Add detailed formatting
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s\n'
            'Function: %(funcName)s, Line: %(lineno)d'
        )
        
        # Add file handler for depth analysis
        fh = logging.FileHandler('knowledge_depth.log')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        self.logger.info("Knowledge Depth Analyzer initialized")

    def analyze_depth(self, domain_id: str) -> Dict[str, float]:
        """
        Analyze knowledge depth for a domain.
        Returns scores for different depth metrics.
        """
        self.logger.debug(f"Starting depth analysis for domain: {domain_id}")
        try:
            with db_manager.get_cursor() as cursor:
                # Get domain information
                cursor.execute("""
                    SELECT d.topic_cluster, COUNT(dm.memory_id) as memory_count,
                           d.evidence_sources
                    FROM expertise_domains d
                    LEFT JOIN domain_memories dm ON d.domain_id = dm.domain_id
                    WHERE d.domain_id = ?
                    GROUP BY d.domain_id
                """, (domain_id,))
                
                domain_data = cursor.fetchone()
                if not domain_data:
                    self.logger.warning(f"No data found for domain {domain_id}")
                    return self._get_default_scores()
                
                self.logger.debug(f"Domain data: {dict(domain_data)}")
                
                # Calculate different depth metrics
                vertical_depth = self._calculate_vertical_depth(cursor, domain_id)
                horizontal_breadth = self._calculate_horizontal_breadth(cursor, domain_id)
                application_score = self._calculate_application_capability(cursor, domain_id)
                
                # Calculate overall depth using configured weights
                weights = KNOWLEDGE_DEPTH_SETTINGS['weights']
                overall_depth = (
                    vertical_depth * weights['vertical_depth'] +
                    horizontal_breadth * weights['horizontal_breadth'] + 
                    application_score * weights['application']
                )

                scores = {
                    'vertical_depth': vertical_depth,
                    'horizontal_breadth': horizontal_breadth,
                    'application_capability': application_score,
                    'overall_depth': overall_depth
                }
                
                self.logger.info(f"Depth analysis completed for {domain_id}: {scores}")
                return scores
                
        except Exception as e:
            self.logger.error(f"Error in depth analysis: {str(e)}", exc_info=True)
            return self._get_default_scores()

    def _calculate_vertical_depth(self, cursor, domain_id: str) -> float:
        """Calculate vertical depth based on content complexity"""
        cursor.execute("""
            SELECT m.content, m.importance
            FROM memory_items m
            JOIN domain_memories dm ON m.id = dm.memory_id
            WHERE dm.domain_id = ?
        """, (domain_id,))
        
        memories = cursor.fetchall()
        if not memories:
            return KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth']
        
        # Calculate complexity based on content length, structure and importance
        complexities = []
        for memory in memories:
            content = memory['content']
            importance = memory['importance']
            
            # Content length factor
            length_score = min(1.0, len(content.split()) / 50)  # Normalize by 50 words
            
            # Technical term factor
            technical_terms = ['class', 'method', 'function', 'decorator', 'metaclass', 
                             'inheritance', 'polymorphism', 'algorithm']
            term_score = sum(1 for term in technical_terms if term in content.lower()) / len(technical_terms)
            
            # Combine factors
            complexity = (length_score * 0.4 + term_score * 0.3 + importance * 0.3)
            complexities.append(complexity)
        
        avg_complexity = sum(complexities) / len(complexities)
        return min(KNOWLEDGE_DEPTH_SETTINGS['analysis']['max_depth'], avg_complexity)

    def _calculate_horizontal_breadth(self, cursor, domain_id: str) -> float:
        """Calculate horizontal breadth of knowledge"""
        self.logger.debug(f"Calculating horizontal breadth for {domain_id}")
        
        try:
            return self._analyze_connections(cursor, domain_id)
        except Exception as e:
            self.logger.error(f"Error calculating horizontal breadth: {str(e)}", exc_info=True)
            return KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_breadth']

    def _analyze_connections(self, cursor, domain_id: str) -> float:
        """Analyze connection strength and breadth"""
        cursor.execute("""
            SELECT COUNT(*), AVG(strength) 
            FROM memory_connections
            WHERE source_id IN (
                SELECT memory_id FROM domain_memories WHERE domain_id = ?
            )
        """, (domain_id,))
        
        count, avg_strength = cursor.fetchone()
        if not count or not avg_strength:
            return KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_breadth']
            
        # Only consider strong connections based on threshold
        if avg_strength < KNOWLEDGE_DEPTH_SETTINGS['thresholds']['connection']:
            return KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_breadth']
            
        return min(KNOWLEDGE_DEPTH_SETTINGS['analysis']['max_breadth'], 
                  avg_strength)

    def _calculate_application_capability(self, cursor, domain_id: str) -> float:
        """Calculate application capability of knowledge"""
        self.logger.debug(f"Calculating application capability for {domain_id}")
        
        try:
            return self._analyze_application_capability(cursor, domain_id)
        except Exception as e:
            self.logger.error(f"Error calculating application capability: {str(e)}", exc_info=True)
            return KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth']

    def _analyze_application_capability(self, cursor, domain_id: str) -> float:
        """Analyze ability to apply knowledge"""
        cursor.execute("""
            SELECT COUNT(*) 
            FROM memory_items
            WHERE id IN (
                SELECT memory_id FROM domain_memories WHERE domain_id = ?
            )
            AND content LIKE '%example%' OR content LIKE '%application%' OR content LIKE '%use case%'
        """, (domain_id,))
        
        application_count = cursor.fetchone()[0]
        if application_count < KNOWLEDGE_DEPTH_SETTINGS['thresholds']['application']:
            return KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth']
            
        return min(KNOWLEDGE_DEPTH_SETTINGS['analysis']['max_depth'],
                  application_count / 10.0)  # Normalize to 0-1 range

    def _get_default_scores(self) -> Dict[str, float]:
        """Return default scores when analysis fails"""
        return {
            'vertical_depth': KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth'],
            'horizontal_breadth': KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_breadth'],
            'application_capability': KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth'],
            'overall_depth': KNOWLEDGE_DEPTH_SETTINGS['analysis']['min_depth']
        }

# Global instance
knowledge_depth = KnowledgeDepthAnalyzer() 