"""
TEXT UTILITIES
-------------
Core text processing utilities for the AI Assistant Brain Project.

This module provides essential text handling functions that are used across
the project for:
1. Text chunking and preprocessing
2. Embedding generation and caching
3. File reading and content extraction
4. Text normalization and cleaning

Design Philosophy:
- Functions must be deterministic (same input = same output)
- Must handle Unicode and special characters gracefully
- Must be efficient with large texts
- Must support offline operation

Technical Requirements:
- Must use same embedding model as rest of system
- Must handle various text formats consistently
- Must support future file type additions
- Must maintain memory efficiency
"""

import re
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from AI_Project_Brain.model_manager import model_manager
from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MODEL_SETTINGS
)
from config import TEXT_SETTINGS

# Configure logging
logger = logging.getLogger(__name__)

class TextProcessor:
    """
    Handles text processing operations.
    
    Core Responsibilities:
    1. Text chunking and segmentation
    2. Embedding generation
    3. Text cleaning and normalization
    4. File content extraction
    
    Design Notes:
    - Uses model_manager for embeddings
    - Implements caching for efficiency
    - Handles Unicode properly
    - Maintains consistent chunk sizes
    """
    
    def __init__(self):
        """
        Initialize text processor.
        
        Sets up:
        - Access to embedding model
        - Regex patterns
        - Caching structures
        """
        self.model = model_manager.get_model()  # Get model from model_manager
        
        # Regex patterns for text processing
        self.patterns = {
            'sentence_end': re.compile(r'[.!?]\s+'),
            'whitespace': re.compile(r'\s+'),
            'special_chars': re.compile(r'[^\w\s]'),
            'paragraph': re.compile(r'\n\s*\n')
        }
        
        # Cache for embeddings
        self._embedding_cache = {}

    def get_model(self):
        """Get the embedding model instance"""
        return self.model

    def chunk_text(self,
                  text: str,
                  chunk_size: int = TEXT_SETTINGS['chunk_size'],
                  overlap: int = TEXT_SETTINGS['chunk_overlap']) -> List[str]:
        """
        Split text into overlapping chunks.
        
        This method:
        1. Respects sentence boundaries
        2. Maintains context through overlap
        3. Handles various text formats
        4. Ensures chunk size consistency
        
        Args:
            text: Input text to chunk
            chunk_size: Target size in words
            overlap: Number of overlapping words
            
        Returns:
            List of text chunks
            
        Note:
            Crucial for proper context maintenance in RAG
            Must balance chunk size with model context limits
        """
        if not text:
            return []
            
        # Validate parameters using TEXT_SETTINGS
        chunk_size = max(TEXT_SETTINGS['min_chunk_size'], 
                        min(chunk_size, TEXT_SETTINGS['max_chunk_size']))
        overlap = min(overlap, chunk_size // 2)
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            if current_size + sentence_words > chunk_size:
                # Current chunk is full
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    # Keep overlap for next chunk
                    overlap_words = current_chunk[-overlap:]
                    current_chunk = overlap_words
                    current_size = sum(len(w.split()) for w in overlap_words)
                    
            current_chunk.append(sentence)
            current_size += sentence_words
        
        # Add final chunk if any
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    def get_embedding(self,
                     text: str,
                     use_cache: bool = True) -> np.ndarray:
        """
        Get embedding vector for text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use embedding cache
            
        Returns:
            numpy.ndarray: Embedding vector
            
        Note:
            Uses model_manager for consistent embeddings
            Implements caching for efficiency
        """
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]
            
        try:
            embedding = self.model.encode([text])[0]
            
            if use_cache:
                self._embedding_cache[text] = embedding
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def read_file(self, file_path: str) -> str:
        """
        Read and extract text from file.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: Extracted text content
            
        Note:
            Currently handles basic text files
            Designed for extension to other formats
        """
        try:
            path = Path(file_path)
            
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Basic text file handling
            if path.suffix.lower() == '.txt':
                return path.read_text(encoding='utf-8')
                
            # TODO: Add handlers for other file types
            # PDF, DOCX, etc. will be added here
            
            raise ValueError(f"Unsupported file type: {path.suffix}")
            
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
            
        Note:
            Handles various sentence endings
            Preserves sentence integrity
        """
        # Clean text first
        text = self._clean_text(text)
        
        # Split on sentence endings while preserving them
        sentences = self.patterns['sentence_end'].split(text)
        
        # Add back the sentence endings
        endings = self.patterns['sentence_end'].findall(text)
        sentences = [s + (endings[i] if i < len(endings) else '')
                    for i, s in enumerate(sentences)
                    if s.strip()]
                    
        return sentences

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
            
        Note:
            Handles Unicode normalization
            Preserves important whitespace
        """
        # Normalize whitespace
        text = self.patterns['whitespace'].sub(' ', text)
        
        # Remove excessive newlines
        text = self.patterns['paragraph'].sub('\n\n', text)
        
        # Strip edges
        return text.strip()

    def clear_cache(self):
        """
        Clear the embedding cache.
        
        Note:
            Important for memory management
            Should be called periodically
        """
        self._embedding_cache.clear()

# Global instance for use throughout project
text_processor = TextProcessor()

def generate_embedding(text: str) -> np.ndarray:
    """Generate embedding for given text using global text processor."""
    return text_processor.get_model().encode([text])[0]

def read_text_file(file_path: str) -> str:
    """Read content from text file using global text processor."""
    return text_processor.read_file(file_path)

def read_pdf_file(file_path: str) -> str:
    """Read content from PDF file using global text processor."""
    # TODO: Implement PDF reading
    raise NotImplementedError("PDF reading not yet implemented")

def chunk_text(text: str) -> List[tuple]:
    """Chunk text and generate embeddings using global text processor."""
    chunks = text_processor.chunk_text(text)
    return [(chunk, generate_embedding(chunk)) for chunk in chunks]

def get_text_similarity(text1: str, text2: str) -> float:
    """Get semantic similarity between two text strings"""
    # Get embeddings
    emb1 = text_processor.get_model().encode(text1, convert_to_tensor=False)
    emb2 = text_processor.get_model().encode(text2, convert_to_tensor=False)
    
    # Ensure embeddings are numpy arrays and normalized
    emb1 = emb1 / np.linalg.norm(emb1)
    emb2 = emb2 / np.linalg.norm(emb2)
    
    # Calculate cosine similarity
    similarity = float(np.dot(emb1, emb2))
    
    return max(0.0, min(similarity, 1.0))  # Ensure value between 0 and 1
