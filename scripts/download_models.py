"""
MODEL DOWNLOADER
---------------
Downloads and caches required models for offline use.
"""

import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from sentence_transformers import SentenceTransformer
from AI_Project_Brain.model_manager import DEFAULT_MODEL

def download_models():
    cache_dir = os.path.join(project_root, "AI_Project_Brain", "models")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading and caching default model ({DEFAULT_MODEL})...")
    print(f"Cache directory: {cache_dir}")
    
    model = SentenceTransformer(DEFAULT_MODEL, cache_folder=cache_dir)
    print("Model cached successfully!")

if __name__ == "__main__":
    download_models() 