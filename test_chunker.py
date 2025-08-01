#!/usr/bin/env python3

import sqlite3
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from chunker import Chunker
from settings import Settings
from database import Database

def test_chunker():
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    
    # Initialize with small chunk sizes for testing
    settings = Settings("test-model")
    settings.chunk_size = 50  # Small for testing
    settings.chunk_overlap = 10  # Small overlap
    
    # Initialize database with extensions (this might fail without extensions)
    try:
        conn = Database.initialize(conn, settings)
    except RuntimeError as e:
        print(f"Warning: Could not load extensions: {e}")
        print("Testing without extensions - token counting will not work")
        return
    
    # Create chunker
    chunker = Chunker(conn, settings)
    
    # Test with sample text
    test_text = """# Deep Learning Neural Networks

Deep learning utilizes artificial neural networks with multiple layers to process and learn from vast amounts of data. These networks automatically discover intricate patterns and representations without manual feature engineering. 

Convolutional neural networks excel at image recognition tasks, while recurrent neural networks handle sequential data like text and speech. Popular frameworks include TensorFlow, PyTorch, and Keras. 

Deep learning has revolutionized computer vision, natural language processing, and speech recognition applications."""
    
    print("Original text:")
    print(test_text)
    print(f"\nText length: {len(test_text)} characters")
    
    # Test token counting
    try:
        token_count = chunker._get_token_count(test_text)
        print(f"Token count: {token_count}")
    except Exception as e:
        print(f"Token counting failed: {e}")
        return
    
    # Test chunking
    chunks = chunker.chunk(test_text)
    
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Content: {chunk.content}")
        print(f"Length: {len(chunk.content)} characters")
        try:
            tokens = chunker._get_token_count(chunk.content)
            print(f"Tokens: {tokens}")
        except Exception as e:
            print(f"Token counting failed: {e}")

if __name__ == "__main__":
    test_chunker()