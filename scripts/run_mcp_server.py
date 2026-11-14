#!/usr/bin/env python3
"""Entry point for the SQLite RAG MCP server"""
import sys
from pathlib import Path

# Add the src directory to the path if needed
script_dir = Path(__file__).parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from sqlite_rag.mcp_server import main

if __name__ == "__main__":
    main()