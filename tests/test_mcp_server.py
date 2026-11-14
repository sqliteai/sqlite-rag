"""Tests for the MCP server"""
import pytest
import asyncio
from pathlib import Path
import tempfile
import os


def test_mcp_server_import():
    """Test that the MCP server can be imported"""
    try:
        from sqlite_rag.mcp_server import SQLiteRagMCP, main
        assert SQLiteRagMCP is not None
        assert main is not None
    except ImportError as e:
        pytest.skip(f"MCP dependencies not installed: {e}")


def test_mcp_server_creation():
    """Test that the MCP server can be created"""
    try:
        from sqlite_rag.mcp_server import SQLiteRagMCP
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            server = SQLiteRagMCP(db_path)
            assert server.db_path == db_path
            assert server.app is not None
    except ImportError:
        pytest.skip("MCP dependencies not installed")


def test_mcp_server_tools():
    """Test that the MCP server has the expected tools"""
    try:
        from sqlite_rag.mcp_server import SQLiteRagMCP
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            server = SQLiteRagMCP(db_path)
            
            # Check that tools are registered
            # The tools should be available in the server.app
            assert hasattr(server.app, 'tools')
    except ImportError:
        pytest.skip("MCP dependencies not installed")
