"""MCP Server for SQLite RAG integration"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from mcp.server import MCPServer
from mcp.types import Tool, TextContent

from sqlite_rag.sqliterag import SQLiteRag


class SQLiteRagMCP:
    def __init__(self, db_path: str = "./sqliterag.sqlite"):
        self.db_path = db_path
        self.app = MCPServer("sqlite-rag-server")
        self._setup_tools()

    def _setup_tools(self):
        """Register all available tools"""

        @self.app.tool()
        async def add_directory(
                path: str,
                recursive: bool = True,
                use_relative_paths: bool = False,
                metadata: Optional[str] = None,
                only_extensions: Optional[str] = None,
                exclude_extensions: Optional[str] = None
        ):
            """Add all files from a directory into the database.

            Args:
                path: Directory path to add files from
                recursive: Whether to process subdirectories recursively
                use_relative_paths: Store relative paths instead of absolute paths
                metadata: JSON string of metadata to associate with documents
                only_extensions: Comma-separated list of file extensions to include (e.g. 'py,js,md')
                exclude_extensions: Comma-separated list of file extensions to exclude (e.g. 'pyc,txt')
            """
            try:
                rag = SQLiteRag.create(self.db_path)

                parsed_metadata = json.loads(metadata) if metadata else {}

                parsed_only_extensions = None
                if only_extensions:
                    parsed_only_extensions = [ext.strip() for ext in only_extensions.split(',')]

                parsed_exclude_extensions = None
                if exclude_extensions:
                    parsed_exclude_extensions = [ext.strip() for ext in exclude_extensions.split(',')]

                processed_count = rag.add(
                    path,
                    recursive=recursive,
                    use_relative_paths=use_relative_paths,
                    metadata=parsed_metadata,
                    only_extensions=parsed_only_extensions,
                    exclude_extensions=parsed_exclude_extensions
                )

                return TextContent(
                    type="text",
                    text=f"Successfully processed {processed_count} files from directory: {path}"
                )
            except Exception as e:
                return TextContent(type="text", text=f"Error adding directory: {str(e)}")


        @self.app.tool()
        async def search(query: str, limit: int = 5):
            """Search for documents using hybrid vector + full-text search."""
            try:
                rag = SQLiteRag.create(self.db_path, require_existing=True)
                results = rag.search(query, top_k=limit)

                formatted_results = []
                for result in results:
                    content = f"""
File: {result.uri or 'N/A'}
Score: {result.score:.4f}
Content: {result.content[:500]}{'...' if len(result.content) > 500 else ''}
"""
                    formatted_results.append(TextContent(type="text", text=content))

                return formatted_results
            except Exception as e:
                return [TextContent(type="text", text=f"Error during search: {str(e)}")]

        @self.app.tool()
        async def add_text(text: str, uri: Optional[str] = None, metadata: Optional[str] = None):
            """Add text content to the RAG database."""
            try:
                rag = SQLiteRag.create(self.db_path)
                rag.add_text(
                    text,
                    uri=uri,
                    metadata=json.loads(metadata) if metadata else {}
                )
                return TextContent(
                    type="text",
                    text="Text successfully added to the database."
                )
            except Exception as e:
                return TextContent(type="text", text=f"Error adding text: {str(e)}")

        @self.app.tool()
        async def list_documents():
            """List all documents in the database."""
            try:
                rag = SQLiteRag.create(self.db_path, require_existing=True)
                documents = rag.list_documents()

                if not documents:
                    return [TextContent(type="text", text="No documents found in the database.")]

                formatted_docs = []
                for doc in documents:
                    content = f"""
ID: {doc.id}
URI: {doc.uri or 'N/A'}
Created: {doc.created_at.strftime('%Y-%m-%d %H:%M:%S') if doc.created_at else 'N/A'}
Content Preview: {doc.content[:100]}{'...' if len(doc.content) > 100 else ''}
"""
                    formatted_docs.append(TextContent(type="text", text=content))

                return formatted_docs
            except Exception as e:
                return [TextContent(type="text", text=f"Error listing documents: {str(e)}")]

        @self.app.tool()
        async def remove_document(identifier: str):
            """Remove a document by ID or URI."""
            try:
                rag = SQLiteRag.create(self.db_path, require_existing=True)
                success = rag.remove_document(identifier)

                if success:
                    return TextContent(
                        type="text",
                        text=f"Document {identifier} removed successfully."
                    )
                else:
                    return TextContent(
                        type="text",
                        text=f"Failed to remove document {identifier}."
                    )
            except Exception as e:
                return TextContent(type="text", text=f"Error removing document: {str(e)}")

        @self.app.tool()
        async def get_settings():
            """Get current configuration settings of the RAG system."""
            try:
                rag = SQLiteRag.create(self.db_path, require_existing=True)
                settings = rag.get_settings()

                settings_str = json.dumps(settings, indent=2, default=str)
                return TextContent(
                    type="text",
                    text=f"Current settings:\n{settings_str}"
                )
            except Exception as e:
                return TextContent(type="text", text=f"Error getting settings: {str(e)}")

    async def run(self):
        """Run the MCP server"""
        await self.app.run_stdio_async()


def main():
    """Entry point for the MCP server"""
    db_path = sys.argv[1] if len(sys.argv) > 1 else "./sqliterag.sqlite"

    if not Path(db_path).is_absolute():
        db_path = str(Path.cwd() / db_path)

    server = SQLiteRagMCP(db_path)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()