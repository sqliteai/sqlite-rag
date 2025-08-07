#!/usr/bin/env python3
import shlex
import sys
from typing import Optional

import typer

from .sqliterag import SQLiteRag


class CLI:
    """Main class to handle CLI commands"""

    def __init__(self, app: typer.Typer):
        self.app = app

    def __call__(self, *args, **kwds):
        if len(sys.argv) == 1:
            repl_mode()
        else:
            self.app()


app = typer.Typer()
cli = CLI(app)


@app.command()
def add(
    path: str = typer.Argument(..., help="File or directory path to add"),
    recursive: bool = typer.Option(
        False, "-r", "--recursive", help="Recursively add all files in directories"
    ),
):
    """Add a file path to the database"""
    rag = SQLiteRag()
    rag.add(path, recursive=recursive)


@app.command()
def add_text(text: str, uri: Optional[str] = None):
    """Add a text to the database"""
    rag = SQLiteRag()
    rag.add_text(text, uri=uri, metadata={})


@app.command("list")
def list_documents():
    """List all documents in the database"""
    rag = SQLiteRag()
    documents = rag.list_documents()

    if not documents:
        typer.echo("No documents found in the database.")
        return

    # Print stats
    typer.echo(f"Total documents: {len(documents)}")

    typer.echo(f"{'ID':<36} {'URI/Content':<50} {'Created At':<20}")
    typer.echo("-" * 106)

    for doc in documents:
        # Show URI if available, otherwise show first chars of content
        uri_or_content = doc.uri or (
            doc.content[:47] + "..." if len(doc.content) > 47 else doc.content
        )
        created_at = (
            doc.created_at.strftime("%Y-%m-%d %H:%M:%S") if doc.created_at else "N/A"
        )
        typer.echo(f"{doc.id:<36} {uri_or_content:<50} {created_at:<20}")


@app.command()
def remove(
    identifier: str,
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt"),
):
    """Remove document by path or UUID"""
    rag = SQLiteRag()

    # Find the document first
    document = rag.find_document(identifier)
    if not document:
        typer.echo(f"Document not found: {identifier}")
        raise typer.Exit(1)

    # Show document details
    typer.echo(f"Found document:")
    typer.echo(f"ID: {document.id}")
    typer.echo(f"URI: {document.uri or 'N/A'}")
    typer.echo(
        f"Created: {document.created_at.strftime('%Y-%m-%d %H:%M:%S') if document.created_at else 'N/A'}"
    )
    typer.echo(
        f"Content preview: {document.content[:200]}{'...' if len(document.content) > 200 else ''}"
    )
    typer.echo()

    # Ask for confirmation unless -y flag is used
    if not yes:
        confirm = typer.confirm("Are you sure you want to delete this document?")
        if not confirm:
            typer.echo("Cancelled.")
            return

    # Remove the document
    success = rag.remove_document(identifier)
    if success:
        typer.echo("Document removed successfully.")
    else:
        typer.echo("Failed to remove document.")
        raise typer.Exit(1)


@app.command()
def rebuild(
    remove_missing: bool = typer.Option(
        False, "--remove-missing", help="Remove documents whose files are not found"
    )
):
    """Rebuild embeddings and full-text index"""
    rag = SQLiteRag()

    typer.echo("Starting rebuild process...")

    result = rag.rebuild(remove_missing=remove_missing)

    typer.echo(f"Rebuild completed:")
    typer.echo(f"  Total documents: {result['total']}")
    typer.echo(f"  Reprocessed: {result['reprocessed']}")
    typer.echo(f"  Not found: {result['not_found']}")
    typer.echo(f"  Removed: {result['removed']}")


@app.command()
def reset(
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt")
):
    """Reset/clear the entire database"""
    rag = SQLiteRag()

    # Show warning and ask for confirmation unless -y flag is used
    if not yes:
        typer.echo(
            "WARNING: This will permanently delete all documents and data from the database!"
        )
        typer.echo(f"Database file: {rag.settings.db_path}")
        typer.echo()
        confirm = typer.confirm("Are you sure you want to reset the entire database?")
        if not confirm:
            typer.echo("Reset cancelled.")
            return

    typer.echo("Resetting database...")

    success = rag.reset()

    if success:
        typer.echo("Database reset completed successfully.")
    else:
        typer.echo("Failed to reset database.")
        raise typer.Exit(1)


@app.command()
def search(
    query: str, limit: int = typer.Option(10, help="Number of results to return")
):
    """Search for documents using hybrid vector + full-text search"""
    rag = SQLiteRag()
    results = rag.search(query, top_k=limit)

    if not results:
        typer.echo("No documents found matching the query.")
        return

    typer.echo(f"Found {len(results)} documents:")
    # print the position, the snippet and the uri as a table
    typer.echo(f"{'Position':<10} {'Content':<50}")
    typer.echo("-" * 60)
    for i, doc in enumerate(results, start=1):
        uri_or_content = doc.uri or (
            doc.content[:47] + "..." if len(doc.content) > 47 else doc.content
        )
        typer.echo(f"{i:<10} {uri_or_content:<50}")


def repl_mode():
    """Interactive REPL mode"""
    typer.echo("Entering interactive mode. Type 'help' for commands or 'exit' to quit.")

    while True:
        try:
            command = input("\nsqlite-rag> ").strip()
            if not command:
                continue

            if command == "exit":
                break
            elif command == "help":
                try:
                    app(["--help"], standalone_mode=False)
                except SystemExit:
                    pass

                typer.echo("Additional commands in this interactive mode:")
                typer.echo("  help  - Show this help")
                typer.echo("  exit  - Exit REPL")
            else:
                try:
                    # Parse command and delegate to typer app
                    args = shlex.split(command)
                    app(args, standalone_mode=False)
                except SystemExit:
                    # Typer raises SystemExit on errors, catch it to stay in REPL
                    pass
                except Exception as e:
                    typer.echo(f"Error: {e}")

        except KeyboardInterrupt:
            typer.echo("\nExiting...")
            break
        except EOFError:
            break


if __name__ == "__main__":
    cli()
