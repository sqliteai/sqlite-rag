#!/usr/bin/env python3
import json
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
def set(settings: Optional[str] = typer.Argument(None)):
    """Set the model and database path"""
    pass


@app.command()
def add(
    path: str = typer.Argument(..., help="File or directory path to add"),
    recursive: bool = typer.Option(
        False, "-r", "--recursive", help="Recursively add all files in directories"
    ),
    absolute_paths: bool = typer.Option(
        False,
        "--absolute-paths",
        help="Store absolute paths instead of relative paths",
        is_flag=True,
    ),
    metadata: Optional[str] = typer.Option(
        None,
        "--metadata",
        help="Optional metadata in JSON format to associate with the document",
        metavar="JSON",
        show_default=False,
    ),
):
    """Add a file path to the database"""
    rag = SQLiteRag.create()
    rag.add(
        path,
        recursive=recursive,
        absolute_paths=absolute_paths,
        metadata=json.loads(metadata or "{}"),
    )


@app.command()
def add_text(
    text: str,
    uri: Optional[str] = None,
    metadata: Optional[str] = typer.Option(
        None,
        "--metadata",
        help="Optional metadata in JSON format to associate with the document",
        metavar="JSON",
        show_default=False,
        prompt="Metadata (JSON format, e.g. {'author': 'John Doe', 'date': '2023-10-01'}'",
    ),
):
    """Add a text to the database"""
    rag = SQLiteRag.create()
    rag.add_text(text, uri=uri, metadata=json.loads(metadata or "{}"))


@app.command("list")
def list_documents():
    """List all documents in the database"""
    rag = SQLiteRag.create()
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
    rag = SQLiteRag.create()

    # Find the document first
    document = rag.find_document(identifier)
    if not document:
        typer.echo(f"Document not found: {identifier}")
        raise typer.Exit(1)

    # Show document details
    typer.echo("Found document:")
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
    rag = SQLiteRag.create()

    typer.echo("Rebuild process...")

    result = rag.rebuild(remove_missing=remove_missing)

    typer.echo("Rebuild completed:")
    typer.echo(f"  Total documents: {result['total']}")
    typer.echo(f"  Reprocessed: {result['reprocessed']}")
    typer.echo(f"  Not found: {result['not_found']}")
    typer.echo(f"  Removed: {result['removed']}")


@app.command()
def reset(
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt")
):
    """Reset/clear the entire database"""
    rag = SQLiteRag.create()

    # Show warning and ask for confirmation unless -y flag is used
    if not yes:
        typer.echo(
            "WARNING: This will permanently delete all documents and data from the database!"
        )
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
    query: str,
    limit: int = typer.Option(10, help="Number of results to return"),
    debug: bool = typer.Option(
        False, "-d", "--debug", help="Print extra debug information"
    ),
):
    """Search for documents using hybrid vector + full-text search"""
    rag = SQLiteRag.create()
    results = rag.search(query, top_k=limit)

    if not results:
        typer.echo("No documents found matching the query.")
        return

    typer.echo(f"Found {len(results)} documents:")

    if debug:
        # Enhanced debug table with better formatting
        typer.echo(
            f"{'#':<3} {'Preview':<55} {'URI':<35} {'C.Rank':<33} {'V.Rank':<8} {'FTS.Rank':<9} {'V.Dist':<18} {'FTS.Score':<18}"
        )
        typer.echo("─" * 180)

        for idx, doc in enumerate(results, 1):
            # Clean snippet display
            snippet = doc.snippet.replace("\n", " ").replace("\r", "")
            if len(snippet) > 52:
                snippet = snippet[:49] + "..."

            # Clean URI display
            uri = doc.document.uri or "N/A"
            if len(uri) > 32:
                uri = "..." + uri[-29:]

            # Format debug values with proper precision
            c_rank = (
                f"{doc.combined_rank:.17f}" if doc.combined_rank is not None else "N/A"
            )
            v_rank = str(doc.vec_rank) if doc.vec_rank is not None else "N/A"
            fts_rank = str(doc.fts_rank) if doc.fts_rank is not None else "N/A"
            v_dist = (
                f"{doc.vec_distance:.6f}" if doc.vec_distance is not None else "N/A"
            )
            fts_score = f"{doc.fts_score:.6f}" if doc.fts_score is not None else "N/A"

            typer.echo(
                f"{idx:<3} {snippet:<55} {uri:<35} {c_rank:<33} {v_rank:<8} {fts_rank:<9} {v_dist:<18} {fts_score:<18}"
            )
    else:
        # Clean simple table for normal view
        typer.echo(f"{'#':<3} {'Preview':<60} {'URI':<40}")
        typer.echo("─" * 105)

        for idx, doc in enumerate(results, 1):
            # Clean snippet display
            snippet = doc.snippet.replace("\n", " ").replace("\r", "")
            if len(snippet) > 57:
                snippet = snippet[:54] + "..."

            # Clean URI display
            uri = doc.document.uri or "N/A"
            if len(uri) > 37:
                uri = "..." + uri[-34:]

            typer.echo(f"{idx:<3} {snippet:<60} {uri:<40}")


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
