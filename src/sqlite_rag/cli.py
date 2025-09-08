#!/usr/bin/env python3
import json
import shlex
import sys
from pathlib import Path
from typing import Optional

import typer

from sqlite_rag.database import Database
from sqlite_rag.settings import SettingsManager

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


@app.command("settings")
def show_settings():
    """Show current settings"""
    rag = SQLiteRag.create()
    current_settings = rag.get_settings()

    typer.echo("Current settings:")
    for key, value in current_settings.items():
        typer.echo(f"  {key}: {value}")


@app.command("set")
def set_settings(
    model_path_or_name: Optional[str] = typer.Option(
        None, help="Path to the embedding model file or Hugging Face model name"
    ),
    model_config: Optional[str] = typer.Option(
        None, help="Model configuration parameters"
    ),
    embedding_dim: Optional[int] = typer.Option(
        None, help="Dimension of the embedding vectors"
    ),
    vector_type: Optional[str] = typer.Option(
        None, help="Vector storage type (FLOAT16, FLOAT32, etc.)"
    ),
    other_vector_options: Optional[str] = typer.Option(
        None, help="Additional vector configuration"
    ),
    chunk_size: Optional[int] = typer.Option(
        None, help="Size of text chunks for processing"
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None, help="Token overlap between consecutive chunks"
    ),
    quantize_scan: Optional[bool] = typer.Option(
        None, help="Whether to quantize vector for faster search"
    ),
    quantize_preload: Optional[bool] = typer.Option(
        None, help="Whether to preload quantized vectors in memory for faster search"
    ),
    weight_fts: Optional[float] = typer.Option(
        None, help="Weight for full-text search results"
    ),
    weight_vec: Optional[float] = typer.Option(
        None, help="Weight for vector search results"
    ),
):
    """Change default settings for the RAG system.

    Update model configuration, embedding parameters, chunking settings,
    and search weights. Only specify the options you want to change.
    Use 'sqlite-rag settings' to view current values.
    """
    # Build updates dict from all provided parameters
    updates = {
        "model_path_or_name": model_path_or_name,
        "model_config": model_config,
        "embedding_dim": embedding_dim,
        "vector_type": vector_type,
        "other_vector_options": other_vector_options,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "quantize_scan": quantize_scan,
        "quantize_preload": quantize_preload,
        "weight_fts": weight_fts,
        "weight_vec": weight_vec,
    }

    # Filter out None values (unset options)
    updates = {k: v for k, v in updates.items() if v is not None}

    if not updates:
        typer.echo("No settings provided to update.")
        show_settings()
        return

    conn = Database.new_connection()
    settings_manager = SettingsManager(conn)
    settings_manager.prepare_settings(updates)

    show_settings()
    typer.echo("Settings updated.")


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
    ),
    metadata: Optional[str] = typer.Option(
        None,
        "--metadata",
        help="Optional metadata in JSON format to associate with the document",
        metavar="JSON",
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


@app.command("download-model")
def download_model(
    model_id: str = typer.Argument(
        ..., help="Hugging Face model ID (e.g., Qwen/Qwen3-Embedding-0.6B-GGUF)"
    ),
    gguf_file: str = typer.Argument(
        ..., help="GGUF filename to download (e.g., Qwen3-Embedding-0.6B-Q8_0.gguf)"
    ),
    local_dir: str = typer.Option(
        "./models", "--local-dir", "-d", help="Local directory to download to"
    ),
    revision: str = typer.Option(
        "main", "--revision", "-r", help="Model revision/branch to download from"
    ),
):
    """Download a GGUF model file from Hugging Face"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        typer.echo(
            "Error: huggingface_hub not found. Install with: pip install huggingface_hub"
        )
        raise typer.Exit(1)

    # Create local directory structure
    local_path = Path(local_dir) / model_id
    local_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Downloading {gguf_file} from {model_id}...")

    try:
        # Download the specific GGUF file
        downloaded_path = hf_hub_download(
            repo_id=model_id,
            filename=gguf_file,
            local_dir=str(local_path),
            revision=revision,
        )

        final_path = Path(downloaded_path)
        typer.echo(f"Downloaded to: {final_path}")

        if final_path.exists():
            typer.echo(f"File size: {final_path.stat().st_size / (1024*1024):.1f} MB")

    except Exception as e:
        typer.echo(f"Error downloading model: {e}")
        raise typer.Exit(1)


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
