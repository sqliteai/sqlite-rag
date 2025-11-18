#!/usr/bin/env python3
import itertools
import json
import os
import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import typer
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

from sqlite_rag.database import Database
from sqlite_rag.settings import SettingsManager

from .cli_configure import (
    build_configure_signature,
    filter_setting_updates,
)
from .formatters import get_formatter
from .sqliterag import SQLiteRag

DEFAULT_DATABASE_PATH = "./sqliterag.sqlite"


class RAGContext:
    """Manage CLI state and RAG object reuse"""

    def __init__(self):
        self.rag: Optional[SQLiteRag] = None
        self.in_repl = False
        self.database_path: str = ""

    def enter_repl(self):
        """Enter REPL mode"""
        self.in_repl = True

    def get_rag(self, require_existing: bool = False) -> SQLiteRag:
        """Create or reuse SQLiteRag instance"""
        if not self.database_path:
            raise ValueError("Database path not set. Use --database option.")

        if self.in_repl:
            if self.rag is None:
                self.rag = SQLiteRag.create(
                    self.database_path, require_existing=require_existing
                )
            return self.rag
        else:
            # Regular mode - create new instance
            typer.echo(f"Database: {Path(self.database_path).resolve()}")
            return SQLiteRag.create(
                self.database_path, require_existing=require_existing
            )


rag_context = RAGContext()


class CLI:
    """Main class to handle CLI commands"""

    def __init__(self, app: typer.Typer):
        self.app = app

    def __call__(self, *args, **kwds):
        self.app()


app = typer.Typer()
cli = CLI(app)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    database: str = typer.Option(
        DEFAULT_DATABASE_PATH,
        "--database",
        "-db",
        help="Path to the SQLite database file",
    ),
):
    """SQLite RAG - Retrieval Augmented Generation with SQLite"""
    ctx.ensure_object(dict)
    ctx.obj["rag_context"] = rag_context

    if not rag_context.in_repl:
        rag_context.database_path = database

    # If no subcommand was invoked, enter REPL mode
    if ctx.invoked_subcommand is None and not rag_context.in_repl:
        rag_context.enter_repl()
        typer.echo(f"Database: {Path(database).resolve()}")

        repl_mode()


@app.command("settings")
def show_settings(ctx: typer.Context):
    """Show current settings"""
    rag_context = ctx.obj["rag_context"]
    try:
        rag = rag_context.get_rag(require_existing=True)
    except FileNotFoundError:
        typer.echo("Database not found. No settings available.")
        raise typer.Exit(1)

    current_settings = rag.get_settings()

    typer.echo("Current settings:")
    for key, value in current_settings.items():
        typer.echo(f"  {key}: {value}")


@app.command("configure")
def configure_settings(
    ctx: typer.Context,
    force: bool = False,
    **settings_values: Any,
):
    """Configure settings for the RAG system.

    Update model configuration, embedding parameters, chunking settings,
    and search weights. Only specify the options you want to change.
    Use 'sqlite-rag settings' to view current values.
    """
    rag_context = ctx.obj["rag_context"]

    updates = filter_setting_updates(settings_values)

    if not updates:
        typer.echo("No settings provided to configure.")
        show_settings(ctx)
        return

    conn = Database.new_connection(rag_context.database_path)
    settings_manager = SettingsManager(conn)
    settings_manager.configure(updates, force=force)

    show_settings(ctx)
    typer.echo("Settings updated.")


configure_settings.__signature__ = build_configure_signature()


@app.command()
def add(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="File or directory path to add"),
    recursive: bool = typer.Option(
        False, "-r", "--recursive", help="Recursively add all files in directories"
    ),
    use_relative_paths: bool = typer.Option(
        False,
        "--relative-paths",
        help="Store relative paths instead of absolute paths",
    ),
    metadata: Optional[str] = typer.Option(
        None,
        "--metadata",
        help="Optional metadata in JSON format to associate with the document",
        metavar="JSON",
    ),
    only_extensions: Optional[str] = typer.Option(
        None,
        "--only",
        help="Only process these file extensions from supported list (comma-separated, e.g. 'py,js')",
    ),
    exclude_extensions: Optional[str] = typer.Option(
        None,
        "--exclude",
        help="File extensions to exclude (comma-separated, e.g. 'py,js')",
    ),
):
    """Add a file path to the database"""
    rag_context = ctx.obj["rag_context"]
    start_time = time.time()

    only_list = (
        [e.strip().lstrip(".").lower() for e in only_extensions.split(",") if e.strip()]
        if only_extensions
        else None
    )
    exclude_list = (
        [
            e.strip().lstrip(".").lower()
            for e in exclude_extensions.split(",")
            if e.strip()
        ]
        if exclude_extensions
        else None
    )

    rag = rag_context.get_rag()
    rag.add(
        path,
        recursive=recursive,
        use_relative_paths=use_relative_paths,
        metadata=json.loads(metadata or "{}"),
        only_extensions=only_list,
        exclude_extensions=exclude_list,
    )

    elapsed_time = time.time() - start_time
    typer.echo(f"{elapsed_time:.2f} seconds")


@app.command()
def add_text(
    ctx: typer.Context,
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
    rag_context = ctx.obj["rag_context"]
    rag = rag_context.get_rag()
    rag.add_text(text, uri=uri, metadata=json.loads(metadata or "{}"))
    typer.echo("Text added.")


@app.command("list")
def list_documents(ctx: typer.Context):
    """List all documents in the database"""
    rag_context = ctx.obj["rag_context"]
    rag = rag_context.get_rag(require_existing=True)
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
    ctx: typer.Context,
    identifier: str,
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt"),
):
    """Remove document by path or UUID"""
    rag_context = ctx.obj["rag_context"]
    rag = rag_context.get_rag(require_existing=True)

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
    ctx: typer.Context,
    remove_missing: bool = typer.Option(
        False, "--remove-missing", help="Remove documents whose files are not found"
    ),
):
    """Rebuild embeddings and full-text index"""
    rag_context = ctx.obj["rag_context"]
    rag = rag_context.get_rag(require_existing=True)

    typer.echo("Rebuild process...")

    result = rag.rebuild(remove_missing=remove_missing)

    typer.echo("Rebuild completed:")
    typer.echo(f"  Total documents: {result['total']}")
    typer.echo(f"  Reprocessed: {result['reprocessed']}")
    typer.echo(f"  Not found: {result['not_found']}")
    typer.echo(f"  Removed: {result['removed']}")


@app.command()
def reset(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "-y", "--yes", help="Skip confirmation prompt"),
):
    """Reset/clear the entire database"""
    rag_context = ctx.obj["rag_context"]
    rag = rag_context.get_rag(require_existing=True)

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
    ctx: typer.Context,
    query: str,
    limit: int = typer.Option(5, help="Number of results to return"),
    debug: bool = typer.Option(
        False,
        "-d",
        "--debug",
        help="Print extra debug information with sentence-level details",
    ),
    peek: bool = typer.Option(
        False, "--peek", help="Print debug information using compact table format"
    ),
):
    """Search for documents using hybrid vector + full-text search"""
    rag_context = ctx.obj["rag_context"]
    start_time = time.time()

    rag = rag_context.get_rag(require_existing=True)
    results = rag.search(query, top_k=limit)

    search_time = time.time() - start_time

    results = results[:limit]

    # Get the appropriate formatter and display results
    formatter = get_formatter(debug=debug, table_view=peek)
    formatter.format_results(results, query)

    typer.echo(f"{search_time:.3f} seconds")


@app.command()
def ask(
    ctx: typer.Context,
    question: str,
    use_last_chat: bool = typer.Option(
        False,
        "--use-last-chat",
        help="Reuse the previous chat session (REPL mode only)",
    ),
):
    """Ask a question and get an answer using the LLM"""
    rag_context = ctx.obj["rag_context"]

    if use_last_chat and not rag_context.in_repl:
        raise typer.BadParameter(
            "--use-last-chat is only available when running the REPL."
        )

    start_time = time.time()

    rag = rag_context.get_rag(require_existing=True)
    cursor = rag.ask(question, reuse_chat=use_last_chat)

    spinner_stop = threading.Event()

    def spinner() -> None:
        frames = itertools.cycle("\\|/-")
        while not spinner_stop.is_set():
            sys.stdout.write(f"\rthinking {next(frames)}")
            sys.stdout.flush()
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()

    spinner_thread = threading.Thread(target=spinner, daemon=True)
    spinner_thread.start()

    has_tokens = False
    token_count = 0
    try:
        while True:
            row = cursor.fetchone()
            if row is None:
                break

            token = row["reply"]
            if token is None:
                continue

            if not has_tokens:
                spinner_stop.set()
                spinner_thread.join()
                sys.stdout.write("\n")
                has_tokens = True

            sys.stdout.write(token)
            sys.stdout.flush()
            token_count += 1
    finally:
        cursor.close()

    spinner_stop.set()
    spinner_thread.join()

    if has_tokens:
        sys.stdout.write("\n")
        sys.stdout.flush()
    else:
        typer.echo("\nNo response received.")

    elapsed_time = time.time() - start_time
    stats_line = f"{elapsed_time:.3f} seconds"
    if token_count > 0 and elapsed_time > 0:
        tokens_per_sec = token_count / elapsed_time
        stats_line = f"{stats_line} ({token_count} tokens, {tokens_per_sec:.2f} tok/s, usage: {rag.context_used()} tokens)"
    typer.echo(stats_line)


@app.command()
def quantize(
    ctx: typer.Context,
    preload: bool = typer.Option(
        False,
        "--preload",
        help="Preload quantized vectors into memory for faster search",
    ),
    cleanup: bool = typer.Option(
        False,
        "--cleanup",
        help="Clean up quantization structures instead of creating them",
    ),
):
    """Quantize vectors for faster search or clean up quantization structures"""
    rag_context = ctx.obj["rag_context"]
    rag = rag_context.get_rag(require_existing=True)

    if cleanup:
        typer.echo("Cleaning up quantization structures...")
        rag.quantize_cleanup()
        typer.echo("Quantization cleanup completed.")
    else:
        typer.echo("Starting vector quantization...")

        rag.quantize_vectors()
        if preload:
            typer.echo("Preloading quantized vectors into memory...")
            rag.quantize_preload()

        typer.echo(
            "Vector quantization completed. Now you can search with `--quantize-scan` enabled."
        )


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
        # Enable fast transfer
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
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
    """Interactive REPL mode with arrow key support"""
    typer.echo("Entering interactive mode. Type 'help' for commands or 'exit' to quit.")
    typer.echo(
        "Note: --database and configure commands are not available in REPL mode."
    )

    disabled_features = ["configure", "--database", "-db"]
    history = InMemoryHistory()

    while True:
        try:
            command = prompt(
                "sqlite-rag> ",
                history=history,
                enable_history_search=True,
            ).strip()

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
                    # Check for disabled commands in REPL
                    if args and args[0] in disabled_features:
                        typer.echo("Error: command is not available in REPL mode")
                        continue

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
