#!/usr/bin/env python3
from pathlib import Path
import shlex
import sys
from typing import Optional

import typer

from sqliterag import SQLiteRag


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
    rag.add(path, recursively=recursive)


@app.command()
def add_text(text: str, uri: Optional[str] = None, metadata: dict = {}):
    """Add a text to the database"""
    rag = SQLiteRag()
    rag.add_text(text, uri=uri, metadata=metadata)


@app.command("list")
def list_documents():
    """List all documents in the database"""
    pass


@app.command()
def remove(identifier: str):
    """Remove document by path or UUID"""
    pass


@app.command()
def rebuild():
    """Rebuild embeddings and full-text index"""
    pass


@app.command()
def reset():
    """Reset/clear the entire database"""
    pass


@app.command()
def search(
    query: str, limit: int = typer.Option(10, help="Number of results to return")
):
    """Search for documents using hybrid vector + full-text search"""
    pass


def repl_mode():
    """Interactive REPL mode"""
    typer.echo("Entering interactive mode. Type 'help' for commands or 'exit' to quit.")

    while True:
        try:
            command = input("sqlite-rag> ").strip()
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
