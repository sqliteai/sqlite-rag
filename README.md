<img src="https://private-user-images.githubusercontent.com/6153996/490482446-6e1326c5-9009-4b2d-afc1-48b7867fa215.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTgxMDM3MjMsIm5iZiI6MTc1ODEwMzQyMywicGF0aCI6Ii82MTUzOTk2LzQ5MDQ4MjQ0Ni02ZTEzMjZjNS05MDA5LTRiMmQtYWZjMS00OGI3ODY3ZmEyMTUucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDkxNyUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA5MTdUMTAwMzQzWiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9ZTI1NmZjOWJlNTY2NGM4ZmRhNTkzYzAyMWFlOTFmNjdmMmI3OWI2Mzk5MjY2NzFiMDE2NDk4ZGY1ZTFjMjNkOSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.mnWZsUVwZRjpV2nz9WDX9OA9MvkbqT4DO8nQR5trKQI" alt="https://sqlite.ai" width="110"/>

# SQLite RAG

[![Run Tests](https://github.com/sqliteai/sqlite-rag/actions/workflows/test.yaml/badge.svg?branch=main&event=release)](https://github.com/sqliteai/sqlite-rag/actions/workflows/test.yaml)
[![codecov](https://codecov.io/github/sqliteai/sqlite-rag/graph/badge.svg?token=30KYPY7864)](https://codecov.io/github/sqliteai/sqlite-rag)
![PyPI - Version](https://img.shields.io/pypi/v/sqlite-rag?link=https%3A%2F%2Fpypi.org%2Fproject%2Fsqlite-rag%2F)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sqlite-rag?link=https%3A%2F%2Fpypi.org%2Fproject%2Fsqlite-rag)

A hybrid search engine built on SQLite with [SQLite AI](https://github.com/sqliteai/sqlite-ai) and [SQLite Vector](https://github.com/sqliteai/sqlite-vector) extensions. SQLite RAG combines vector similarity search with full-text search ([FTS5](https://www.sqlite.org/fts5.html) extension) using Reciprocal Rank Fusion (RRF) for enhanced document retrieval.

## Features

- **Hybrid Search**: Combines vector embeddings with full-text search for optimal results
- **SQLite-based**: Built on SQLite with AI and Vector extensions for reliability and performance
- **Multi-format Text Support**: Process text file formats including PDF, DOCX, Markdown, code files
- **Recursive Character Text Splitter**: Token-aware text chunking with configurable overlap
- **Interactive CLI**: Command-line interface with interactive REPL mode
- **Flexible Configuration**: Customizable embedding models, search weights, and chunking parameters

## Installation

```bash
pip install sqlite-rag
```

## Quick Start

Download the model [Embedding Gemma](https://huggingface.co/unsloth/embeddinggemma-300m-GGUF) from Hugging Face chosen as default model:

```bash
sqlite-rag download-model unsloth/embeddinggemma-300m-GGUF embeddinggemma-300M-Q8_0.gguf
```

Then start with default settings:

```bash
# Initialize sqliterag.sqlite database and add documents
sqlite-rag add /path/to/documents --recursive

# Search your documents
sqlite-rag search "your search query"

# Interactive mode
sqlite-rag
> help
> search "interactive search"
> exit
```

For help run:

```bash
sqlite-rag --help
```

## CLI Commands

### Configuration

Settings are stored in the database and should be set before adding any documents.

```bash
# Interactive configuration
sqlite-rag configure

# View current settings
sqlite-rag settings

# View available configuration options
sqlite-rag configure --help
```

To use a different database path, use the global `--database` option:

```bash
# Single command with custom database
sqlite-rag --database mydb.db add-text "What's AI?"

# Interactive mode with custom database
sqlite-rag --database mydb.db
```

### Model Management

You can experiment with other models from Hugging Face by downloading them with:

```bash
# Download GGUF models from Hugging Face
sqlite-rag download-model <model-repo> <filename>
```

## Supported File Formats

SQLite RAG supports the following file formats:

- **Text**: `.txt`, `.md`, `.mdx`, `.csv`, `.json`, `.xml`, `.yaml`, `.yml`
- **Documents**: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- **Code**: `.c`, `.cpp`, `.css`, `.go`, `.h`, `.hpp`, `.html`, `.java`, `.js`, `.mjs`, `.kt`, `.php`, `.py`, `.rb`, `.rs`, `.swift`, `.ts`, `.tsx`
- **Web Frameworks**: `.svelte`, `.vue`

## Development

### Installation

For development, clone the repository and install with development dependencies:

```bash
# Clone the repository
git clone https://github.com/sqliteai/sqlite-rag.git
cd sqlite-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .[dev]
```
## How It Works

1. **Document Processing**: Files are processed and split into overlapping chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using AI models
3. **Dual Indexing**: Content is indexed for both vector similarity and full-text search
4. **Hybrid Search**: Queries are processed through both search methods
5. **Result Fusion**: Results are combined using Reciprocal Rank Fusion for optimal relevance
