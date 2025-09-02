# SQLite RAG

A hybrid search engine built on SQLite with AI and Vector extensions. SQLite-RAG combines vector similarity search with full-text search using Reciprocal Rank Fusion (RRF) for enhanced document retrieval.

## Features

- **Hybrid Search**: Combines vector embeddings with full-text search for optimal results
- **SQLite-based**: Built on SQLite with AI and Vector extensions for reliability and performance
- **Multi-format Support**: Process 25+ file formats including PDF, DOCX, Markdown, code files
- **Intelligent Chunking**: Token-aware text chunking with configurable overlap
- **Interactive CLI**: Command-line interface with interactive REPL mode
- **Flexible Configuration**: Customizable embedding models, search weights, and chunking parameters

## Installation

```bash
pip install sqlite-rag
```

## Quick Start

```bash
# Initialize and add documents
sqlite-rag add /path/to/documents --recursive

# Search your documents
sqlite-rag search "your search query"

# Interactive mode
sqlite-rag
> help
> search "interactive search"
> exit
```

## CLI Commands

### Document Management

**Add files or directories:**
```bash
sqlite-rag add <path> [--recursive] [--absolute-paths] [--metadata '{"key": "value"}']
```

**Add raw text:**
```bash
sqlite-rag add-text "your text content" [uri] [--metadata '{"key": "value"}']
```

**List all documents:**
```bash
sqlite-rag list
```

**Remove documents:**
```bash
sqlite-rag remove <path-or-uuid> [--yes]
```

### Search & Query

**Hybrid search:**
```bash
sqlite-rag search "your query" [--limit 10] [--debug]
```

Use `--debug` to see detailed ranking information including vector ranks, FTS ranks, and combined scores.

### Database Operations

**Rebuild indexes and embeddings:**
```bash
sqlite-rag rebuild [--remove-missing]
```

**Clear entire database:**
```bash
sqlite-rag reset [--yes]
```

### Configuration

**View current settings:**
```bash
sqlite-rag settings
```

**Update configuration:**
```bash
sqlite-rag set [options]
```

Available settings:
- `--model-path-or-name`: Embedding model (file path or HuggingFace model)
- `--embedding-dim`: Vector dimensions
- `--chunk-size`: Text chunk size (tokens)
- `--chunk-overlap`: Token overlap between chunks
- `--weight-fts`: Full-text search weight (0.0-1.0)
- `--weight-vec`: Vector search weight (0.0-1.0)
- `--quantize-scan`: Enable quantized vectors for faster search
- `--quantize-preload`: Preload quantized vectors in memory

## Python API

```python
from sqlite_rag import SQLiteRag

# Create RAG instance
rag = SQLiteRag.create("./database.sqlite")

# Add documents
rag.add("/path/to/documents", recursive=True)
rag.add_text("Raw text content", uri="doc.txt")

# Search
results = rag.search("search query", top_k=5)
for result in results:
    print(f"Score: {result.score}")
    print(f"Content: {result.content}")
    print(f"URI: {result.uri}")

# List documents
documents = rag.list_documents()

# Remove document
rag.remove_document("document-id-or-path")

# Database operations
rag.rebuild(remove_missing=True)
rag.reset()
```

## Supported File Formats

SQLite-RAG supports 25+ file formats through the MarkItDown library:

- **Text**: `.txt`, `.md`, `.csv`, `.json`, `.xml`
- **Documents**: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- **Code**: `.py`, `.js`, `.html`, `.css`, `.sql`
- **And many more**: `.rtf`, `.odt`, `.epub`, `.zip`, etc.

## How It Works

1. **Document Processing**: Files are processed and split into overlapping chunks
2. **Embedding Generation**: Text chunks are converted to vector embeddings using AI models
3. **Dual Indexing**: Content is indexed for both vector similarity and full-text search
4. **Hybrid Search**: Queries are processed through both search methods
5. **Result Fusion**: Results are combined using Reciprocal Rank Fusion for optimal relevance

## Default Configuration

- **Model**: Qwen3-Embedding-0.6B (Q8_0 quantized, 1024 dimensions)
- **Chunking**: 12,000 tokens per chunk with 1,200 token overlap
- **Vectors**: FLOAT16 storage with cosine similarity
- **Search**: Equal weighting (1.0) for vector and full-text results
- **Database**: `./sqliterag.sqlite`

## Extensions Required

SQLite-RAG requires these SQLite extensions:

- **[sqlite-ai](https://github.com/sqliteai/sqlite-ai)**: LLM model loading and embedding generation
- **[sqlite-vector](https://github.com/sqliteai/sqlite-vector)**: Vector storage and similarity search

These are automatically installed as dependencies.
