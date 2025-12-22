# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Local GraphRAG MCP Server is a Model Context Protocol (MCP) server that provides a local, embedded Knowledge Graph for Retrieval-Augmented Generation (RAG). Unlike standard vector stores, this implements Graph RAG using vector search to find relevant entities ("anchors") and then traversing relationships to retrieve connected context.

**Core Philosophy**: "One Duck to Rule Them All"
- Single process architecture (no Docker containers or background services)
- Single file storage (entire graph, vector index, and properties in one `.duckdb` file)
- Hybrid native operations (vector search and graph traversal in the same engine)

## Architecture

### The Stack
- **Storage Layer**: Two SQL tables: `NODES` and `EDGES`
- **Semantic Layer**: `NODES` table contains `description_embedding` column indexed by HNSW (via `vss` extension)
- **Graph Layer**: `CREATE PROPERTY GRAPH` view maps foreign keys in `EDGES` to IDs in `NODES`, enabling SQL/PGQ (SQL:2023 standard) graph pattern queries

### DuckDB Extensions Required
1. **duckpgq**: Property Graph Queries for relationship traversal
2. **vss**: Vector Similarity Search for semantic entry points
3. **json**: Schema-less property storage

### Retrieval Strategy: "Anchor & Expand"
1. **Anchor (Vector Search)**: Embed query → Find top-k semantically relevant nodes in `NODES` table
2. **Expansion (Graph Traversal)**: Execute Property Graph Query starting from anchor nodes, traversing outwards (configurable hops)

## Data Model

### Nodes Table
- `id`: Unique entity identifier (e.g., "entity:auth_service")
- `label`: Entity type (e.g., "Service", "Database")
- `description`: Natural language description
- `description_embedding`: Vector representation (indexed via HNSW)
- `properties`: JSON blob for schema-less attributes

### Edges Table
- `source`: Source node ID
- `target`: Target node ID
- `label`: Relationship type (e.g., "CONNECTS_TO", "MANAGED_BY")
- `properties`: JSON blob for edge metadata

## MCP Tools

### add_knowledge_graph
Directly adds nodes and edges to the graph. Used when the LLM has extracted structured information from documents.

Input schema:
```json
{
  "nodes": [{"id": "entity:...", "label": "...", "description": "...", "properties": {...}}],
  "edges": [{"source": "entity:...", "target": "entity:...", "label": "...", "properties": {...}}]
}
```

### query_graph_rag
The primary retrieval tool implementing the "Anchor & Expand" strategy.

Input schema:
```json
{
  "query": "Natural language query",
  "top_k_anchors": 3,
  "hops": 2
}
```

Internally:
1. Embeds query
2. Finds top-k nodes via vector similarity
3. Runs DuckPGQ graph traversal from anchor nodes
4. Returns structured subgraph as context

## Configuration

### Environment Variables
- `GRAPHRAG_DB_PATH`: Path to .duckdb file (default: `./graph.duckdb`)
- `GRAPHRAG_EMBED_MODEL`: Sentence transformer model (default: `all-MiniLM-L6-v2`)
- `GRAPHRAG_DEBUG`: Set to `true` to log generated SQL/PGQ queries

### MCP Server Configuration
Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "local-graphrag": {
      "command": "local-graphrag-mcp",
      "args": [
        "--db-path", "/absolute/path/to/my_graph.duckdb",
        "--embedding-model", "all-MiniLM-L6-v2"
      ]
    }
  }
}
```

## CLI Commands (Planned)

```bash
# Initialize a new graph database
local-graphrag init ./knowledge_graph.duckdb

# Ingest a document (auto-extracts entities using LLM)
local-graphrag ingest ./docs/architecture.md --extract-model claude-3-5-sonnet

# Custom extraction prompt
local-graphrag ingest ./paper.pdf --prompt-file ./custom_extraction.txt

# Query interactively
local-graphrag query "How does the ingestion pipeline work?"
```

## Development Setup

### Prerequisites
- Python 3.10+
- C++ compiler (may be needed for DuckDB extensions on some Linux distros)

### Installation
```bash
# From source
git clone https://github.com/nonatofabio/local_graphrag_mcp.git
cd local_graphrag_mcp
pip install -e .

# Dev dependencies
pip install -r requirements-dev.txt
```

### Running Tests
```bash
# All tests
pytest tests/

# Specific test file
pytest tests/test_extensions.py
```

## Key Design Decisions

### Why SQL/PGQ over Python Graph Libraries?
1. **Performance**: C++ vectorized execution vs Python loops (O(N²) for traversals)
2. **Standardization**: SQL:2023 standard instead of custom Cypher parsers
3. **Zero-Copy**: Vector DB and Graph DB share the same table rows—no data synchronization

### Why Graph RAG vs Standard Vector RAG?
- Standard RAG retrieves chunks based on keyword similarity
- Graph RAG retrieves based on relationships, enabling context-aware retrieval
- Example: Query "Who manages Auth Service?" → Finds "Auth Service" node → Traverses `[MANAGED_BY]` edge → Returns "Alice" even if Alice's description doesn't mention "Auth Service"

## Ingestion Workflow
Unlike simple vector stores, Graph RAG requires structured extraction before storage:
1. Host LLM reads document
2. LLM identifies entities (nodes) and relationships (edges)
3. LLM calls `add_knowledge_graph()` with structured data
4. MCP server calculates embeddings for node descriptions
5. Data inserted into DuckDB tables with graph topology preserved
