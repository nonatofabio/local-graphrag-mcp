[This is a tentative README, unformatted, to help drive the development of this project.]

Local GraphRAG MCP Server

<!-- mcp-name: io.github.nonatofabio/local-graphrag-mcp -->

A Model Context Protocol (MCP) server that provides a local, embedded Knowledge Graph for Retrieval-Augmented Generation (RAG).

Unlike standard vector stores, this project implements Graph RAG: it uses vector search to find relevant entities ("anchors") and then traverses relationships to retrieve connected context. It runs entirely in a single process using DuckDB + DuckPGQ + VSS.

Why Graph RAG?

Standard RAG (like FAISS) retrieves chunks based on keyword similarity.
Graph RAG retrieves chunks based on relationships.

Standard RAG: User asks "Who is the manager of the Auth Service?" -> Returns chunks mentioning "Auth Service".

Graph RAG: Finds "Auth Service" node -> Traverses [MANAGED_BY] edge -> Returns "Alice", even if the text about Alice doesn't mention "Auth Service".

Features

Core Capabilities

Single-File Database: Entire graph and vector index stored in one portable .duckdb file.

Hybrid Retrieval: Combines HNSW Vector Search (entry) + Property Graph Traversal (expansion).

Zero-Copy Architecture: No syncing between a Graph DB and a Vector DB; they are the same table rows.

Standardized Querying: Uses SQL/PGQ (SQL:2023 standard) for graph patterns.

MCP Compatible: Plug-and-play with Claude Desktop, Cursor, or any MCP client.

Quickstart

# Install
pip install local-graphrag-mcp

# Initialize a new graph database
local-graphrag init ./knowledge_graph.duckdb

# Ingest a document (auto-extracts entities using your local LLM or API)
local-graphrag ingest ./docs/architecture.md --extract-model claude-3-5-sonnet

# Query interactively
local-graphrag query "How does the ingestion pipeline work?"


Installation

Prerequisites

Python 3.10+

A working C++ compiler (sometimes needed for DuckDB extensions on specific Linux distros, though wheels usually suffice).

From PyPI

pip install local-graphrag-mcp


From Source

git clone [https://github.com/nonatofabio/local_graphrag_mcp.git](https://github.com/nonatofabio/local_graphrag_mcp.git)
cd local_graphrag_mcp
pip install -e .


Usage

1. Running the MCP Server

Add to your claude_desktop_config.json:

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


2. Available Tools

The server exposes the following tools to the LLM:

add_knowledge_graph

Directly adds nodes and edges to the graph. Useful when the LLM has already "read" a file and wants to memorize structured info.

Input Schema:

{
  "nodes": [
    {"id": "entity:auth_service", "label": "Service", "description": "The authentication microservice", "properties": {"language": "Python"}}
  ],
  "edges": [
    {"source": "entity:auth_service", "target": "entity:postgres_db", "label": "CONNECTS_TO", "properties": {"port": 5432}}
  ]
}


query_graph_rag

The magic tool. It performs the "Anchor & Expand" search strategy.

Input Schema:

{
  "query": "What database does the auth service use?",
  "top_k_anchors": 3,
  "hops": 2
}


How it works internaly:

Embed: Embeds input query: [0.12, 0.88, ...]

Vector Search: Finds top 3 nodes in NODES table similar to the query.

Graph Expansion: Runs a DuckPGQ query:

FROM GRAPH_TABLE (
    mcp_graph
    MATCH (start)-[e]->(end)
    WHERE start.id IN ('found_id_1', 'found_id_2')
    COLUMNS (start.properties, e.label, end.properties)
)


Context: Returns a structured subgraph to the LLM.

Configuration

Environment Variables

You can configure the server via environment variables or CLI flags:

GRAPHRAG_DB_PATH: Path to the .duckdb file (Default: ./graph.duckdb)

GRAPHRAG_EMBED_MODEL: Sentence transformer model (Default: all-MiniLM-L6-v2)

GRAPHRAG_DEBUG: Set to true to see generated SQL/PGQ queries in logs.

Custom Extraction Prompt

If using the CLI ingest command, you can provide a custom prompt for the extraction phase:

local-graphrag ingest ./paper.pdf --prompt-file ./custom_extraction.txt


Development

Architecture Overview

[ LLM Client ] <--> [ MCP Server (FastMCP) ] <--> [ DuckDB Process ]
                                                         |
                                         +---------------+---------------+
                                         |                               |
                                     [ NODES Table ]               [ EDGES Table ]
                                     (id, vector, json)           (src, tgt, label)
                                         |                               |
                                         +------- [ Property Graph ] ----+
                                                  (View via DuckPGQ)


Running Tests

# Install dev dependencies
pip install -r requirements-dev.txt

# Run unit tests
pytest tests/

# Run specific DuckDB extension tests
pytest tests/test_extensions.py


Roadmap

[ ] v0.1.0: Basic DuckDB + DuckPGQ + VSS integration.

[ ] v0.2.0: "Smart Ingest" CLI that uses local LLMs (Ollama) to extract entities.

[ ] v0.3.0: Visualization server (interactive graph explorer).

[ ] v0.4.0: Multi-hop reasoning prompts.

License

MIT