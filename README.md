# Local GraphRAG MCP Server

<!-- mcp-name: io.github.nonatofabio/local-graphrag-mcp -->

A Model Context Protocol (MCP) server that provides a local, embedded Knowledge Graph for Retrieval-Augmented Generation (RAG).

Unlike standard vector stores, this project implements **Graph RAG**: it uses vector search to find relevant entities ("anchors") and then traverses relationships to retrieve connected context. It runs entirely in a single Python process using FAISS for vector search and NetworkX for graph operations.

## Why Graph RAG?

**Standard RAG** (like FAISS alone) retrieves chunks based on semantic similarity to the query.

**Graph RAG** retrieves chunks based on semantic similarity AND relationship traversal.

**Example:**
- **Standard RAG**: User asks "Who manages the Auth Service?" → Finds chunks semantically similar to the query → May miss "Alice" if her description doesn't semantically match "manages auth service"
- **Graph RAG**: Finds "Auth Service" node via semantic search → Traverses `[MANAGED_BY]` edge → Returns "Alice" and her full context, even if her description doesn't directly relate to the query

## Features

### Core Capabilities

- **Pure Python**: FAISS (vector search) + NetworkX (graph traversal) - no external services required
- **Single Process**: Runs entirely in-process, no Docker containers or background services
- **Hybrid Retrieval**: Combines vector similarity search (anchor discovery) with graph traversal (context expansion)
- **Persistent Storage**: Three-file storage system (FAISS index, NetworkX graph, metadata JSON)
- **MCP Compatible**: Plug-and-play with Claude Desktop, Cursor, or any MCP client
- **Configurable Traversal**: Control anchor count (top-k) and expansion depth (hops)

## Installation

### Prerequisites

- Python 3.10+
- pip

### From Source

```bash
git clone https://github.com/nonatofabio/local_graphrag_mcp.git
cd local_graphrag_mcp
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

### Ollama Setup (for Local Entity Extraction)

The CLI uses **Ollama** by default for local entity extraction. Install and set up Ollama:

```bash
# Install Ollama (macOS)
brew install ollama

# Or download from: https://ollama.ai

# Start Ollama service
ollama serve

# Pull the default model (Llama 3.2 3B)
ollama pull llama3.2:3b
```

**Optional: Cloud Extraction with Claude API**

For higher precision entity extraction, install the optional cloud dependencies:

```bash
pip install -e ".[cloud]"
export ANTHROPIC_API_KEY='your-key-here'
```

## Usage

### CLI Commands

The CLI provides document indexing with automatic entity extraction:

```bash
# Index documents using local Ollama (default)
local-graphrag index document.pdf

# Index multiple documents recursively
local-graphrag index -r documents/

# Index with specific Ollama model
local-graphrag index --extract-model llama3.1:8b document.pdf

# Index using Claude API (requires anthropic package and API key)
local-graphrag index --use-cloud document.pdf

# Query the knowledge graph
local-graphrag search "What database does the auth service use?"

# List indexed entities
local-graphrag list
```

### 1. Running the MCP Server

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-graphrag": {
      "command": "local-graphrag-mcp",
      "args": [
        "--index-dir", "/absolute/path/to/graphrag_data",
        "--embed", "all-MiniLM-L6-v2"
      ]
    }
  }
}
```

**Arguments:**
- `--index-dir`: Directory where graph data will be stored (default: `./graphrag_index`)
- `--embed`: Sentence transformer model name (default: `all-MiniLM-L6-v2`)
- `--debug`: Enable debug logging (optional)

### 2. Available Tools

The server exposes two tools to the LLM:

#### `add_knowledge_graph`

Directly adds nodes and edges to the knowledge graph. Use this when the LLM has extracted structured information from documents.

**Input Schema:**
```json
{
  "nodes": [
    {
      "id": "entity:auth_service",
      "label": "Service",
      "description": "The authentication microservice handling user login",
      "properties": {"language": "Python", "version": "2.1.0"}
    }
  ],
  "edges": [
    {
      "source": "entity:auth_service",
      "target": "entity:postgres_db",
      "label": "CONNECTS_TO",
      "properties": {"protocol": "TCP"}
    }
  ]
}
```

**Returns:**
```json
{
  "success": true,
  "nodes_added": 1,
  "edges_added": 1
}
```

#### `query_graph_rag`

The primary retrieval tool implementing the "Anchor & Expand" strategy.

**Input Schema:**
```json
{
  "query": "What database does the auth service use?",
  "top_k_anchors": 3,
  "hops": 2
}
```

**Parameters:**
- `query`: Natural language query
- `top_k_anchors`: Number of most similar nodes to find (default: 3)
- `hops`: How many relationship hops to traverse from anchors (default: 2)

**Returns:**
```json
{
  "success": true,
  "query": "What database does the auth service use?",
  "anchors": [
    {
      "id": "entity:auth_service",
      "label": "Service",
      "description": "The authentication microservice...",
      "similarity_score": 0.85
    }
  ],
  "subgraph": {
    "nodes": [...],
    "edges": [...]
  }
}
```

### How It Works Internally

**Anchor & Expand Strategy:**

1. **Embed**: Converts your query into a 384-dimensional vector using sentence-transformers
2. **Vector Search**: Uses FAISS to find the top-k most similar nodes (anchors) via L2 distance
3. **Graph Expansion**: Performs BFS traversal from anchor nodes up to N hops
   - Traverses both outgoing edges (successors)
   - Collects all connected nodes and relationships
4. **Context Assembly**: Returns structured subgraph to the LLM with anchor nodes and expanded neighborhood

**Example:**
```
Query: "What database does the auth service use?"

Step 1 (Anchor): Find "entity:auth_service" (similarity: 0.85)
Step 2 (Expand 1-hop): Traverse [CONNECTS_TO] → "entity:postgres_db"
Step 3 (Expand 2-hops): Traverse from postgres_db → "entity:backup_service"

Result: Subgraph with 3 nodes and 2 edges
```

## Architecture

### Stack Overview

```
[ LLM Client ] <--> [ MCP Server ] <--> [ GraphRAGVectorStore ]
                                              |
                              +---------------+---------------+
                              |                               |
                         [ FAISS Index ]              [ NetworkX Graph ]
                       (vector similarity)          (relationship traversal)
                              |                               |
                              +----------- Shared IDs --------+
```

### Data Model

**Nodes:**
- `id`: Unique entity identifier (e.g., "entity:auth_service")
- `label`: Entity type (e.g., "Service", "Database", "Person")
- `description`: Natural language description (embedded as 384D vector)
- `properties`: Schema-less JSON attributes

**Edges:**
- `source`: Source node ID
- `target`: Target node ID
- `label`: Relationship type (e.g., "CONNECTS_TO", "MANAGED_BY")
- `properties`: Schema-less JSON metadata

### Storage

Three files are created in the index directory:

1. **`nodes.index`**: FAISS IndexFlatL2 (vector embeddings)
2. **`graph.gpickle`**: NetworkX directed graph (nodes, edges, properties)
3. **`metadata.json`**: Node descriptions and embedding model info

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run standalone integration test
python tests/test_standalone.py
```

### Project Structure

```
local-graphrag-mcp/
├── local_graphrag_mcp/
│   ├── __init__.py          # Package exports
│   ├── __main__.py          # CLI entry point
│   └── server.py            # Core implementation (585 lines)
├── tests/
│   ├── __init__.py
│   └── test_standalone.py   # End-to-end integration test
├── pyproject.toml           # Package configuration
├── .mcp.json.example        # Example MCP config
├── CLAUDE.md                # Development guidance
└── README.md                # This file
```

### Key Design Decisions

**Why FAISS + NetworkX instead of a graph database?**
1. **Simplicity**: Pure Python, no external services (Redis, Neo4j, etc.)
2. **Performance**: FAISS for fast vector search, NetworkX for in-memory graph traversal
3. **Portability**: Entire graph fits in three files, easy to version control or share

**Why not DuckDB/DuckPGQ?**
- Initial implementation attempted SQL/PGQ property graphs
- Compatibility issues with DuckPGQ extension across DuckDB versions
- Complex SQL syntax for graph patterns vs. simple Python NetworkX API
- Decision: Prioritize simplicity and reliability

## Roadmap

- [x] v0.1.0: Core FAISS + NetworkX implementation
- [x] v0.1.0: MCP server with add/query tools
- [x] v0.1.0: Persistent storage and reload
- [ ] v0.2.0: CLI commands for initialization and ingestion
- [ ] v0.3.0: Smart document ingestion (auto-extract entities via LLM)
- [ ] v0.4.0: Graph visualization tools
- [ ] v0.5.0: Multi-hop reasoning enhancements

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

MIT License - see LICENSE file for details

## Acknowledgments

This project follows patterns established by [local_faiss_mcp](https://github.com/yourusername/local_faiss_mcp) for MCP integration and coding style.
