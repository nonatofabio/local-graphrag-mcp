"""
Local GraphRAG MCP Server

A Model Context Protocol server that implements GraphRAG using FAISS for
vector search and NetworkX for graph traversal.
"""

import argparse
import asyncio
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import faiss
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize MCP server
app = Server("local-graphrag-mcp")

# Global vector store (initialized in main())
vector_store: "GraphRAGVectorStore | None" = None


class GraphRAGVectorStore:
    """
    GraphRAG Vector Store using FAISS for vector search and NetworkX for graph traversal.

    Implements the "Anchor & Expand" strategy:
    1. Vector search to find relevant nodes (anchors) using FAISS
    2. Graph traversal to expand context from anchors using NetworkX
    """

    def __init__(self, index_dir: str | Path, embedding_model_name: str = "all-MiniLM-L6-v2", debug: bool = False):
        """
        Initialize GraphRAG vector store.

        Args:
            index_dir: Directory to store FAISS index and graph data
            embedding_model_name: Sentence-transformers model name
            debug: Enable debug logging
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / "nodes.index"
        self.graph_path = self.index_dir / "graph.gpickle"
        self.metadata_path = self.index_dir / "metadata.json"

        self.embedding_model_name = embedding_model_name
        self.debug = debug

        # Lazy-loaded embedding model
        self._embedding_model = None
        self._dimension = None

        # Initialize FAISS index
        self.index = None
        self.graph = nx.DiGraph()  # Directed graph
        self.metadata = {"nodes": [], "model": embedding_model_name}

        # Load existing data if available
        self._load()

    @property
    def embedding_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model on first access."""
        if self._embedding_model is None:
            if self.debug:
                logging.info(f"Loading embedding model: {self.embedding_model_name}")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    @property
    def dimension(self) -> int:
        """Get embedding dimension from the model."""
        if self._dimension is None:
            # Get dimension by encoding a test string
            test_embedding = self.embedding_model.encode(["test"], convert_to_numpy=True)
            self._dimension = test_embedding.shape[1]
            if self.debug:
                logging.info(f"Embedding dimension: {self._dimension}")
        return self._dimension

    def _load(self):
        """Load existing FAISS index, graph, and metadata from disk."""
        # Load FAISS index
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.debug:
                logging.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
        else:
            # Create new index (using L2 distance)
            self.index = faiss.IndexFlatL2(self.dimension)
            if self.debug:
                logging.info(f"Created new FAISS index (dimension={self.dimension})")

        # Load NetworkX graph
        if self.graph_path.exists():
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            if self.debug:
                logging.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        else:
            self.graph = nx.DiGraph()
            if self.debug:
                logging.info("Created new empty graph")

        # Load metadata
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            if self.debug:
                logging.info(f"Loaded metadata for {len(self.metadata['nodes'])} nodes")
        else:
            self.metadata = {"nodes": [], "model": self.embedding_model_name}
            if self.debug:
                logging.info("Created new metadata")

    def save(self):
        """Save FAISS index, graph, and metadata to disk."""
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))

        # Save NetworkX graph
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save metadata
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        if self.debug:
            logging.info("Saved index, graph, and metadata to disk")

    def add_nodes_and_edges(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Add nodes and edges to the knowledge graph.

        Args:
            nodes: List of node dicts with keys: id, label, description, properties
            edges: List of edge dicts with keys: source, target, label, properties

        Returns:
            Result dict with success status and counts
        """
        try:
            # Compute embeddings for all node descriptions
            descriptions = [node['description'] for node in nodes]
            embeddings = self.embedding_model.encode(descriptions, convert_to_numpy=True)

            if self.debug:
                logging.info(f"Computed embeddings for {len(nodes)} nodes")

            # Add nodes to FAISS and NetworkX
            for i, node in enumerate(nodes):
                node_id = node['id']

                # Add to FAISS index
                embedding = embeddings[i].astype('float32').reshape(1, -1)
                self.index.add(embedding)

                # Add to NetworkX graph with attributes
                self.graph.add_node(
                    node_id,
                    label=node['label'],
                    description=node['description'],
                    properties=node.get('properties', {}),
                    index_position=self.index.ntotal - 1  # Track FAISS position
                )

                # Add to metadata
                self.metadata['nodes'].append({
                    'id': node_id,
                    'label': node['label'],
                    'description': node['description'],
                    'properties': node.get('properties', {})
                })

            if self.debug:
                logging.info(f"Added {len(nodes)} nodes to index and graph")

            # Add edges to NetworkX graph
            for edge in edges:
                self.graph.add_edge(
                    edge['source'],
                    edge['target'],
                    label=edge['label'],
                    properties=edge.get('properties', {})
                )

            if self.debug:
                logging.info(f"Added {len(edges)} edges to graph")

            # Save to disk
            self.save()

            return {
                "success": True,
                "nodes_added": len(nodes),
                "edges_added": len(edges)
            }

        except Exception as e:
            logging.error(f"Error adding nodes and edges: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def query_graph_rag(
        self,
        query_text: str,
        top_k_anchors: int = 3,
        hops: int = 2
    ) -> dict[str, Any]:
        """
        Query the knowledge graph using the Anchor & Expand strategy.

        Args:
            query_text: Natural language query
            top_k_anchors: Number of anchor nodes to find via vector search
            hops: Number of graph traversal hops

        Returns:
            Result dict with anchors and expanded subgraph
        """
        try:
            if self.index.ntotal == 0:
                return {
                    "success": True,
                    "query": query_text,
                    "anchors": [],
                    "subgraph": {"nodes": [], "edges": []}
                }

            # Step 1: Embed query
            query_embedding = self.embedding_model.encode(query_text, convert_to_numpy=True).astype('float32').reshape(1, -1)

            if self.debug:
                logging.info(f"Query: {query_text}")
                logging.info(f"Finding top {top_k_anchors} anchors...")

            # Step 2: Find anchor nodes via FAISS vector similarity
            k = min(top_k_anchors, self.index.ntotal)
            distances, indices = self.index.search(query_embedding, k)

            # Get anchor node IDs
            anchor_nodes = []
            anchor_ids = []
            for i, idx in enumerate(indices[0]):
                node_data = self.metadata['nodes'][idx]
                node_data['distance'] = float(distances[0][i])
                anchor_nodes.append(node_data)
                anchor_ids.append(node_data['id'])

            if self.debug:
                logging.info(f"Found {len(anchor_nodes)} anchor nodes")
                for anchor in anchor_nodes:
                    logging.info(f"  - {anchor['id']} (distance: {anchor['distance']:.4f})")

            # Step 3: Expand via graph traversal (BFS from anchors)
            expanded_nodes = set(anchor_ids)
            expanded_edges = []

            for anchor_id in anchor_ids:
                if anchor_id not in self.graph:
                    continue

                # BFS traversal from anchor
                visited = {anchor_id}
                queue = [(anchor_id, 0)]  # (node_id, current_hop)

                while queue:
                    current_node, current_hop = queue.pop(0)

                    if current_hop >= hops:
                        continue

                    # Get outgoing edges
                    for neighbor in self.graph.successors(current_node):
                        expanded_nodes.add(neighbor)
                        edge_data = self.graph.edges[current_node, neighbor]
                        expanded_edges.append({
                            'source': current_node,
                            'target': neighbor,
                            'label': edge_data.get('label', ''),
                            'properties': edge_data.get('properties', {})
                        })

                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, current_hop + 1))

            if self.debug:
                logging.info(f"Expanded to {len(expanded_nodes)} nodes and {len(expanded_edges)} edges")

            # Step 4: Format results
            subgraph_nodes = []
            for node_id in expanded_nodes:
                if node_id in self.graph:
                    node_attrs = self.graph.nodes[node_id]
                    subgraph_nodes.append({
                        'id': node_id,
                        'label': node_attrs.get('label', ''),
                        'description': node_attrs.get('description', ''),
                        'properties': node_attrs.get('properties', {}),
                        'is_anchor': node_id in anchor_ids
                    })

            return {
                "success": True,
                "query": query_text,
                "anchors": anchor_nodes,
                "subgraph": {
                    "nodes": subgraph_nodes,
                    "edges": expanded_edges
                }
            }

        except Exception as e:
            logging.error(f"Error querying graph: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query_text
            }


# ============================================================================
# MCP Server Handlers
# ============================================================================

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="add_knowledge_graph",
            description="Add nodes and edges to the knowledge graph. Embeddings are computed automatically for node descriptions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "description": "List of nodes to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "description": "Unique node identifier (e.g., 'entity:auth_service')"
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Entity type (e.g., 'Service', 'Database', 'Person')"
                                },
                                "description": {
                                    "type": "string",
                                    "description": "Natural language description for embedding"
                                },
                                "properties": {
                                    "type": "object",
                                    "description": "Additional properties as JSON",
                                    "default": {}
                                }
                            },
                            "required": ["id", "label", "description"]
                        }
                    },
                    "edges": {
                        "type": "array",
                        "description": "List of edges to add",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "description": "Source node ID"
                                },
                                "target": {
                                    "type": "string",
                                    "description": "Target node ID"
                                },
                                "label": {
                                    "type": "string",
                                    "description": "Relationship type (e.g., 'CONNECTS_TO', 'MANAGES')"
                                },
                                "properties": {
                                    "type": "object",
                                    "description": "Additional properties as JSON",
                                    "default": {}
                                }
                            },
                            "required": ["source", "target", "label"]
                        }
                    }
                },
                "required": ["nodes", "edges"]
            }
        ),
        Tool(
            name="query_graph_rag",
            description="Query the knowledge graph using the Anchor & Expand strategy. Finds relevant nodes via vector search, then expands via graph traversal.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query"
                    },
                    "top_k_anchors": {
                        "type": "integer",
                        "description": "Number of anchor nodes to find via vector search",
                        "default": 3
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Number of graph traversal hops",
                        "default": 2
                    }
                },
                "required": ["query"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle MCP tool calls."""
    global vector_store

    if vector_store is None:
        return [TextContent(type="text", text="Error: Vector store not initialized")]

    try:
        if name == "add_knowledge_graph":
            # Extract and validate arguments
            nodes = arguments.get("nodes", [])
            edges = arguments.get("edges", [])

            if not nodes:
                return [TextContent(type="text", text="Error: No nodes provided")]

            # Call the vector store method
            result = vector_store.add_nodes_and_edges(nodes, edges)

            if result["success"]:
                message = f"✓ Successfully added {result['nodes_added']} nodes and {result['edges_added']} edges to the knowledge graph."
            else:
                message = f"✗ Failed to add knowledge graph: {result.get('error', 'Unknown error')}"

            return [TextContent(type="text", text=message)]

        elif name == "query_graph_rag":
            # Extract arguments
            query = arguments.get("query", "")
            top_k_anchors = arguments.get("top_k_anchors", 3)
            hops = arguments.get("hops", 2)

            if not query:
                return [TextContent(type="text", text="Error: No query provided")]

            # Call the vector store method
            result = vector_store.query_graph_rag(query, top_k_anchors, hops)

            if result["success"]:
                # Format results as readable text
                anchors = result.get("anchors", [])
                subgraph = result.get("subgraph", {})
                nodes = subgraph.get("nodes", [])
                edges = subgraph.get("edges", [])

                message = f"# Query Results: \"{query}\"\n\n"

                # Show anchors
                message += f"## Anchor Nodes ({len(anchors)} found)\n"
                for anchor in anchors:
                    message += f"- **{anchor['id']}** ({anchor['label']}): {anchor['description']}\n"
                    message += f"  Distance: {anchor.get('distance', 0):.4f}\n"

                # Show expanded subgraph
                message += f"\n## Expanded Subgraph\n"
                message += f"- Nodes: {len(nodes)}\n"
                message += f"- Edges: {len(edges)}\n\n"

                # Show edges (relationships)
                if edges:
                    message += "### Relationships:\n"
                    for edge in edges:
                        message += f"- {edge['source']} --[{edge['label']}]--> {edge['target']}\n"

                # Include JSON for programmatic access
                message += f"\n### Raw JSON:\n```json\n{json.dumps(result, indent=2)}\n```"

                return [TextContent(type="text", text=message)]
            else:
                message = f"✗ Query failed: {result.get('error', 'Unknown error')}"
                return [TextContent(type="text", text=message)]

        else:
            return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    except Exception as e:
        logging.error(f"Error in tool call '{name}': {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Main entry point for the MCP server."""
    global vector_store

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Local GraphRAG MCP Server")
    parser.add_argument(
        "--index-dir",
        type=str,
        default=".",
        help="Directory for graph database (default: current directory)"
    )
    parser.add_argument(
        "--embed",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers embedding model (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr
    )

    # Initialize vector store
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Initializing GraphRAG vector store at {index_dir}")
    logging.info(f"Embedding model: {args.embed}")

    try:
        vector_store = GraphRAGVectorStore(
            index_dir=index_dir,
            embedding_model_name=args.embed,
            debug=args.debug
        )
        logging.info("GraphRAG MCP Server ready")
    except Exception as e:
        logging.error(f"Failed to initialize vector store: {e}")
        sys.exit(1)

    # Run the MCP server
    asyncio.run(stdio_server(app))


if __name__ == "__main__":
    main()
