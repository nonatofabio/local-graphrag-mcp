"""Local GraphRAG MCP Server - Graph-based RAG with DuckDB"""

from .server import GraphRAGVectorStore, app, main

__all__ = ["GraphRAGVectorStore", "app", "main"]
__version__ = "0.1.0"
