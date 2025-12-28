#!/usr/bin/env python3
"""
Command-line interface for local-graphrag.

Commands:
- index: Extract entities from documents and add to knowledge graph
- search: Query the knowledge graph using GraphRAG
- list: List all indexed documents/entities
"""

import sys
import os
import json
import argparse
from pathlib import Path
from glob import glob as glob_files
from typing import List, Optional, Dict, Any
from collections import defaultdict
from .server import GraphRAGVectorStore
from .entity_extractor import extract_entities_from_file
from .colors import success, error, info, warning
from .progress import create_file_progress, update_progress_description, progress_print


def find_mcp_config() -> Optional[Path]:
    """
    Find MCP configuration file.

    Search order:
    1. ./.mcp.json (local/project-specific)
    2. ~/.*/mcp.json (user configs like ~/.claude/.mcp.json)
    3. ~/.mcp.json (fallback)

    Returns:
        Path to MCP config file, or None if not found
    """
    # 1. Local config
    local_config = Path('./.mcp.json')
    if local_config.exists():
        return local_config

    # 2. Search home directory for any .*/mcp.json
    home = Path.home()
    for config_path in home.glob('*/mcp.json'):
        # Check it's a hidden directory (starts with .)
        if config_path.parent.name.startswith('.'):
            return config_path

    # 3. Fallback to ~/.mcp.json
    fallback_config = home / '.mcp.json'
    if fallback_config.exists():
        return fallback_config

    return None


def read_mcp_config(config_path: Path) -> Dict[str, Any]:
    """Read and parse MCP configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(warning(f"Failed to read MCP config: {e}"), file=sys.stderr)
        return {}


def get_graphrag_config() -> Dict[str, Any]:
    """
    Get local-graphrag-mcp configuration from MCP config.

    Returns:
        Dict with: index_dir, embed_model, extract_model
    """
    config_path = find_mcp_config()

    if config_path:
        print(info(f"Using MCP config: {config_path}"), file=sys.stderr)
        mcp_config = read_mcp_config(config_path)

        # Extract local-graphrag-mcp server config
        servers = mcp_config.get('mcpServers', {})
        graphrag_config = servers.get('local-graphrag', {})

        if graphrag_config:
            args = graphrag_config.get('args', [])

            # Parse args to extract configuration
            config = {
                'index_dir': './graphrag_index',
                'embed_model': 'all-MiniLM-L6-v2',
                'extract_model': 'claude-3-5-sonnet-20241022'
            }

            # Parse args list
            i = 0
            while i < len(args):
                if args[i] == '--index-dir' and i + 1 < len(args):
                    config['index_dir'] = args[i + 1]
                    i += 2
                elif args[i] == '--embed' and i + 1 < len(args):
                    config['embed_model'] = args[i + 1]
                    i += 2
                elif args[i] == '--extract-model' and i + 1 < len(args):
                    config['extract_model'] = args[i + 1]
                    i += 2
                else:
                    i += 1

            return config

    # No config found - create default local config
    return create_default_config()


def create_default_config() -> Dict[str, Any]:
    """
    Create default .mcp.json in current directory.

    Returns:
        Default configuration dict
    """
    config_path = Path('./.mcp.json')

    default_config = {
        'mcpServers': {
            'local-graphrag': {
                'command': 'local-graphrag-mcp',
                'args': [
                    '--index-dir',
                    './graphrag_index'
                ]
            }
        }
    }

    # Only create if it doesn't exist
    if not config_path.exists():
        print(info(f"Creating default MCP config: {config_path}"), file=sys.stderr)
        with open(config_path, 'w') as f:
            json.dump(default_config, f, indent=2)

    return {
        'index_dir': './graphrag_index',
        'embed_model': 'all-MiniLM-L6-v2',
        'extract_model': 'claude-3-5-sonnet-20241022'
    }


def collect_files(patterns: List[str], recursive: bool = False) -> List[Path]:
    """
    Collect files from patterns and folders.

    Args:
        patterns: List of file paths, glob patterns, or folders
        recursive: Whether to search folders recursively

    Returns:
        List of file paths to index
    """
    files = []

    for pattern in patterns:
        path = Path(pattern)

        # If it's a directory
        if path.is_dir():
            if recursive:
                # Recursively find all files
                for ext in ['*.txt', '*.md', '*.pdf', '*.docx', '*.html', '*.rst', '*.log']:
                    files.extend(path.rglob(ext))
            else:
                # Only files in this directory
                for item in path.iterdir():
                    if item.is_file():
                        files.append(item)

        # If it's a glob pattern
        elif '*' in pattern or '?' in pattern:
            matched = glob_files(pattern, recursive=recursive)
            files.extend([Path(f) for f in matched if Path(f).is_file()])

        # If it's a single file
        elif path.is_file():
            files.append(path)

        else:
            print(warning(f"Path not found: {pattern}"), file=sys.stderr)

    # Remove duplicates and sort
    unique_files = sorted(set(files))
    return unique_files


def cmd_index(args):
    """Extract entities from documents and add to knowledge graph."""
    # Check for API key if using cloud
    use_cloud = getattr(args, 'use_cloud', False)
    if use_cloud and not os.environ.get('ANTHROPIC_API_KEY'):
        print(error("ANTHROPIC_API_KEY environment variable not set"), file=sys.stderr)
        print(info("Set your API key: export ANTHROPIC_API_KEY='your-key-here'"), file=sys.stderr)
        print(info("Or use local extraction (default): remove --use-cloud flag"), file=sys.stderr)
        return 1

    # Get configuration
    config = get_graphrag_config()

    # Collect files to index
    files = collect_files(args.files, recursive=args.recursive)

    if not files:
        print(error("No files found to index"), file=sys.stderr)
        return 1

    # Print appropriate header based on file count
    extraction_method = "cloud (Claude API)" if use_cloud else "local (Ollama)"
    if len(files) > 1:
        print(f"\nExtracting entities from {len(files)} files using {extraction_method}...\n")
    else:
        print(f"\nExtracting entities from {len(files)} file(s) using {extraction_method}...\n")

    # Initialize graph store
    index_dir = Path(config['index_dir']).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist

    # Check if index already exists
    nodes_index_path = index_dir / "nodes.index"
    if nodes_index_path.exists():
        print(info(f"Adding to existing graph at: {index_dir}"))
    else:
        print(info(f"Creating new graph at: {index_dir}"))

    graph_store = GraphRAGVectorStore(
        index_dir=str(index_dir),
        embedding_model_name=config['embed_model']
    )

    # Index each file
    success_count = 0
    fail_count = 0
    total_nodes = 0
    total_edges = 0

    # Create progress bar wrapper for files
    files_iter, show_progress = create_file_progress(files, desc="Indexing")

    for file_path in files_iter:
        try:
            # Update progress bar with current filename
            update_progress_description(files_iter, file_path, "Indexing")
            progress_print(f"📄 Indexing: {file_path}", show_progress)

            # Extract entities from document
            extraction_result = extract_entities_from_file(
                file_path,
                use_cloud=use_cloud,
                model=args.extract_model if hasattr(args, 'extract_model') else None
            )

            nodes = extraction_result['nodes']
            edges = extraction_result['edges']

            # Add source metadata to all nodes
            for node in nodes:
                if 'properties' not in node:
                    node['properties'] = {}
                node['properties']['source'] = str(file_path)

            # Add to graph
            result = graph_store.add_nodes_and_edges(nodes, edges)

            if result["success"]:
                nodes_added = result["nodes_added"]
                edges_added = result["edges_added"]
                total_nodes += nodes_added
                total_edges += edges_added
                progress_print(
                    f"   {success(f'Added {nodes_added} nodes and {edges_added} edges')}",
                    show_progress
                )
                success_count += 1
            else:
                err_msg = result.get("error", "Unknown error")
                progress_print(f"   {error(f'Failed: {err_msg}')}", show_progress)
                fail_count += 1

        except Exception as e:
            err_str = str(e)
            progress_print(f"   {error(f'Error: {err_str}')}", show_progress)
            fail_count += 1

    print(f"\n{'='*60}")
    print("Indexing complete!")
    if success_count > 0:
        print(f"  {success(f'Success: {success_count} file(s)')}")
        print(f"  {info(f'Total: {total_nodes} nodes, {total_edges} edges added')}")
    if fail_count > 0:
        print(f"  {error(f'Failed: {fail_count} file(s)')}")
    print(info(f"Total nodes in graph: {graph_store.index.ntotal}"))
    print(info(f"Total edges in graph: {graph_store.graph.number_of_edges()}"))
    print(info(f"Index location: {index_dir}"))
    print(f"{'='*60}")

    return 0 if fail_count == 0 else 1


def cmd_search(args):
    """Query the knowledge graph using GraphRAG."""
    # Get configuration
    config = get_graphrag_config()

    # Initialize graph store
    index_dir = Path(config['index_dir']).resolve()
    nodes_index_path = index_dir / "nodes.index"

    if not nodes_index_path.exists():
        print(error(f"No graph found at {index_dir}"), file=sys.stderr)
        print(info("Run 'local-graphrag index <files>' first to create a graph"), file=sys.stderr)
        return 1

    graph_store = GraphRAGVectorStore(
        index_dir=str(index_dir),
        embedding_model_name=config['embed_model']
    )

    # Perform Graph RAG query
    result = graph_store.query_graph_rag(
        args.query,
        top_k_anchors=args.top_k,
        hops=args.hops
    )

    if not result['success']:
        print(error(f"Query failed: {result.get('error')}"), file=sys.stderr)
        return 1

    anchors = result['anchors']
    subgraph = result['subgraph']

    if not anchors:
        print(info("No results found."))
        return 0

    print(f"\n{'='*60}")
    print(f"Query: {args.query}")
    print(f"{'='*60}\n")

    # Print anchor nodes
    print(f"Found {len(anchors)} anchor node(s):\n")
    for i, anchor in enumerate(anchors, 1):
        print(f"{i}. {anchor['id']} ({anchor['label']})")
        print(f"   Similarity: {anchor['similarity_score']:.4f}")
        print(f"   {anchor['description'][:200]}...")
        print()

    # Print subgraph summary
    print(f"{'='*60}")
    print(f"Subgraph contains:")
    print(f"  - {len(subgraph['nodes'])} nodes")
    print(f"  - {len(subgraph['edges'])} edges")
    print(f"{'='*60}\n")

    # Print all nodes in subgraph
    print("Nodes in subgraph:\n")
    for node in subgraph['nodes']:
        print(f"  • {node['id']} ({node['label']}): {node['description'][:100]}...")

    print()

    # Print all edges in subgraph
    if subgraph['edges']:
        print("Relationships:\n")
        for edge in subgraph['edges']:
            print(f"  • {edge['source']} --[{edge['label']}]--> {edge['target']}")

    print()

    return 0


def cmd_list(args):
    """List all indexed documents/entities in the graph."""
    # Get configuration
    config = get_graphrag_config()

    # Build index dir path
    index_dir = Path(config['index_dir']).resolve()
    metadata_path = index_dir / "metadata.json"

    # Check if metadata exists
    if not metadata_path.exists():
        if args.json:
            print(json.dumps({"nodes": [], "sources": [], "total_nodes": 0}))
        else:
            print(info("No entities indexed yet."))
            print(info("Run 'local-graphrag index <files>' to index documents."))
        return 0

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(error(f"Failed to read metadata: {e}"), file=sys.stderr)
        return 1

    nodes = metadata.get('nodes', [])

    if not nodes:
        if args.json:
            print(json.dumps({"nodes": [], "sources": [], "total_nodes": 0}))
        else:
            print(info("No entities indexed yet."))
            print(info("Run 'local-graphrag index <files>' to index documents."))
        return 0

    # Group by source
    source_nodes = defaultdict(list)
    for node in nodes:
        source = node.get('properties', {}).get('source', 'unknown')
        source_nodes[source].append(node)

    # Sort sources alphabetically
    sorted_sources = sorted(source_nodes.keys())

    if args.json:
        # JSON output
        output = {
            "sources": [
                {
                    "source": source,
                    "node_count": len(source_nodes[source]),
                    "nodes": source_nodes[source]
                }
                for source in sorted_sources
            ],
            "total_nodes": len(nodes)
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print(f"\nIndexed Documents ({len(sorted_sources)} total):\n")
        print("="*60)

        for i, source in enumerate(sorted_sources, 1):
            node_count = len(source_nodes[source])
            print(f"\n{i}. {source}")
            print(f"   Nodes: {node_count}")

            # Show first few node IDs
            node_ids = [n['id'] for n in source_nodes[source]]
            if len(node_ids) <= 3:
                print(f"   Entities: {', '.join(node_ids)}")
            else:
                print(f"   Entities: {', '.join(node_ids[:3])}, ... (+{len(node_ids)-3} more)")

        print("\n" + "="*60)
        print(info(f"Total entities: {len(nodes)}"))
        print(info(f"Index location: {index_dir}"))

    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Local GraphRAG knowledge graph CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The CLI uses configuration from MCP config files in this order:
  1. ./.mcp.json (local/project-specific)
  2. ~/.claude/.mcp.json (Claude Code config)
  3. ~/.mcp.json (fallback)

If no config exists, creates ./.mcp.json with default settings.

Examples:
  # Index single file (uses local Ollama by default)
  local-graphrag index document.pdf

  # Index multiple files
  local-graphrag index doc1.pdf doc2.txt doc3.md

  # Index all files in a folder recursively
  local-graphrag index -r documents/

  # Index with specific Ollama model
  local-graphrag index --extract-model llama3.1:8b document.pdf

  # Index using Claude API (cloud, requires ANTHROPIC_API_KEY)
  export ANTHROPIC_API_KEY='your-key-here'
  local-graphrag index --use-cloud document.pdf

  # Index with cloud and custom model
  local-graphrag index --use-cloud --extract-model claude-opus-4-5 doc.pdf

  # Query the graph
  local-graphrag search "What database does the auth service use?"

  # Query with more anchors and deeper traversal
  local-graphrag search -k 5 --hops 3 "Who manages the infrastructure?"

  # List indexed documents
  local-graphrag list
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Index command
    index_parser = subparsers.add_parser('index', help='Extract entities and index documents')
    index_parser.add_argument(
        'files',
        nargs='+',
        help='Files, folders, or glob patterns to index'
    )
    index_parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively search folders for documents'
    )
    index_parser.add_argument(
        '--use-cloud',
        action='store_true',
        help='Use Claude API for extraction instead of local Ollama (requires ANTHROPIC_API_KEY)'
    )
    index_parser.add_argument(
        '--extract-model',
        type=str,
        help='Model for entity extraction (default: llama3.2:3b for local, claude-3-5-sonnet for cloud)'
    )

    # Search command
    search_parser = subparsers.add_parser('search', help='Query the knowledge graph')
    search_parser.add_argument(
        'query',
        type=str,
        help='Natural language search query'
    )
    search_parser.add_argument(
        '-k', '--top-k',
        type=int,
        default=3,
        dest='top_k',
        help='Number of anchor nodes to find (default: 3)'
    )
    search_parser.add_argument(
        '--hops',
        type=int,
        default=2,
        help='Number of relationship hops for graph traversal (default: 2)'
    )

    # List command
    list_parser = subparsers.add_parser('list', help='List all indexed documents/entities')
    list_parser.add_argument(
        '--json',
        action='store_true',
        help='Output in JSON format for machine-readable output'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'index':
        return cmd_index(args)
    elif args.command == 'search':
        return cmd_search(args)
    elif args.command == 'list':
        return cmd_list(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
