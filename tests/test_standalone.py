"""
End-to-end integration test for GraphRAG Vector Store.

This test can be run standalone: python tests/test_standalone.py
"""

import tempfile
import os
from local_graphrag_mcp.server import GraphRAGVectorStore


def test_graph_rag_vector_store():
    """Complete end-to-end test of GraphRAG functionality."""

    print("\n" + "="*70)
    print("GraphRAG Vector Store - End-to-End Integration Test")
    print("="*70 + "\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, 'test_graph.duckdb')

        # ====================================================================
        # Step 1: Initialize
        # ====================================================================
        print("Step 1: Initializing GraphRAG vector store...")
        store = GraphRAGVectorStore(db_path, debug=True)
        print("✓ Vector store initialized\n")

        # ====================================================================
        # Step 2: Add nodes and edges
        # ====================================================================
        print("Step 2: Adding nodes and edges to knowledge graph...")

        nodes = [
            {
                "id": "entity:auth_service",
                "label": "Service",
                "description": "Authentication microservice handling user login and session management",
                "properties": {"language": "Python", "version": "2.1.0"}
            },
            {
                "id": "entity:postgres_db",
                "label": "Database",
                "description": "PostgreSQL database storing user credentials and session data",
                "properties": {"version": "14.0", "port": 5432}
            },
            {
                "id": "entity:alice",
                "label": "Person",
                "description": "Alice Chen, lead engineer responsible for authentication services",
                "properties": {"role": "Lead Engineer", "team": "Platform"}
            },
            {
                "id": "entity:api_gateway",
                "label": "Service",
                "description": "API Gateway routing requests to backend microservices",
                "properties": {"language": "Go", "version": "1.5.0"}
            }
        ]

        edges = [
            {
                "source": "entity:auth_service",
                "target": "entity:postgres_db",
                "label": "CONNECTS_TO",
                "properties": {"protocol": "TCP"}
            },
            {
                "source": "entity:alice",
                "target": "entity:auth_service",
                "label": "MANAGES",
                "properties": {}
            },
            {
                "source": "entity:api_gateway",
                "target": "entity:auth_service",
                "label": "ROUTES_TO",
                "properties": {"path": "/auth"}
            }
        ]

        result = store.add_nodes_and_edges(nodes, edges)

        assert result['success'] is True, f"Failed to add nodes/edges: {result.get('error')}"
        assert result['nodes_added'] == 4, f"Expected 4 nodes, got {result['nodes_added']}"
        assert result['edges_added'] == 3, f"Expected 3 edges, got {result['edges_added']}"

        print(f"✓ Added {result['nodes_added']} nodes and {result['edges_added']} edges\n")

        # ====================================================================
        # Step 3: Query with Graph RAG - Test 1 (Person query)
        # ====================================================================
        print("Step 3: Testing Graph RAG query - 'Who manages the auth service?'")

        query1 = "Who manages the authentication system?"
        result1 = store.query_graph_rag(query1, top_k_anchors=2, hops=1)

        assert result1['success'] is True, f"Query failed: {result1.get('error')}"
        assert len(result1['anchors']) > 0, "No anchors found"

        print(f"✓ Found {len(result1['anchors'])} anchor(s)")
        for anchor in result1['anchors']:
            print(f"  - {anchor['id']}: {anchor['description'][:60]}...")

        # Check that we found relevant entities
        anchor_ids = [a['id'] for a in result1['anchors']]
        subgraph_node_ids = [n['id'] for n in result1['subgraph']['nodes']]

        print(f"✓ Subgraph contains {len(result1['subgraph']['nodes'])} nodes and {len(result1['subgraph']['edges'])} edges\n")

        # ====================================================================
        # Step 4: Query with Graph RAG - Test 2 (Database query)
        # ====================================================================
        print("Step 4: Testing Graph RAG query - 'What database is used?'")

        query2 = "What database does the authentication service use?"
        result2 = store.query_graph_rag(query2, top_k_anchors=3, hops=2)

        assert result2['success'] is True, f"Query failed: {result2.get('error')}"
        assert len(result2['anchors']) > 0, "No anchors found"

        print(f"✓ Found {len(result2['anchors'])} anchor(s)")
        for anchor in result2['anchors']:
            print(f"  - {anchor['id']}: {anchor['description'][:60]}...")

        print(f"✓ Subgraph contains {len(result2['subgraph']['nodes'])} nodes and {len(result2['subgraph']['edges'])} edges\n")

        # ====================================================================
        # Step 5: Test persistence (reload from disk)
        # ====================================================================
        print("Step 5: Testing persistence...")

        # Reload from disk
        store2 = GraphRAGVectorStore(db_path, debug=False)
        print("✓ Reopened vector store from disk")

        # Query again to verify data persisted
        result3 = store2.query_graph_rag("authentication", top_k_anchors=2, hops=1)

        assert result3['success'] is True, "Query after reload failed"
        assert len(result3['anchors']) > 0, "No data found after reload"

        print(f"✓ Data persisted correctly ({len(result3['anchors'])} anchors found)\n")

        # ====================================================================
        # Step 6: Verify graph structure
        # ====================================================================
        print("Step 6: Verifying graph structure...")

        # Check node count in FAISS index
        assert store2.index.ntotal == 4, f"Expected 4 nodes in index, found {store2.index.ntotal}"
        print(f"✓ FAISS index contains {store2.index.ntotal} vectors")

        # Check node count in NetworkX graph
        assert store2.graph.number_of_nodes() == 4, f"Expected 4 nodes in graph, found {store2.graph.number_of_nodes()}"
        print(f"✓ NetworkX graph contains {store2.graph.number_of_nodes()} nodes")

        # Check edge count
        assert store2.graph.number_of_edges() == 3, f"Expected 3 edges, found {store2.graph.number_of_edges()}"
        print(f"✓ NetworkX graph contains {store2.graph.number_of_edges()} edges")

        # Check metadata
        assert len(store2.metadata['nodes']) == 4, f"Expected 4 nodes in metadata, found {len(store2.metadata['nodes'])}"
        print(f"✓ Metadata contains {len(store2.metadata['nodes'])} node records")

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70 + "\n")

        return True


if __name__ == "__main__":
    try:
        test_graph_rag_vector_store()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        exit(1)
