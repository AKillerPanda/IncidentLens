"""
End-to-end pipeline integration test (no Elasticsearch required).

Validates that the full pipeline code path works from synthetic data through
graph building, feature extraction, GNN encoding, to the agent tool dispatch —
all with mocked ES calls.
"""

import json
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.Backend.graph_data_wrapper import build_sliding_window_graphs
from src.Backend.temporal_gnn import (
    preprocess_graph,
    preprocess_graphs,
    sanitize_graph,
    normalize_features_global,
    apply_normalization,
    EvolvingGNN,
)
from src.Backend.gnn_interface import BaseGNNEncoder, create_dataloaders, compute_class_weights
from src.Backend.graph import network, build_sample_graph, build_snapshot_dataset
from src.Backend.csv_to_json import _safe_val, dataframe_to_ndjson_chunks


# ═══════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════

def make_synthetic_packets(n=200, seed=42):
    """Create a realistic synthetic packet DataFrame."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp": np.sort(rng.uniform(0, 20, n)),
        "src_ip": [f"10.0.0.{rng.integers(1, 6)}" for _ in range(n)],
        "dst_ip": [f"192.168.1.{rng.integers(1, 6)}" for _ in range(n)],
        "protocol": rng.choice([6, 17], n),
        "packet_length": rng.integers(60, 1500, n),
        "payload_length": rng.integers(0, 1400, n),
        "dst_port": rng.choice([80, 443, 53, 1900], n),
        "label": rng.choice([0, 1], n, p=[0.8, 0.2]),
    })


# ═══════════════════════════════════════════════
# E2E: CSV → NDJSON → Graphs → Preprocess → GNN
# ═══════════════════════════════════════════════

class TestEndToEndPipeline:
    """Full pipeline without ES — validates all code paths connect."""

    def test_csv_to_graphs_to_gnn(self, tmp_path):
        """
        1. Create synthetic CSV data
        2. Write to NDJSON
        3. Load from NDJSON
        4. Build sliding window graphs
        5. Preprocess graphs
        6. Normalize features
        7. Run through GNN model
        8. Verify output shapes
        """
        # Step 1-2: Create and write data
        df = make_synthetic_packets(200)
        files = dataframe_to_ndjson_chunks(df, str(tmp_path), chunk_size=100)
        assert len(files) == 2

        # Step 3: Load back from NDJSON
        records = []
        for fpath in files:
            with open(fpath, "r") as f:
                for line in f:
                    records.append(json.loads(line))
        df_loaded = pd.DataFrame(records)
        assert len(df_loaded) == 200
        assert "timestamp" in df_loaded.columns

        # Step 4: Build graphs
        graphs, id_to_ip = build_sliding_window_graphs(
            df_loaded, window_size=5.0, stride=2.5,
        )
        assert len(graphs) > 0
        assert isinstance(id_to_ip, dict)

        # Verify graph properties
        for g in graphs:
            assert hasattr(g, "edge_index")
            assert hasattr(g, "x")
            assert hasattr(g, "edge_attr")
            assert hasattr(g, "y")
            assert hasattr(g, "window_id")
            assert g.x.shape[1] == 6  # node features
            assert g.edge_attr.shape[1] == 5  # edge features
            assert g.num_nodes == g.x.shape[0]

        # Verify unique window IDs
        wids = [g.window_id for g in graphs]
        assert len(set(wids)) == len(wids), "window_ids must be unique"

        # Step 5: Sanitize + preprocess
        sanitized = [sanitize_graph(g) for g in graphs]
        preprocessed = preprocess_graphs(sanitized)
        assert len(preprocessed) == len(graphs)

        # Step 6: Normalize (returns (graphs, stats_dict) tuple; modifies in-place)
        normalized, stats = normalize_features_global(preprocessed)
        assert isinstance(stats, dict)
        assert "node_mean" in stats
        assert "edge_mean" in stats
        # normalized IS preprocessed (in-place), verify no NaN
        for g in normalized:
            assert not torch.isnan(g.x).any(), "NaN in node features after normalization"

        # Step 7: GNN forward pass
        node_dim = normalized[0].x.shape[1]
        edge_dim = normalized[0].edge_attr.shape[1] if normalized[0].edge_attr is not None else 5
        model = EvolvingGNN(input_dim=node_dim, hidden_dim=16, edge_feat_dim=edge_dim)
        model.eval()

        # Use last few graphs as a sequence
        seq_len = min(3, len(normalized))
        sequence = normalized[-seq_len:]
        with torch.no_grad():
            logits = model(sequence)

        last_graph = sequence[-1]
        expected_edges = last_graph.edge_index.shape[1]
        assert logits.shape == (expected_edges,), (
            f"Logits shape {logits.shape} != expected ({expected_edges},)"
        )
        assert not torch.isnan(logits).any()

    def test_graph_algebraic_to_pyg(self):
        """Verify graph.py algebraic graph → PyG Data pipeline."""
        g = build_sample_graph()
        assert g.num_nodes == 4

        # Adjacency and degree
        adj = g.build_sparse_adjacency()
        assert adj.shape == (4, 4)
        out_deg = g.out_degree()
        assert out_deg.shape[0] == 4

        # Convert to PyG
        data = g.to_pyg_data()
        assert data.edge_index.shape[1] == 5
        assert data.num_nodes == 4

    def test_build_snapshot_dataset(self):
        """Verify graph.py's snapshot-based graph building."""
        df = pd.DataFrame({
            "timestamp": [0, 1, 2, 3, 4, 5, 6, 7],
            "src_ip": ["A", "B", "A", "C", "A", "B", "C", "A"],
            "dst_ip": ["B", "A", "C", "A", "B", "C", "A", "B"],
            "protocol": ["TCP"] * 8,
            "packet_length": [100] * 8,
            "payload_length": [50] * 8,
            "label": [0, 0, 1, 0, 1, 0, 0, 1],
        })
        data_list, node_map, flows_df = build_snapshot_dataset(df, delta_t=4.0)
        assert len(data_list) > 0
        assert len(node_map) == 3  # A, B, C

    def test_class_weights_from_graphs(self):
        """Verify class weight computation from realistic graphs."""
        df = make_synthetic_packets(100)
        graphs, _ = build_sliding_window_graphs(df)
        w = compute_class_weights(graphs)
        assert w.shape[0] >= 2
        assert not torch.isnan(w).any()
        assert not torch.isinf(w).any()

    def test_dataloaders_from_graphs(self):
        """Verify DataLoader creation works with realistic graphs."""
        df = make_synthetic_packets(300, seed=99)
        graphs, _ = build_sliding_window_graphs(df, window_size=3.0, stride=1.5)
        if len(graphs) < 3:
            pytest.skip("Not enough graphs")

        train, val, test, info = create_dataloaders(graphs, batch_size=4)
        assert info["total"] == len(graphs)
        assert info["train"] > 0

    def test_network_object_attached_to_graphs(self):
        """Every graph from build_sliding_window_graphs should have a .network."""
        df = make_synthetic_packets(50)
        graphs, _ = build_sliding_window_graphs(df)
        for g in graphs:
            assert hasattr(g, "network")
            assert isinstance(g.network, network)


# ═══════════════════════════════════════════════
# E2E: Agent tool dispatch (mocked ES)
# ═══════════════════════════════════════════════

class TestAgentToolsE2E:
    """Verify agent_tools.dispatch returns valid JSON for all tools."""

    @patch("src.Backend.wrappers.health_check")
    def test_dispatch_health(self, mock_health):
        import src.Backend.agent_tools as at
        mock_health.return_value = {"status": "green", "number_of_nodes": 1}
        result = json.loads(at.dispatch("es_health_check", {}))
        assert result["cluster_status"] == "green"

    @patch("src.Backend.wrappers.search_flows")
    def test_dispatch_search_flows(self, mock_search):
        import src.Backend.agent_tools as at
        mock_search.return_value = [{"_id": "f1", "label": 1}]
        result = json.loads(at.dispatch("search_flows", {"label": 1}))
        assert result["count"] == 1

    @patch("src.Backend.wrappers.get_flow")
    def test_dispatch_get_flow(self, mock_get):
        import src.Backend.agent_tools as at
        mock_get.return_value = {"src_ip": "10.0.0.1", "dst_ip": "10.0.0.2"}
        result = json.loads(at.dispatch("get_flow", {"flow_id": "test123"}))
        assert result["src_ip"] == "10.0.0.1"

    def test_dispatch_unknown_tool(self):
        import src.Backend.agent_tools as at
        result = json.loads(at.dispatch("totally_fake_tool", {}))
        assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
