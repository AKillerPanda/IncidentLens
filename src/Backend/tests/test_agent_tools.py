"""
Tests for agent_tools.py — Tool registry, dispatch, sanitization, caching.

These tests mock wrappers.* calls (no ES dependency) and validate the tool
logic, schema integrity, and dispatch mechanism.
"""

import json
import math
import os
import sys
import time
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
import src.Backend.agent_tools as agent_tools


# ═══════════════════════════════════════════════
# TOOL REGISTRY INTEGRITY
# ═══════════════════════════════════════════════

class TestToolRegistry:

    def test_registry_not_empty(self):
        assert len(agent_tools._REGISTRY) > 0

    def test_all_tools_callable(self):
        for name, fn in agent_tools._REGISTRY.items():
            assert callable(fn), f"Tool {name} is not callable"

    def test_tool_schemas_match_registry(self):
        """Every tool in TOOL_SCHEMAS must be registered, and vice versa."""
        schema_names = {s["function"]["name"] for s in agent_tools.TOOL_SCHEMAS}
        registry_names = set(agent_tools._REGISTRY.keys())
        assert schema_names == registry_names, (
            f"Mismatch — in schemas only: {schema_names - registry_names}, "
            f"in registry only: {registry_names - schema_names}"
        )

    def test_schema_required_fields(self):
        """Each schema must have type, function.name, function.description, function.parameters."""
        for schema in agent_tools.TOOL_SCHEMAS:
            assert "function" in schema
            fn = schema["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            params = fn["parameters"]
            assert params.get("type") == "object"

    def test_list_tools(self):
        tools = agent_tools.list_tools()
        assert len(tools) > 0
        for t in tools:
            assert isinstance(t, str)


# ═══════════════════════════════════════════════
# _sanitize_for_json
# ═══════════════════════════════════════════════

class TestSanitizeForJson:

    def test_nan_to_none(self):
        result = agent_tools._sanitize_for_json({"a": float("nan")})
        assert result["a"] is None

    def test_inf_to_none(self):
        result = agent_tools._sanitize_for_json({"a": float("inf"), "b": float("-inf")})
        assert result["a"] is None
        assert result["b"] is None

    def test_np_types(self):
        result = agent_tools._sanitize_for_json({
            "i": np.int64(42),
            "f": np.float32(3.14),
            "b": np.bool_(True),
        })
        assert result["i"] == 42
        assert isinstance(result["i"], int)
        assert isinstance(result["f"], float)
        assert result["b"] is True

    def test_nested_dict(self):
        result = agent_tools._sanitize_for_json({
            "outer": {"inner": float("nan")}
        })
        assert result["outer"]["inner"] is None

    def test_list_in_dict(self):
        result = agent_tools._sanitize_for_json({
            "vals": [1, float("nan"), np.int64(3)]
        })
        assert result["vals"] == [1, None, 3]

    def test_string_passthrough(self):
        result = agent_tools._sanitize_for_json({"s": "hello"})
        assert result["s"] == "hello"

    def test_none_passthrough(self):
        result = agent_tools._sanitize_for_json({"x": None})
        assert result["x"] is None


# ═══════════════════════════════════════════════
# DISPATCH
# ═══════════════════════════════════════════════

class TestDispatch:

    @patch.dict(agent_tools._REGISTRY, {
        "test_tool": lambda **kw: {"result": kw.get("x", 0) * 2},
    })
    def test_dispatch_normal(self):
        result = agent_tools.dispatch("test_tool", {"x": 5})
        parsed = json.loads(result)
        assert parsed["result"] == 10

    def test_dispatch_unknown_tool(self):
        result = agent_tools.dispatch("nonexistent_tool_xyz", {})
        parsed = json.loads(result)
        assert "error" in parsed

    @patch.dict(agent_tools._REGISTRY, {
        "failing_tool": lambda **kw: (_ for _ in ()).throw(ValueError("boom")),
    })
    def test_dispatch_exception(self):
        result = agent_tools.dispatch("failing_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed


# ═══════════════════════════════════════════════
# INDIVIDUAL TOOL MOCKED TESTS
# ═══════════════════════════════════════════════

class TestToolHealthCheck:

    @patch("src.Backend.wrappers.health_check")
    def test_health_check(self, mock_health):
        mock_health.return_value = {"status": "green", "number_of_nodes": 1}
        result = agent_tools._REGISTRY["es_health_check"]()
        assert result["cluster_status"] == "green"
        assert result["number_of_nodes"] == 1


class TestToolSearchFlows:

    @patch("src.Backend.wrappers.search_flows")
    def test_search_no_filters(self, mock_search):
        mock_search.return_value = [{"_id": "1", "label": 0}]
        result = agent_tools._REGISTRY["search_flows"]()
        assert result["count"] == 1
        # match_all query
        mock_search.assert_called_once()
        call_query = mock_search.call_args[1]["query"]
        assert "match_all" in call_query

    @patch("src.Backend.wrappers.search_flows")
    def test_search_with_filters(self, mock_search):
        mock_search.return_value = []
        result = agent_tools._REGISTRY["search_flows"](label=1, src_ip="10.0.0.1")
        assert result["count"] == 0
        call_query = mock_search.call_args[1]["query"]
        assert "bool" in call_query


class TestToolGetFlow:

    @patch("src.Backend.wrappers.get_flow")
    def test_flow_found(self, mock_get):
        mock_get.return_value = {"src_ip": "10.0.0.1"}
        result = agent_tools._REGISTRY["get_flow"](flow_id="abc")
        assert result["src_ip"] == "10.0.0.1"

    @patch("src.Backend.wrappers.get_flow")
    def test_flow_not_found(self, mock_get):
        mock_get.return_value = None
        result = agent_tools._REGISTRY["get_flow"](flow_id="missing")
        assert "error" in result


class TestToolDetect:

    @patch("src.Backend.wrappers.search_flows")
    def test_label_method(self, mock_search):
        mock_search.return_value = [{"_id": "1", "label": 1}]
        result = agent_tools._REGISTRY["detect_anomalies"](method="label")
        assert result["method"] == "label"
        assert result["count"] == 1

    @patch("src.Backend.wrappers.search_anomalous_flows")
    def test_score_method(self, mock_search):
        mock_search.return_value = [{"_id": "2", "prediction_score": 0.9}]
        result = agent_tools._REGISTRY["detect_anomalies"](method="score", threshold=0.8)
        assert result["method"] == "prediction_score"
        assert result["count"] == 1

    @patch("src.Backend.wrappers.search_flows")
    @patch("src.Backend.wrappers.feature_stats_by_label")
    def test_stats_method(self, mock_stats, mock_search):
        mock_stats.return_value = {
            "packet_count": {"label_0": {"avg": 10.0, "std_deviation": 5.0}},
        }
        mock_search.return_value = []
        result = agent_tools._REGISTRY["detect_anomalies"](method="stats")
        assert result["method"] == "statistical"
        assert result["cutoff"] == 20.0  # 10 + 2*5


class TestToolFeatureStats:

    @patch("src.Backend.wrappers.feature_stats_by_label")
    def test_returns_stats(self, mock_stats):
        mock_stats.return_value = {"packet_count": {"label_0": {"avg": 10}}}
        result = agent_tools._REGISTRY["feature_stats"]()
        assert "packet_count" in result


# ═══════════════════════════════════════════════
# STATS CACHE
# ═══════════════════════════════════════════════

class TestStatsCache:

    def test_cache_structure(self):
        """The stats cache dict should have the expected keys."""
        assert "data" in agent_tools._STATS_CACHE
        assert "ts" in agent_tools._STATS_CACHE


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
