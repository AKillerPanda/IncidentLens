"""
Tests for csv_to_json.py — CSV→NDJSON conversion pipeline.

Covers: _safe_val, load_and_merge, dataframe_to_ndjson_chunks,
        write_metadata, convert (end-to-end).
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from src.Backend.csv_to_json import (
    _safe_val,
    load_and_merge,
    dataframe_to_ndjson_chunks,
    write_metadata,
    convert,
)


# ═══════════════════════════════════════════════
# _safe_val — JSON-safe scalar conversion
# ═══════════════════════════════════════════════

class TestSafeVal:

    def test_none(self):
        assert _safe_val(None) is None

    def test_nan(self):
        assert _safe_val(float("nan")) is None

    def test_inf(self):
        assert _safe_val(float("inf")) is None

    def test_neg_inf(self):
        assert _safe_val(float("-inf")) is None

    def test_np_int(self):
        result = _safe_val(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    def test_np_float(self):
        result = _safe_val(np.float32(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 0.01

    def test_np_bool(self):
        result = _safe_val(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_string_passthrough(self):
        assert _safe_val("hello") == "hello"

    def test_regular_int_passthrough(self):
        assert _safe_val(42) == 42

    def test_regular_float_passthrough(self):
        assert _safe_val(3.14) == 3.14


# ═══════════════════════════════════════════════
# load_and_merge — CSV loading + label merge
# ═══════════════════════════════════════════════

class TestLoadAndMerge:

    def test_merge_normal(self, tmp_path):
        """Normal merge with matching packet_index."""
        packets = pd.DataFrame({
            "packet_index": [0, 1, 2, 3, 4],
            "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "src_ip": ["10.0.0.1"] * 5,
            "dst_ip": ["10.0.0.2"] * 5,
            "packet_length": [100, 200, 300, 400, 500],
        })
        labels = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2, 3, 4],
            "x": [0, 0, 1, 0, 1],
        })
        pkts_path = str(tmp_path / "packets.csv")
        labels_path = str(tmp_path / "labels.csv")
        packets.to_csv(pkts_path, index=False)
        labels.to_csv(labels_path, index=False)

        merged = load_and_merge(pkts_path, labels_path)
        assert len(merged) == 5
        assert "label" in merged.columns
        assert merged["label"].sum() == 2  # two malicious

    def test_merge_missing_labels_fill_zero(self, tmp_path):
        """Packets without a matching label get label=0."""
        packets = pd.DataFrame({
            "packet_index": [0, 1, 2],
            "timestamp": [1.0, 2.0, 3.0],
        })
        labels = pd.DataFrame({
            "Unnamed: 0": [0],  # only label for packet 0
            "x": [1],
        })
        pkts_path = str(tmp_path / "packets.csv")
        labels_path = str(tmp_path / "labels.csv")
        packets.to_csv(pkts_path, index=False)
        labels.to_csv(labels_path, index=False)

        merged = load_and_merge(pkts_path, labels_path)
        assert merged.loc[merged["packet_index"] == 0, "label"].iloc[0] == 1
        assert merged.loc[merged["packet_index"] == 1, "label"].iloc[0] == 0
        assert merged.loc[merged["packet_index"] == 2, "label"].iloc[0] == 0


# ═══════════════════════════════════════════════
# dataframe_to_ndjson_chunks
# ═══════════════════════════════════════════════

class TestNdjsonChunks:

    def test_single_chunk(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        files = dataframe_to_ndjson_chunks(df, str(tmp_path), chunk_size=10)
        assert len(files) == 1
        # Read back
        with open(files[0], "r") as f:
            lines = f.readlines()
        assert len(lines) == 3
        record = json.loads(lines[0])
        assert record["a"] == 1

    def test_multiple_chunks(self, tmp_path):
        df = pd.DataFrame({"x": range(10)})
        files = dataframe_to_ndjson_chunks(df, str(tmp_path), chunk_size=3)
        assert len(files) == 4  # 3+3+3+1
        # Verify last file has 1 record
        with open(files[-1], "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_nan_values_become_null(self, tmp_path):
        df = pd.DataFrame({"val": [1.0, float("nan"), 3.0]})
        files = dataframe_to_ndjson_chunks(df, str(tmp_path), chunk_size=10)
        with open(files[0], "r") as f:
            lines = f.readlines()
        record = json.loads(lines[1])
        assert record["val"] is None

    def test_empty_dataframe(self, tmp_path):
        df = pd.DataFrame({"a": []})
        files = dataframe_to_ndjson_chunks(df, str(tmp_path), chunk_size=10)
        assert len(files) == 0  # ceil(0/10) = 0

    def test_creates_directory(self, tmp_path):
        nested = str(tmp_path / "sub" / "dir")
        df = pd.DataFrame({"a": [1]})
        files = dataframe_to_ndjson_chunks(df, nested, chunk_size=10)
        assert os.path.exists(nested)
        assert len(files) == 1


# ═══════════════════════════════════════════════
# write_metadata
# ═══════════════════════════════════════════════

class TestWriteMetadata:

    def test_metadata_written(self, tmp_path):
        path = write_metadata(str(tmp_path), total_rows=100, chunk_size=50,
                              files=["a.json", "b.json"])
        assert os.path.exists(path)
        with open(path) as f:
            meta = json.load(f)
        assert meta["total_rows"] == 100
        assert meta["num_files"] == 2
        assert "created_at" in meta


# ═══════════════════════════════════════════════
# convert (end-to-end)
# ═══════════════════════════════════════════════

class TestConvert:

    def test_full_pipeline(self, tmp_path):
        """End-to-end: create CSVs → convert → verify NDJSON output."""
        packets = pd.DataFrame({
            "packet_index": range(20),
            "timestamp": np.linspace(0, 10, 20),
            "src_ip": ["10.0.0.1"] * 20,
            "dst_ip": ["10.0.0.2"] * 20,
            "protocol": [6] * 20,
            "packet_length": [100] * 20,
            "payload_length": [50] * 20,
        })
        labels = pd.DataFrame({
            "Unnamed: 0": range(20),
            "x": [0] * 15 + [1] * 5,
        })
        pkts_path = str(tmp_path / "packets.csv")
        labels_path = str(tmp_path / "labels.csv")
        outdir = str(tmp_path / "output")
        packets.to_csv(pkts_path, index=False)
        labels.to_csv(labels_path, index=False)

        result = convert(pkts_path, labels_path, outdir, chunk_size=10, max_rows=None)
        assert result["total_rows"] == 20
        assert result["num_files"] == 2  # 20 / 10 = 2 chunks
        assert os.path.exists(os.path.join(outdir, "metadata.json"))

    def test_max_rows_cap(self, tmp_path):
        packets = pd.DataFrame({
            "packet_index": range(50),
            "timestamp": range(50),
        })
        labels = pd.DataFrame({
            "Unnamed: 0": range(50),
            "x": [0] * 50,
        })
        pkts_path = str(tmp_path / "packets.csv")
        labels_path = str(tmp_path / "labels.csv")
        outdir = str(tmp_path / "output")
        packets.to_csv(pkts_path, index=False)
        labels.to_csv(labels_path, index=False)

        result = convert(pkts_path, labels_path, outdir, chunk_size=100, max_rows=10)
        assert result["total_rows"] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
