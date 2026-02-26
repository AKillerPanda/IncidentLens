# IncidentLens — Backend

Developer reference for the Python backend powering IncidentLens.

---

## Module Map

| Module | Responsibility |
|:-------|:---------------|
| **main.py** | CLI shim — imports `main()` from `tests/testingentry.py` (ensures `src.Backend.*` path resolution) |
| **tests/testingentry.py** | Actual CLI implementation — `health`, `ingest`, `investigate`, `serve`, `convert` |
| **agent.py** | LLM agent orchestrator: multi-step reasoning loop with OpenAI tool-calling. After processing tool calls the loop continues automatically (no redundant `finish_reason` check) |
| **agent_tools.py** | 19 tools exposed to the agent (ES queries, counterfactuals, severity, graph analysis, ML anomalies, severity breakdown, CF search). Includes thread-safe `_STATS_CACHE` (30 s TTL, `threading.Lock`), `_GRAPH_CACHE`, `set_graph_cache()`, and `_sanitize_for_json()` helper (handles `np.bool_`, `np.integer`, `np.floating`, NaN/Inf). Validates window indices with negative-value rejection. Severity z-score computation fully numpy-vectorised |
| **server.py** | FastAPI server (v0.2.0) — 21 REST endpoints + 1 WebSocket. Includes 7 ES-native analytics endpoints (severity breakdown, runtime severity search, paginated search, CF text search, composite aggregation, ML anomalies, ML influencers). Proper HTTP error codes (502/404). Uses `asyncio.to_thread` for non-blocking ES calls. Thread-safe singletons (`_get_agent`, `_cached_detect` via `threading.Lock` with TOCTOU-safe read-compute-write inside lock). Aggregate endpoint has field whitelist. CORS origins read from `INCIDENTLENS_CORS_ORIGINS` env var. `reload` conditional on `INCIDENTLENS_DEV` env var (off by default for production). Also runnable directly via `python server.py` (`if __name__` block uses `src.Backend.server:app` import path) |
| **wrappers.py** | Singleton ES client with thread-safe initialization (`threading.Lock`) (1 962 lines, 53 functions). Index management, bulk flow/embedding ingestion, kNN search, counterfactual diff, ML jobs. PIT leak protection (try/finally on search_with_pagination). New ES-native features: ILM policies, ingest pipelines (NaN cleanup), index templates, runtime severity fields (Painless), search_after + PIT pagination, composite aggregations, full-text CF search. `generate_embeddings()` auto-corrects `embedding_dim` to match actual GNN output dimensions. Fallback IPs use valid 10.x.x.x encoding for large node IDs |
| **graph_data_wrapper.py** | Vectorised sliding-window graph builder — pure numpy, zero Python for-loops over packets. Counterfactual tools use batch `(T,F)` matrix ops and sum-of-squares update formulas. Window finding uses `np.argmax`/`np.argmin` |
| **graph.py** | Core graph data structures (`node`, `network`), snapshot dataset builder. `build_edge_index()` uses numpy pre-allocated arrays. `build_window_data()` uses `argsort`+`searchsorted` for vectorised window splitting with `window_start` temporal field |
| **train.py** | EdgeGNN (GraphSAGE + Edge MLP) training pipeline with class-imbalance handling |
| **temporal_gnn.py** | EvolveGCN-O — semi-temporal GNN with LSTM-evolved weights for sequence-level detection. Neural ODE variant (`EvolvingGNN_ODE`) with RK4 default. `_preprocess_single()` follows training pipeline order: sanitize → recompute_node_features → normalize → preprocess. `recompute_node_features()` initialises 6-dim zero-valued features when edge_attr is insufficient. Checkpoint loading uses `weights_only=True` for security |
| **gnn_interface.py** | `BaseGNNEncoder` abstract class — the contract any GNN must satisfy to plug into the pipeline. Checkpoint loading validates dimensions and uses `weights_only=True`. `compute_class_weights()` correctly handles unobserved classes |
| **ingest_pipeline.py** | 8-step data pipeline: load NDJSON → build graphs → index flows → embeddings → counterfactuals. Defines `RAW_PACKETS_INDEX` and `RAW_PACKETS_MAPPING`. Refreshes `incidentlens-*` indices (not `_all`) after indexing |
| **csv_to_json.py** | Converts raw CSV datasets to chunked NDJSON for the ingest pipeline. `_safe_val` handles NaN, Inf, and numpy scalar types |
| **GNN.py** | *Deprecated* — standalone EdgeGNN duplicate kept for reference only (canonical implementation: `train.py`). Safe to delete. |
| **backup/** | `temporal_gnn_v1_backup.py` — earlier temporal GNN version kept for reference |

> `__init__.py` files exist in `Backend/`, `Backend/backup/`, and `Backend/tests/` for package resolution.

---

## Data Flow

```
Raw CSV (Kitsune SSDP Flood)
    │
    ▼  csv_to_json.py
NDJSON chunks (data/packets_*.json)
    │
    ▼  ingest_pipeline.load_ndjson_files()
pandas DataFrame
    │
    ▼  graph_data_wrapper.build_sliding_window_graphs()
list[PyG Data]  ─────────────────────────────────────────┐
    │                                                     │
    ├─▶ wrappers.index_graphs_bulk()                      │
    │       → ES: incidentlens-flows                      │
    │                                                     │
    ├─▶ wrappers.generate_embeddings()                    │
    │       → ES: incidentlens-embeddings (kNN-indexed)   │
    │                                                     │
    ├─▶ wrappers.build_and_index_counterfactual()         │
    │       → ES: incidentlens-counterfactuals            │
    │                                                     │
    ├─▶ train.train_edge_gnn()                            │
    │       → EdgeGNN model                               │
    │                                                     │
    └─▶ temporal_gnn.prepare_temporal_dataset()            │
            → Temporal sequences → EvolveGCN-O             
```

---

## Graph Schema

Each PyG `Data` object produced by `graph_data_wrapper` contains:

| Attribute | Shape | Description |
|:----------|:------|:------------|
| `edge_index` | `(2, E)` long | Directed edges: src → dst |
| `x` | `(N, 6)` float | Node features: bytes_sent, bytes_recv, pkts_sent, pkts_recv, out_degree, in_degree |
| `edge_attr` | `(E, 5)` float | Edge features: packet_count, total_bytes, mean_payload, mean_iat, std_iat |
| `y` | `(E,)` long | Ground-truth edge labels: 0 = normal, 1 = malicious |
| `num_nodes` | int | Number of unique IPs in the window |
| `window_start` | float | Timestamp of window start |
| `window_id` | int | Sequential window identifier |
| `network` | object | `graph.network` with node/edge metadata |

---

## Elasticsearch Indices

### incidentlens-flows

Aggregated flow features per edge per time window.

| Field | Type | Description |
|:------|:-----|:------------|
| `flow_id` | keyword | Deterministic MD5 hash ID |
| `window_id` | integer | Time window index |
| `window_start` | float | Window start timestamp |
| `src_ip` / `dst_ip` | ip | Source and destination addresses |
| `protocol` | keyword | Network protocol |
| `packet_count` | float | Number of packets in the flow |
| `total_bytes` | float | Sum of packet sizes |
| `mean_payload` / `mean_iat` / `std_iat` | float | Payload and inter-arrival-time stats |
| `label` | integer | Ground truth (0/1) |
| `prediction` / `prediction_score` | integer/float | GNN model output |
| `timestamp` | date | Index timestamp (epoch_millis or ISO) |

### incidentlens-embeddings

Per-flow embedding vectors for kNN counterfactual retrieval.

| Field | Type | Description |
|:------|:-----|:------------|
| `flow_id` | keyword | Links to flows index |
| `label` | integer | 0 = normal, 1 = malicious |
| `prediction` | integer | Model prediction (0/1) |
| `window_id` | integer | Time window index |
| `src_ip` / `dst_ip` | ip | Source and destination addresses |
| `embedding` | dense_vector (cosine, kNN-indexed) | GNN or projection embedding (default dim = `EMBEDDINGS_MAPPING_DIM` = 16) |

### incidentlens-counterfactuals

Feature-level diffs explaining each anomaly.

| Field | Type | Description |
|:------|:-----|:------------|
| `cf_id` | keyword | Unique counterfactual document ID |
| `flow_id` | keyword | The anomalous flow |
| `nearest_normal_id` | keyword | Closest normal flow via kNN |
| `prediction` / `cf_prediction` | keyword | Original and counterfactual predictions |
| `similarity_score` | float | Cosine similarity score |
| `feature_diffs` | nested | Per-feature: original_value, cf_value, abs_diff, pct_change, direction |
| `edges_removed` | nested | Graph-level: which edges were removed in the perturbation |
| `explanation_text` | text | Human-readable explanation narrative |
| `timestamp` | date | Index timestamp (epoch_millis or ISO) |

### incidentlens-packets

Raw individual packet records from the dataset (`ingest_pipeline.py`).

| Field | Type | Description |
|:------|:-----|:------------|
| `packet_index` | integer | Sequential packet number |
| `timestamp` | double | Packet capture timestamp |
| `inter_arrival_time` | float | Time since previous packet |
| `src_ip` / `dst_ip` | ip | Source and destination addresses |
| `src_port` / `dst_port` | integer | Port numbers |
| `protocol` | integer | IP protocol number |
| `ttl` | integer | Time to live |
| `ip_header_len` | integer | IP header length |
| `tcp_flags` | float | TCP flag bitmask |
| `udp_length` | float | UDP datagram length |
| `payload_length` | integer | Payload size |
| `packet_length` | integer | Total packet size |
| `label` | integer | Ground truth (0/1) |

---

## Key Constants

| Constant | Module | Value | Description |
|:---------|:-------|:------|:------------|
| `FLOWS_INDEX` | wrappers.py | `"incidentlens-flows"` | Flow index name |
| `COUNTERFACTUALS_INDEX` | wrappers.py | `"incidentlens-counterfactuals"` | Counterfactual index name |
| `EMBEDDINGS_INDEX` | wrappers.py | `"incidentlens-embeddings"` | Embeddings index name |
| `EMBEDDINGS_MAPPING_DIM` | wrappers.py | `16` | Default embedding vector dimensionality |
| `FEATURE_FIELDS` | wrappers.py | `["packet_count", "total_bytes", "mean_payload", "mean_iat", "std_iat"]` | Metric fields used for stats and analysis |
| `RAW_PACKETS_INDEX` | ingest_pipeline.py | `"incidentlens-packets"` | Raw packets index name |
| `DATA_DIR` | ingest_pipeline.py | `"data/"` | Default NDJSON data directory |
| `ILM_POLICY_NAME` | wrappers.py | `"incidentlens-ilm-policy"` | Index lifecycle management policy |
| `INGEST_PIPELINE_NAME` | wrappers.py | `"incidentlens-nan-cleanup"` | ES ingest pipeline for NaN/Inf cleanup |
| `SEVERITY_RUNTIME_FIELDS` | wrappers.py | (Painless scripts) | Runtime fields: `severity_level`, `severity_score`, `traffic_volume_category` |

---

## Agent Tool Dispatch

All 19 tools follow the OpenAI function-calling schema format. The LLM agent calls them via `agent_tools.dispatch(tool_name, args)`, which routes to the appropriate `wrappers.*` function.

**Detection:** `es_health_check`, `detect_anomalies`, `search_flows`, `get_flow`, `search_raw_packets`

**Analysis:** `feature_stats`, `feature_percentiles`, `significant_terms`, `find_similar_incidents`

**Explainability:** `counterfactual_analysis`, `counterfactual_narrative`, `explain_flow`, `graph_edge_counterfactual`, `graph_window_comparison`, `search_counterfactuals`

**Assessment:** `assess_severity`, `severity_breakdown`

**ML:** `get_ml_anomaly_records`, `get_ml_influencers`

---

## Server Endpoints

Start with `python src/Backend/main.py serve --port 8000`.

### Pydantic Request Models

```python
class InvestigateRequest(BaseModel):
    query: str = ""               # empty = auto-detect anomalies

class DetectRequest(BaseModel):
    method: str = "label"          # "label" | "score" | "stats"
    threshold: float = 0.5
    size: int = 50

class PaginatedSearchRequest(BaseModel):
    query: str = ""                # text query or "" for match_all
    size: int = 20
    search_after: list | None = None
    pit_id: str | None = None
```

### REST

| Method | Path | Handler |
|:-------|:-----|:--------|
| `GET` | `/health` | Server + ES health |
| `POST` | `/api/investigate` | Full investigation (JSON) |
| `POST` | `/api/detect` | Quick anomaly detection |
| `GET` | `/api/flows` | List flows (optional label/IP filters) |
| `GET` | `/api/stats` | Feature stats by label |
| `POST` | `/api/counterfactual` | Counterfactual for a flow |
| `GET` | `/api/severity/{flow_id}` | Severity assessment |
| `GET` | `/api/similar/{flow_id}` | kNN similar incidents |
| `GET` | `/api/tools` | List agent tools |
| `GET` | `/api/incidents` | Anomalous flows as frontend `Incident` objects |
| `GET` | `/api/incidents/{id}` | Single incident detail by flow ID |
| `GET` | `/api/incidents/{id}/graph` | Network graph scoped to incident IPs |
| `GET` | `/api/incidents/{id}/logs` | ES-style log entries scoped to incident IPs |
| `GET` | `/api/severity-breakdown` | Runtime-field severity distribution |
| `GET` | `/api/flows/severity` | Query flows by runtime severity level |
| `POST` | `/api/flows/search` | Paginated search (search_after + PIT) |
| `GET` | `/api/counterfactuals/search` | Full-text search over CF narratives |
| `GET` | `/api/aggregate/{field}` | Composite aggregation (paginated) |
| `GET` | `/api/ml/anomalies` | ES ML anomaly records |
| `GET` | `/api/ml/influencers` | ES ML influencer results |

> **Error handling:** All REST endpoints check for `"error"` in the tool response and return HTTP 502 (backend error) or 404 (not found) instead of 200 with an error body.

#### Frontend Incident Endpoints

The `/api/incidents*` endpoints are convenience wrappers that reshape raw ES flow data into the `Incident` shape consumed by the React frontend. They use a shared `_flow_to_incident()` helper:

```python
# Maps: raw ES flow dict → {id, title, severity, status, timestamp,
#        affectedSystems, description, anomalyScore}
_flow_to_incident(flow: dict) -> dict
```

| Endpoint | What it does |
|:---------|:-------------|
| `GET /api/incidents` | Calls `detect_anomalies` → maps each flow to `Incident` |
| `GET /api/incidents/{id}` | Calls `get_flow` → maps to single `Incident` |
| `GET /api/incidents/{id}/graph` | Calls `get_flow` → scopes by incident IPs → calls `detect_anomalies` → builds `{nodes, edges}` for D3 force graph |
| `GET /api/incidents/{id}/logs` | Calls `get_flow` → scopes by incident `src_ip` → calls `search_flows` → formats as `{totalHits, logs[], query}` |

### WebSocket

**`WS /ws/investigate`** — Connect and send `{"query": "..."}`. Server streams JSON events:

```json
{"type": "status",      "content": "Starting investigation..."}
{"type": "thinking",    "content": "Analyzing flow patterns..."}
{"type": "tool_call",   "tool": "detect_anomalies", "arguments": {...}}
{"type": "tool_result", "tool": "detect_anomalies", "result": "..."}
{"type": "conclusion",  "content": "## Investigation Summary\n..."}
{"type": "done"}
```

All events also include a `timestamp` field (Unix epoch float).

---

## Running Tests

```bash
# Run all pytest tests
python -m pytest src/Backend/tests/ -v

# Run unified runner (unittest + pytest)
python src/Backend/tests/run_all.py
```

316 tests (95 unittest + 221 pytest) across 6 test suites:

| Test File | Focus |
|:----------|:------|
| `test_gnn_edge_cases.py` | Graph construction, GNN forward/backward, normalization, collation |
| `test_temporal_gnn_full.py` | EvolveGCN-O training, temporal sequences, snapshot preparation |
| `test_temporal_gnn_meticulous.py` | Edge-case coverage for LSTM weight evolution, empty graphs, single-node graphs |
| `test_csv_to_json.py` | CSV converter: `_safe_val` NaN/Inf/numpy, merge, NDJSON chunks, metadata, full pipeline |
| `test_agent_tools.py` | Agent tool registry, dispatch, sanitization, individual tools (5 of 19) |
| `test_e2e_pipeline.py` | Full pipeline integration: CSV → NDJSON → graphs → preprocess → normalize → GNN forward |
| `run_all.py` | Unified test runner — loads all suites, prints summary |

---

## Environment Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `OPENAI_API_KEY` | (none) | API key for the LLM agent |
| `OPENAI_MODEL` | `gpt-4o` | Model to use for the agent |
| `OPENAI_BASE_URL` | (none) | Custom endpoint (e.g., `http://localhost:11434/v1` for Ollama) |
| `PORT` | `8000` | Server port |
| `INCIDENTLENS_CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated list of allowed CORS origins |
| `INCIDENTLENS_DEV` | (none) | Set to `1` or `true` to enable hot-reload in `python server.py` |
| `INCIDENTLENS_DATA_ROOT` | `data/` | Root directory for NDJSON data files |
| `ELASTICSEARCH_URL` | `http://localhost:9200` | Elasticsearch endpoint (used in Docker compose) |
| `INCIDENTLENS_PACKETS_CSV` | `data/ssdp_packets_rich.csv` | Default raw packets CSV path |
| `INCIDENTLENS_LABELS_CSV` | `data/SSDP_Flood_labels.csv` | Default ground-truth labels CSV path |

> **Note:** The Elasticsearch URL (`http://localhost:9200`) is hardcoded in `wrappers.get_client()`. There is no env var to override it — modify the source if you need a different host.

---

## AgentConfig

The `IncidentAgent` uses a dataclass for configuration. These are **compile-time defaults**, not env vars:

| Field | Default | Description |
|:------|:--------|:------------|
| `max_steps` | `15` | Maximum reasoning steps per investigation |
| `temperature` | `0.1` | LLM sampling temperature (low = deterministic) |
| `max_tokens` | `4096` | Max tokens per LLM response |

---

## CORS & Middleware

The FastAPI server reads allowed CORS origins from the `INCIDENTLENS_CORS_ORIGINS` environment variable (comma-separated, default: `http://localhost:5173,http://localhost:3000`):

```python
_ALLOWED_ORIGINS = os.getenv(
    "INCIDENTLENS_CORS_ORIGINS", "http://localhost:5173,http://localhost:3000"
).split(",")
```

> **Production note:** Set `INCIDENTLENS_CORS_ORIGINS` to your frontend domain(s) before deploying.

The agent is instantiated as a **thread-safe singleton** (`IncidentAgent`) and reused across all requests.

---

## Elasticsearch ML Functions

`wrappers.py` also exposes 4 Elasticsearch ML anomaly-detection functions (optional — require an ML-licensed cluster):

| Function | Purpose |
|:---------|:--------|
| `create_anomaly_detection_job()` | Create an ES ML job for flow anomaly detection |
| `create_anomaly_datafeed()` | Attach a datafeed to an ML job |
| `get_anomaly_records()` | Fetch anomaly records from an ML job |
| `get_influencers()` | Get top influencers from an ML anomaly job |

---

## Key Design Decisions

1. **Edge-level embeddings** — GNNs produce node embeddings; we concatenate `[node_emb[src], node_emb[dst]]` to get per-flow embeddings for kNN counterfactual retrieval.
2. **Singleton ES client** — Reused across all requests; matches Elasticsearch SDK best practices.
3. **Pre-processed GNN inputs** — Self-loops, degree normalization, and NaN sanitization happen *once* at data-prep time, not in every forward pass.
4. **Numpy-first graph building** — `graph_data_wrapper` uses composite key packing, `np.add.at`, and `searchsorted` for window assignment — no Python loops over packets. `graph.py` uses pre-allocated numpy arrays for edge indices and `argsort`+`searchsorted` for window splitting.
5. **Vectorized analysis paths** — Edge perturbation counterfactuals use batch `(T,F)` matrix ops with sum-of-squares update formulas. Window comparison, anomalous/normal finding, and severity z-scores all use vectorized numpy.
6. **Graceful mock fallback** — Frontend hooks silently fall back to mock data when the backend is unreachable, enabling offline UI development.
7. **ES-native analytics** — Runtime severity fields (Painless), ILM policies, ingest pipelines, index templates, composite aggregations, PIT-based pagination, and full-text CF search — leveraging ES capabilities directly rather than application-level reimplementations.
8. **Async server** — All ES calls in `server.py` are wrapped in `asyncio.to_thread()` for non-blocking request handling.

---

## Module Deep Dives

### wrappers.py (1 933 lines, 53 functions)

Key exported functions beyond the module-map summary:

| Function | Purpose |
|:---------|:--------|
| `get_client(url)` | Return (or create) a singleton ES client |
| `set_gnn_encoder()` / `get_gnn_encoder()` | Register / retrieve the active GNN model in a global registry |
| `ping()` | Simple ES connectivity check |
| `setup_ilm_policy()` | Create ILM lifecycle policy for index rollover |
| `setup_ingest_pipeline()` | Create ES ingest pipeline for NaN/Inf cleanup |
| `setup_index_templates()` | Create index templates for consistent mappings |
| `create_index()` / `setup_all_indices()` / `delete_all_indices()` | Index lifecycle management |
| `_flow_id()` / `_flow_ids_batch()` | Deterministic MD5 flow-ID generation |
| `index_pyg_graph()` | Index a single PyG `Data` object into ES |
| `index_graphs_bulk()` | Bulk-index a list of PyG graphs |
| `generate_embeddings()` | Run GNN forward pass → index embedding vectors |
| `build_and_index_graphs()` | Convenience: build graphs from CSV + index into ES |
| `build_index_and_embed()` | Convenience: build + index + embed in one call |
| `index_embeddings()` | Bulk-index precomputed embedding vectors |
| `knn_search_nearest_normal()` | kNN search restricted to `label=0` |
| `knn_search()` | Generic kNN embedding search |
| `compute_counterfactual_diff()` | Feature-level diff between anomalous and nearest-normal flow |
| `build_and_index_counterfactual()` | Compute and index a counterfactual into ES |
| `feature_stats_by_label()` | Extended stats per feature grouped by label |
| `feature_percentiles_by_label()` | Percentile distributions per feature |
| `search_anomalous_flows()` | Search flows with `label=1` |
| `get_counterfactuals_for_flow()` | Retrieve all counterfactuals for a flow |
| `format_counterfactual_narrative()` | Generate human-readable CF explanation |
| `significant_terms_by_label()` | IPs/protocols overrepresented in attack traffic |
| `explain_flow_match()` | ES `_explain` API for query matching |
| `search_flows_with_severity()` | Runtime-field severity search (Painless scripts) |
| `aggregate_severity_breakdown()` | Runtime-field severity distribution |
| `search_with_pagination()` | search_after + PIT cursor-based pagination |
| `close_pit()` | Close a point-in-time |
| `composite_aggregation()` | Paginated composite aggregation |
| `full_text_search_counterfactuals()` | Full-text search over CF narratives with highlighting |
| `_self_test()` | Self-test / smoke-test function |

### graph.py

| Item | Type | Key Methods / Purpose |
|:-----|:-----|:----------------------|
| `node` | class | Represents a single IP node |
| `network` | class | Full graph: `add_node`, `add_edge`, `set_node_features`, `build_edge_index`, `build_sparse_adjacency`, `from_edge_list`, `to_pyg_data` |
| `add_window_id()` | function | Assign window IDs to a DataFrame |
| `build_node_map()` | function | Map IP strings to integer IDs |
| `build_flow_table()` | function | Aggregate packets into flows |
| `build_window_data()` | function | Build a single-window PyG Data |
| `build_snapshot_dataset()` | function | Build a list of PyG Data per time window |
| `build_sample_graph()` | function | Generate a small test graph |

### graph_data_wrapper.py

Internal vectorized helpers (all numpy-based):

| Function | Purpose |
|:---------|:--------|
| `_assign_window_ids()` | Broadcast timestamp → window assignment |
| `_aggregate_flows_numpy()` | Per-flow stats via `np.add.at` |
| `_compute_node_features_arrays()` | Degree / traffic volume features |
| `_build_network_fast()` | Build `graph.network` without Python loops |
| `build_sliding_window_graphs()` | **Public** — end-to-end graph construction |

### train.py

| Function | Purpose |
|:---------|:--------|
| `tensor_make_finite_()` | In-place NaN/Inf replacement |
| `sanitize_graphs_inplace()` | Clean all graphs before training |
| `time_split()` | Train/val split by time |
| `normalize_edge_features()` | Z-score normalization |
| `label_stats()` | Print class distribution |
| `compute_pos_weight()` | Class imbalance weight |
| `safe_roc_auc()` | ROC-AUC with error handling |
| `find_best_threshold()` | Threshold optimization on val set |
| `EdgeGNN` | class — GraphSAGE + Edge MLP model |
| `train_edge_gnn()` | End-to-end training loop |

### temporal_gnn.py

| Function | Purpose |
|:---------|:--------|
| `preprocess_graph()` / `preprocess_graphs()` | Add self-loops, degree-normalize |
| `sanitize_graph()` | NaN/Inf cleanup |
| `normalize_features_global()` / `apply_normalization()` | Global feature normalization |
| `recompute_node_features()` | Recalculate node features from edge data |
| `build_temporal_sequences()` | Create sliding windows of graph snapshots |
| `_postprocess_graphs()` | Post-processing after dataset load |
| `prepare_temporal_dataset()` / `prepare_temporal_dataset_from_csv()` | End-to-end dataset prep |
| `collate_temporal_batch()` | Custom PyG batch collation |
| `EvolvingGNN` | class — LSTM-evolved GCN weights |

### gnn_interface.py

| Item | Purpose |
|:-----|:--------|
| `BaseGNNEncoder` | Abstract base class for GNN encoders |
| `.forward()` | Abstract forward pass |
| `.predict()` | Predict anomaly scores |
| `.encode()` | Generate per-flow embedding vectors |
| `.predict_labels()` | Predict binary labels with threshold |
| `.save()` / `.load()` | Model serialization |
| `create_dataloaders()` | Train/val/test DataLoader creation |
| `compute_class_weights()` | Class weight computation for imbalance |

### ingest_pipeline.py

The 8-step pipeline orchestrated by `run_pipeline()`:

| Step | Function | Description |
|:-----|:---------|:------------|
| 1 | `load_ndjson_files()` | Load chunked NDJSON into a DataFrame |
| 2 | `setup_raw_packet_index()` | Create the `incidentlens-packets` index |
| 3 | `index_raw_packets()` | Bulk-index raw packet records |
| 4 | `build_graphs_from_df()` | Build PyG graphs from the DataFrame |
| 5 | `index_all_graphs()` | Index flow data into `incidentlens-flows` |
| 6 | `generate_feature_embeddings()` | Generate and index GNN embeddings |
| 7 | `run_counterfactual_analysis()` | Compute and index counterfactuals |
| 8 | `run_feature_analysis()` | Feature stats + `print_counterfactual_report()` |

