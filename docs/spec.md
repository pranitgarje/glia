# RagSync: Technical Specification

## System Overview
RagSync is a high-performance semantic caching and real-time synchronization library for Retrieval-Augmented Generation (RAG) pipelines.

**Tech Stack:** Python 3.11+, Pydantic, Asyncio, PyTorch (for Cross-Encoders).

## Component I: Two-Stage Retrieval Pipeline
* **Stage 1 (Filter):** Converts incoming user text into a vector embedding using a lightweight model. Queries the cache database using a Hierarchical Navigable Small World (HNSW) algorithm. Must operate in $O(\log N)$ time, avoiding linear $O(N)$ scans. Returns a strict subset of Top-K nearest neighbors. If the cache is empty or lacks vectors within a broad radius, it triggers a Cache Miss.

* **Stage 2 (Judge):** Utilizes a local Cross-Encoder (e.g., ms-marco-MiniLM). Formats and processes the new query alongside Top-K cached queries simultaneously. Outputs an entailment score and compares it against a strict user-configurable threshold.

* **Fallback & Routing:** Upon a Cache Miss, seamlessly forwards the original query to the Main Vector Database and primary LLM. Asynchronously embeds the new query and stores both the vector and LLM response back into the Cache Database.

## Component II: File System Synchronization
* **Core Engine:** Normalizes incoming mutation events into a standardized structure containing `Database_Type`, `Operation_Type`, `Entity_ID`, and a `Timestamp`. Implements a time-windowed debouncing mechanism for bulk updates. Operates asynchronously to avoid blocking the main RAG thread.

* **Trackers:**
  * **Relational DBs (e.g., PostgreSQL):** Hooks into logical replication to read the Write-Ahead Log (WAL).
  * **Vector DBs:** Captures the specific string or integer Vector ID that was modified.
  * **Graph DBs (e.g., Neo4j):** Differentiates between Node property updates and Edge mutations.

* **Integration:** Immediately hands off the normalized `Entity_ID` to the Dependency Graph.

## Component III: Smart Diffing & Invalidation Engine
* **Diffing:** Computes a deterministic hash (e.g., SHA-256) of the entity's core semantic data. Compares the "New State Hash" against the "Previous State Hash" to determine if invalidation is necessary.

* **Dependency Graph:** Maintains a bi-directional mapping of `Source_Entity_ID` to `List[Cache_Keys]`. Lookups must execute in $O(1)$ or $O(\log N)$ time. Prunes corresponding edges when a cache key naturally expires or is deleted to prevent memory leaks.

* **Targeted Pruning:** Executes atomic DELETE commands directly to the Cache Store. Removes the corresponding embedded query vector from the Stage 1 HNSW index. Strictly guarantees that unrelated cache entries remain untouched.

## Component IV: Database Abstraction Layer
* **Contracts:** Defines abstract base classes (e.g., `BaseVectorStore`, `BaseCacheStore`) with standardized method signatures. Mandates connection lifecycle methods including `initialize()`, `health_check()`, and `close()`.

* **Outputs & Features:** Enforces standardized payloads, such as a `SearchResult` object containing `id`, `score`, `text_chunk`, and `metadata`. Requires drivers to declare supported features via boolean flags.

* **Driver Implementations:** Vector drivers must handle metric normalization and validate vector dimensionality. Cache drivers must serialize/deserialize complex objects and translate optional TTL parameters.

* **Error Handling:** Drivers must catch native exceptions (e.g., `redis.exceptions.ConnectionError`) and translate them into standardized library exceptions (e.g., `RagSyncConnectionError`).