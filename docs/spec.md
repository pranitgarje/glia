# Glia: Technical Specification

**Tech Stack:** Python 3.9+, Redis Stack (RediSearch module required).
**Integration Pattern:** Additive wrapper.The library MUST NOT modify the developer's existing document parsing, LLM interaction, or vectorizer initialization . 

## Core Modules
1. **GliaManager (`manager.py`):** Handles `store()` and `check()` methods. Accepts an injected embedding provider (`vectorizer`) at initialization[cite: 126, 218]. Uses RediSearch JSON indexing with a default schema including a `source_id` TAG.
2.**CacheInvalidator (`invalidator.py`):** Executes `delete_by_tag(source_id)` to atomically remove JSON data and index references in a single batched operation.
3. **CacheWatcher (`watcher.py`):** Background service monitoring data sources. Operates via background threads (nest_asyncio compatible) and triggers invalidation via `delete_by_tag()`.

## Implementation Layers (Strict Order)
* **Layer 1 (Foundation):** `exceptions.py`, `events.py` (EventEmitter, WatcherEvent), `schema.py` (SchemaBuilder) .
* **Layer 2 (Adapter Contracts):** `adapters/base.py`, `adapters/polling.py`, `adapters/cdc.py`.
* **Layer 3 (Cache Core):** `manager.py`, `invalidator.py` .
* **Layer 4 (Concrete Adapters):** `vector.py`, `graph.py`, `relational.py` (Polling first, then CDC) .
* **Layer 5 (Watcher Engine):** `runners.py`, `watcher.py`.
* **Layer 6 (Public Surface):** `__init__.py` re-exports .