# RagSync

**High-Performance Semantic Caching & Real-Time Synchronization for RAG Pipelines**

RagSync is a robust, developer-first library designed to solve the two biggest bottlenecks in Retrieval-Augmented Generation (RAG) applications: LLM latency and stale context. 

By implementing a lightning-fast two-stage semantic cache and a database-agnostic real-time monitoring engine, RagSync ensures your RAG pipeline serves instant, contextually accurate answers while autonomously invalidating outdated data.

---

## 🚀 Key Features

* **Two-Stage Retrieval Pipeline:** Achieves $O(\log N)$ retrieval times using an HNSW index for fast candidate generation, followed by precise verification using a local Cross-Encoder (e.g., `ms-marco-MiniLM`) to eliminate cache hallucinations.
* **Real-Time File System Sync:** A core monitoring engine that tracks source database mutations (PostgreSQL WAL tailing, Vector DB polling, Graph DB edge monitoring) without blocking the main application thread.
* **Smart Diffing & Invalidation:** Utilizes deterministic hashing (SHA-256) to differentiate between harmless metadata updates and true semantic changes, surgically pruning only the affected cache entries via an $O(1)$ dependency graph ledger.
* **Database Agnostic:** Built on a strict Database Abstraction Layer. Seamlessly swap between Redis, ChromaDB, FAISS, Pinecone, or PostgreSQL without rewriting your application logic.

## 🏗️ Architecture Overview

RagSync is composed of four decoupled subsystems working in tandem:

1.  **The Cache Search (Component I):** Intercepts user queries, checks the high-speed semantic cache, and routes to the primary LLM only on a verified Cache Miss.
2.  **The Monitor (Component II):** Listens to your primary databases (SQL, Vector, Graph) for `INSERT`, `UPDATE`, or `DELETE` operations.
3.  **The Updater (Component III):** Hashes incoming changes, checks the dependency graph, and issues targeted `DELETE` commands to the cache for stale data.
4.  **The Integrator (Component IV):** Normalizes inputs, outputs, and exceptions across all third-party database drivers.

## 📦 Installation

*(Assuming deployment via pip once published)*

```bash
pip install ragsync