# Glia 🧠
**The self-pruning semantic cache for living RAG architectures.**

Glia is a Python library that wraps around an existing semantic caching workflow to add three critical capabilities: dependency-tagged cache storage, targeted cache invalidation via a background watcher, and developer-controlled embedding injection[cite: 5].

It is an additive wrapper designed to sit alongside your existing notebooks or applications without requiring changes to how you parse documents, generate FAQs, or interact with your LLMs[cite: 6, 58]. 

## 🌟 Core Philosophy: Zero Workflow Modification
Adding Glia requires almost zero refactoring.The document loading pipeline, the LLM setup, the Redis connection, and your overall execution flow remain completely unchanged[cite: 63, 64, 65, 67].You simply add a small initialization block and a single metadata argument to your existing storage call[cite: 7].Everything else continues unchanged[cite: 8].

## ✨ Key Features
***Developer-Controlled Embeddings:** Pass in your existing embedding provider (`embed()` and `embed_many()`); the library does not force or instantiate its own LLMs[cite: 17, 19, 111].
* **Dependency Tagging:** Every cached response is tied to a `source_id` tag (like a file name or node ID) saved directly in the custom schema[cite: 20, 21].
* **Targeted Invalidation:** A dedicated `CacheInvalidator` allows you to delete all entries tied to a specific `source_id` in a single batched, atomic operation without leaving orphaned index entries[cite: 36, 39, 42].
* **Background Watcher:** A background service monitors local file systems for modification or deletion events, automatically triggering invalidation without blocking your main thread[cite: 46, 48, 51].

## 🛠 Prerequisites
* **Python:** 3.9+ [cite: 9]
* **Database:** Redis Stack (the RediSearch module is strictly required) [cite: 9, 114]
* **Embeddings:** Any provider exposing an `embed` and `embed_many` interface (e.g., VertexAITextVectorizer)[cite: 19, 66].

## 🚀 Quick Start & Integration

### 1. Initialization
After your existing vectorizer is initialized, add a single cell to construct the `GliaManager`.Pass your existing vectorizer instance directly in[cite: 71, 72].

```python
from glia import GliaManager, CacheInvalidator, CacheWatcher

# Initialize the manager with your existing vectorizer and Redis URL
cache_manager = GliaManager(
    vectorizer=my_existing_vectorizer, 
    redis_url="redis://localhost:6379", 
    distance_threshold=0.2          
)