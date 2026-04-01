"""
glia/adapters/__init__.py
─────────────────────────
Layer 6 — Public Surface

Exposes the concrete database adapters that developers use to connect
Glia's cache-invalidation machinery to their data sources.

Three adapter families are provided, each supporting both polling and
CDC execution modes:

- VectorDBAdapter   — for vector store backends (e.g. Pinecone, Weaviate,
                      Redis as a vector DB).
- GraphDBAdapter    — for graph database backends (e.g. Neo4j).
- RelationalDBAdapter — for relational database backends (e.g. PostgreSQL,
                        MySQL) via updated_at polling or logical replication.

All three classes inherit from the abstract contracts defined in Layer 2
(adapters/base.py, adapters/polling.py, adapters/cdc.py) and are the only
adapter symbols that developers are expected to import directly.

Internal modules — runners.py, events.py, schema.py, and the abstract
adapter base classes — are not re-exported here and are not part of the
public API.

Typical usage::

    from glia.adapters import VectorDBAdapter, GraphDBAdapter, RelationalDBAdapter
"""

# ---------------------------------------------------------------------------
# Public imports — concrete adapters only
# ---------------------------------------------------------------------------
# Abstract base classes (DatabaseAdapter, PollingAdapter, CDCAdapter) are
# intentionally excluded: they are internal contracts for runners.py and
# the concrete adapter implementations, not symbols developers should ever
# instantiate or type-check against directly.

from glia.adapters.vector import VectorDBAdapter
from glia.adapters.graph import GraphDBAdapter
from glia.adapters.relational import RelationalDBAdapter

# ---------------------------------------------------------------------------
# Explicit public surface
# ---------------------------------------------------------------------------
# __all__ restricts `from glia.adapters import *` to exactly these three
# names and signals to static analysers (mypy, pyright, IDEs) that nothing
# else in this package is considered public.

__all__ = [
    "VectorDBAdapter",
    "GraphDBAdapter",
    "RelationalDBAdapter",
]