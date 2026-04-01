"""
glia/__init__.py
────────────────
Layer 6 — Public Surface

Main entry point for the Glia library.

Glia is an additive wrapper around an existing semantic caching workflow
that adds three capabilities: dependency-tagged cache storage, targeted
cache invalidation via a background watcher, and developer-controlled
embedding injection.  It is designed to slot into an existing RAG notebook
with a single initialisation block and one extra argument on the cache
store call — nothing else in the surrounding workflow needs to change.

Three symbols form the core public API:

- GliaManager        — owns the Redis connection, the RediSearch index
                       lifecycle, and the store() / check() methods.
- CacheInvalidator   — executes targeted delete_by_tag(source_id)
                       operations against the managed index.
- CacheWatcher       — background service that monitors data sources and
                       triggers CacheInvalidator automatically when changes
                       are detected.

Concrete database adapters (VectorDBAdapter, GraphDBAdapter,
RelationalDBAdapter) are available from the glia.adapters sub-package and
are not re-exported here to keep the top-level namespace minimal.

Everything else in the package — runners.py, events.py, schema.py, the
abstract adapter base classes — is internal.  Developers are not expected
to import from those modules directly.

Typical usage::

    from glia import GliaManager, CacheInvalidator, CacheWatcher
    from glia.adapters import VectorDBAdapter, GraphDBAdapter, RelationalDBAdapter
"""

# ---------------------------------------------------------------------------
# Core public imports
# ---------------------------------------------------------------------------
# Each import targets the module that owns the class — never an intermediate
# file — so the import chain is explicit and traceable.
#
# Internal modules (runners.py, events.py, schema.py) are deliberately
# absent: they are implementation details that no developer-facing code
# should depend on directly.

from glia.manager import GliaManager
from glia.invalidator import CacheInvalidator
from glia.watcher import CacheWatcher

# ---------------------------------------------------------------------------
# Adapter re-exports (convenience)
# ---------------------------------------------------------------------------
# Pulled from glia.adapters so developers can use a single import line for
# the three concrete adapter families when they need them.  The adapters
# sub-package owns the authoritative __all__ for these names; we re-export
# them here purely as a convenience shortcut.

from glia.adapters import VectorDBAdapter, GraphDBAdapter, RelationalDBAdapter

# ---------------------------------------------------------------------------
# Explicit public surface
# ---------------------------------------------------------------------------
# __all__ defines the complete public API of the glia package.
# - Anything not listed here is considered private/internal.
# - `from glia import *` will only bind these six names.
# - Static analysers and documentation generators treat this as the
#   authoritative list of symbols Glia exposes.

__all__ = [
    # Cache core
    "GliaManager",
    "CacheInvalidator",
    "CacheWatcher",
    # Concrete adapters (re-exported from glia.adapters)
    "VectorDBAdapter",
    "GraphDBAdapter",
    "RelationalDBAdapter",
]