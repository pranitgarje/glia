"""
glia/adapters/base.py
─────────────────────
Layer 2 — Adapter Contracts

Defines DatabaseAdapter, the root abstract base class for every
concrete data-source adapter in the Glia library.

Architectural constraints (BDUF §2.3 / §5):
- Every concrete adapter MUST inherit from this class.
- mode= MUST be supplied at construction; MissingModeError is raised
  if it is absent.
- This file imports ONLY from the Python standard library and
  glia.exceptions — never from higher-layer glia modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


# Layer 1 import — exceptions.py must already exist.
from glia.exceptions import MissingModeError


# ---------------------------------------------------------------------------
# Valid adapter modes
# ---------------------------------------------------------------------------

VALID_MODES: frozenset[str] = frozenset({"polling", "cdc"})


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class DatabaseAdapter(ABC):
    """
    Root abstract class for all Glia data-source adapters.

    Subclasses implement either a polling contract (PollingAdapter),
    a CDC contract (CDCAdapter), or both.  The `mode` attribute tells
    CacheWatcher / runners.py which execution engine to spawn.

    Attributes
    ----------
    mode : str
        Execution mode for this adapter.  Must be one of
        ``"polling"`` or ``"cdc"``.  Validated in ``__init__``.
    source_id_field : str
        The field name in the data source that maps to Glia's
        ``source_id`` TAG.  Used by ``map_to_source_id()``.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str,
        source_id_field: str,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the adapter and validate ``mode``.

        Parameters
        ----------
        mode:
            Execution strategy — ``"polling"`` or ``"cdc"``.
            Raises ``MissingModeError`` when absent or invalid.
        source_id_field:
            Name of the source field that becomes Glia's ``source_id``.
        **kwargs:
            Forwarded to concrete subclass initialisers.

        Raises
        ------
        MissingModeError
            If ``mode`` is ``None``, empty, or not in
            ``{"polling", "cdc"}``.
        """
        # Explicit None check first — guards against callers who ignore the
        # type hint and pass mode=None.  The `not mode` branch then catches
        # the empty-string case.  Both produce a MissingModeError with a
        # message that names the two valid values so the developer knows
        # exactly what to pass without consulting the docs.
        if mode is None:
            raise MissingModeError(
                "Adapter 'mode' is required but was not supplied. "
                f"Pass mode='polling' or mode='cdc'."
            )
        if not mode or mode not in VALID_MODES:
            raise MissingModeError(
                f"Adapter 'mode' must be one of {sorted(VALID_MODES)}; "
                f"got {mode!r}."
            )

        self.mode: str = mode
        self.source_id_field: str = source_id_field

        # Forward any remaining kwargs up the MRO chain.  This is essential
        # for Layer 4 concrete adapters that use multiple inheritance
        # (e.g. VectorDBAdapter(PollingAdapter, CDCAdapter)) — without this
        # call, cooperative super() breaks and sibling __init__ methods are
        # silently skipped.
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """
        Open a connection to the underlying data source.

        Must be idempotent — calling connect() on an already-connected
        adapter must not raise an error or open a second connection.

        Raises
        ------
        AdapterConnectionError
            If the connection attempt fails.
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close the connection to the underlying data source cleanly.

        Must be idempotent — calling disconnect() on an already-
        disconnected adapter must not raise an error.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Source-ID mapping
    # ------------------------------------------------------------------

    @abstractmethod
    def map_to_source_id(self, record: Any) -> Optional[str]:
        """
        Translate a native data-source identifier into a unified Glia
        ``source_id`` string.

        This method is the single translation layer between the adapter's
        native record format and the ``source_id`` TAG that Glia stores in
        Redis.  It is called by both ``PollingRunner`` (once per record
        yielded from ``poll()``) and ``CDCRunner`` (once per event emitted
        from ``listen()``).  The returned string is passed directly to
        ``CacheInvalidator.delete_by_tag(source_id)``, so its value MUST
        be stable and deterministic: the same logical document must always
        produce the same ``source_id`` string regardless of when or how it
        is observed.

        Translation examples by adapter paradigm
        -----------------------------------------
        **Relational (RelationalDBAdapter)**
            ``record`` is typically a dict row from a change-log table or
            a logical-replication message.  The ``source_id`` is usually a
            composite key, e.g. ``"table:documents|pk:42"`` derived from
            ``record["table_name"]`` and ``record[self.source_id_field]``.

        **Graph (GraphDBAdapter)**
            ``record`` is a Neo4j CDC event dict or a Cypher query result
            node.  The ``source_id`` maps to the node's unique property,
            e.g. ``"doc:node_uuid_here"`` derived from
            ``record["properties"][self.source_id_field]``.

        **Vector (VectorDBAdapter)**
            ``record`` is a metadata dict from the vector store's change
            stream.  The ``source_id`` is typically the document's primary
            key or file path stored in that metadata field, e.g.
            ``"contracts/q4_2024.pdf"`` if ``self.source_id_field``
            is ``"file_path"``.

        Returning ``None``
        ------------------
        Return ``None`` when the record represents an event that is not
        relevant to cache invalidation (e.g. a schema-change event, a row
        in an unmonitored table, or a CDC heartbeat).  Returning ``None``
        tells the runner to skip this record silently — no error is raised
        and no invalidation is triggered.

        Parameters
        ----------
        record:
            A raw record as returned by ``poll()`` or emitted by
            ``listen()``.  The concrete type is defined by the implementing
            adapter and may be a ``dict``, a dataclass, or a vendor-specific
            event object.

        Returns
        -------
        str or None
            A non-empty ``source_id`` string to pass to
            ``CacheInvalidator.delete_by_tag()``, or ``None`` if this
            record should be skipped without triggering invalidation.

        Notes
        -----
        - The returned string MUST NOT contain Redis key-separator
          characters that would break RediSearch TAG queries.  Use ``":"``
          as a namespace separator and avoid ``" "``, ``","`` or ``"|"``
          unless your index schema is configured to handle them.
        - Do not perform I/O inside this method.  All information needed
          to derive the ``source_id`` must be present in ``record`` itself
          or in adapter attributes set at construction time.
        """
        raise NotImplementedError