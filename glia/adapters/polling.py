"""
glia/adapters/polling.py
────────────────────────
Layer 2 — Adapter Contracts

Defines PollingAdapter, the abstract contract that any polling-mode
adapter must satisfy.

Architectural constraints (BDUF §2.3):
- PollingAdapter extends DatabaseAdapter.
- runners.PollingRunner types ONLY against this interface — it never
  imports a concrete adapter class.
- This file imports ONLY from the Python standard library and
  glia.adapters.base — never from higher-layer glia modules.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Iterator, Optional


from glia.adapters.base import DatabaseAdapter


class PollingAdapter(DatabaseAdapter):
    """
    Abstract contract for polling-mode data-source adapters.

    A polling adapter works by periodically calling ``poll()`` to
    retrieve newly changed records since the last known cursor
    position.  ``PollingRunner`` owns the timed loop and calls
    these methods; the adapter owns the query and cursor logic.

    Subclasses MUST implement all abstract methods inherited from
    ``DatabaseAdapter`` (``connect``, ``disconnect``,
    ``map_to_source_id``) in addition to the three methods declared
    here.

    Attributes
    ----------
    poll_interval : float
        Seconds between consecutive ``poll()`` calls.
        Set by the concrete subclass or passed via constructor.
    last_cursor : Any
        The most-recently-committed cursor value.  Persisted between
        poll cycles so that only new changes are returned.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        mode: str,
        source_id_field: str,
        poll_interval: float = 30.0,
        last_cursor: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialise the polling adapter and forward shared parameters
        to ``DatabaseAdapter``.

        Parameters
        ----------
        mode:
            Execution strategy — must be ``"polling"`` for this adapter.
            Validated and stored by ``DatabaseAdapter.__init__``.
        source_id_field:
            Name of the source field that becomes Glia's ``source_id``.
            Stored by ``DatabaseAdapter.__init__``.
        poll_interval:
            Seconds between consecutive ``poll()`` calls.  Defaults to
            ``30.0``.  ``PollingRunner`` reads this attribute at start-up
            to configure its sleep timer — it MUST be set before
            ``start()`` is called.
        last_cursor:
            The initial cursor value.  Defaults to ``None``, which signals
            ``PollingRunner`` that no records have been processed yet and
            the adapter should decide its own bootstrap strategy (e.g.
            query all records, or start from "now").  After the first
            successful batch, ``PollingRunner`` calls ``advance_cursor()``
            and this attribute is superseded by the committed value.
        **kwargs:
            Forwarded up the MRO chain via ``super().__init__(**kwargs)``.
            Required for cooperative multiple-inheritance in Layer 4
            concrete adapters (e.g. ``VectorDBAdapter(PollingAdapter,
            CDCAdapter)``).

        Notes
        -----
        Concrete subclasses that override ``__init__`` MUST call
        ``super().__init__()`` and pass at minimum ``mode`` and
        ``source_id_field`` so that ``DatabaseAdapter`` can validate
        ``mode`` and raise ``MissingModeError`` if it is absent or
        invalid.
        """
        # Delegate mode validation and source_id_field storage to
        # DatabaseAdapter.  kwargs are forwarded for MRO co-operation.
        super().__init__(mode=mode, source_id_field=source_id_field, **kwargs)

        # Instance attributes — typed explicitly so static analysers
        # (mypy, pyright) can resolve them on PollingAdapter instances
        # without having to inspect the concrete subclass.
        self.poll_interval: float = poll_interval
        self.last_cursor: Optional[Any] = last_cursor

    # ------------------------------------------------------------------
    # Polling contract
    # ------------------------------------------------------------------

    @abstractmethod
    def poll(self) -> Iterator[Any]:
        """
        Query the data source for records that changed since
        ``last_cursor`` and yield them one at a time.

        ``PollingRunner`` iterates the generator and calls
        ``map_to_source_id()`` on each yielded record, then forwards
        the result to ``CacheWatcher._dispatch()``.

        Yields
        ------
        Any
            Raw records whose concrete type is defined by the
            implementing adapter (e.g. a dict row, a graph node, a
            change-log entry).

        Notes
        -----
        The method MUST NOT advance ``last_cursor`` internally.
        Cursor advancement is the responsibility of
        ``advance_cursor()``, called by ``PollingRunner`` after the
        full batch has been dispatched.
        """
        raise NotImplementedError

    @abstractmethod
    def get_cursor(self) -> Any:
        """
        Return the current cursor value that marks the boundary
        between already-processed and not-yet-processed records.

        The cursor type is adapter-defined (e.g. a timestamp, an
        auto-increment integer, a log sequence number).

        Returns
        -------
        Any
            Current cursor value.  May be ``None`` on first run
            before any records have been processed.
        """
        raise NotImplementedError

    @abstractmethod
    def advance_cursor(self, new_cursor: Any) -> None:
        """
        Persist ``new_cursor`` as the new lower bound for the
        next ``poll()`` call.

        Called by ``PollingRunner`` after it has successfully
        dispatched all records from the current batch.

        Parameters
        ----------
        new_cursor:
            The cursor value to save.  Must be of the same type
            returned by ``get_cursor()``.
        """
        raise NotImplementedError