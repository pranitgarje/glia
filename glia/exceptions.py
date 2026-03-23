"""
glia/exceptions.py
------------------
Centralised exception hierarchy for the Glia library.

All custom exceptions are defined here so every module can do:
    from glia.exceptions import <ExceptionName>
without creating circular imports.  No other glia module is imported in this file.
"""


class GliaBaseError(Exception):
    """
    Root base class for all Glia-specific exceptions.

    Catch this to handle any library error with a single except clause.
    All other Glia exceptions inherit from this class.
    """


# ---------------------------------------------------------------------------
# Adapter / watcher layer
# ---------------------------------------------------------------------------

class MissingModeError(GliaBaseError):
    """
    Raised by ``DatabaseAdapter.__init__()`` when the required ``mode=``
    constructor parameter is not supplied.

    The ``mode`` parameter must be one of ``"polling"`` or ``"cdc"``.
    Because the choice of execution engine (PollingRunner vs CDCRunner)
    depends entirely on this value, the adapter cannot be initialised
    without it — hence a dedicated exception rather than a plain ValueError.

    A default human-readable message is provided so the exception is
    informative even when raised without an explicit argument::

        raise MissingModeError()
        # MissingModeError: DatabaseAdapter requires a 'mode' argument.
        #   Pass mode='polling' for timed-interval checks or
        #   mode='cdc' for real-time change-data-capture streaming.

    The default message can be overridden for adapter-specific context::

        raise MissingModeError(
            f"{type(self).__name__} requires mode='polling' or mode='cdc'."
        )
    """

    _DEFAULT_MESSAGE: str = (
        "DatabaseAdapter requires a 'mode' argument. "
        "Pass mode='polling' for timed-interval checks or "
        "mode='cdc' for real-time change-data-capture streaming."
    )

    def __init__(self, message: str = _DEFAULT_MESSAGE) -> None:
        super().__init__(message)


class AdapterConnectionError(GliaBaseError):
    """
    Raised when a concrete adapter fails to establish or re-establish
    a connection to its upstream data source (vector DB, graph DB,
    relational DB, etc.).

    Wraps the underlying driver exception via the ``__cause__`` chain so
    callers can inspect the original error if needed::

        try:
            self._client.connect()
        except DriverError as exc:
            raise AdapterConnectionError(
                f"Failed to connect to vector store at '{url}'."
            ) from exc
    """


# ---------------------------------------------------------------------------
# Schema / index layer
# ---------------------------------------------------------------------------

class SchemaValidationError(GliaBaseError):
    """
    Raised by SchemaBuilder when the provided field definitions are
    structurally invalid (e.g. unsupported field type, duplicate field
    name, mismatched vector dimensions).

    Raised *before* any network call is made so the developer gets
    fast feedback during initialisation.
    """


# ---------------------------------------------------------------------------
# Invalidation layer
# ---------------------------------------------------------------------------

class InvalidationError(GliaBaseError):
    """
    Raised by CacheInvalidator when a ``delete_by_tag()`` operation
    cannot be completed — for example, if the RediSearch query itself
    fails or a partial delete leaves the index in an inconsistent state.

    A successful call that simply finds zero matching entries does NOT
    raise this exception; it returns 0 instead (idempotent behaviour).
    """