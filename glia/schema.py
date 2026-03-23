"""
glia/schema.py
--------------
RediSearch index schema construction for the Glia library.

SchemaBuilder is a pure configuration object — it performs *no* I/O.
It is instantiated internally by GliaManager and is not part of the
public API surface that developers are expected to import directly.

Keeping schema logic here (rather than in manager.py) means:
- It is independently unit-testable without a live Redis connection.
- New field types can be added in a single place.
- GliaManager stays focused on connection and request lifecycle.

Dependencies
------------
- ``redisvl`` (RedisVL) for ``IndexSchema`` and field type definitions.
- ``glia.exceptions.SchemaValidationError`` for invalid field configs.
  (No other glia modules are imported.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from glia.exceptions import SchemaValidationError

# ``redisvl`` types are imported at the top level so import errors surface
# immediately at library load time rather than at first use.
try:
    from redisvl.schema import IndexSchema  # type: ignore[import]
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Glia requires the 'redisvl' package.  "
        "Install it with: pip install redisvl"
    ) from exc


# ---------------------------------------------------------------------------
# Default schema constants
# ---------------------------------------------------------------------------

#: Name used for the RediSearch index when none is supplied.
DEFAULT_INDEX_NAME: str = "llmcache"

#: Key prefix under which JSON cache documents are stored in Redis.
DEFAULT_KEY_PREFIX: str = "glia:"

#: Default embedding dimension (matches textembedding-gecko@003).
DEFAULT_VECTOR_DIMS: int = 768

#: Distance metric used for vector similarity search.
DEFAULT_DISTANCE_METRIC: str = "cosine"

#: Algorithm used for approximate nearest-neighbour search.
DEFAULT_VECTOR_ALGORITHM: str = "flat"

#: Vector datatype sent to RediSearch.
DEFAULT_VECTOR_DATATYPE: str = "float32"


# ---------------------------------------------------------------------------
# SchemaBuilder
# ---------------------------------------------------------------------------

class SchemaBuilder:
    """
    Constructs a ``redisvl.IndexSchema`` for Glia's cache index.

    The *default* schema always includes four fields:

    +-----------------+--------+----------------------------------------------+
    | Field name      | Type   | Purpose                                      |
    +=================+========+==============================================+
    | ``prompt``      | text   | Human-readable prompt string (full-text).    |
    +-----------------+--------+----------------------------------------------+
    | ``response``    | tag    | Exact cached answer string.                  |
    +-----------------+--------+----------------------------------------------+
    | ``source_id``   | tag    | Dependency tag — targeted invalidation key.  |
    +-----------------+--------+----------------------------------------------+
    | ``prompt_vector``| vector | Embedding for KNN similarity search.         |
    +-----------------+--------+----------------------------------------------+

    Additional tag or numeric fields can be appended via ``custom_fields``
    at construction time, or programmatically with ``add_tag_field()`` and
    ``add_numeric_field()`` before calling ``build()``.

    Parameters
    ----------
    index_name:
        Name of the RediSearch index.  Defaults to ``DEFAULT_INDEX_NAME``.
    key_prefix:
        Redis key namespace for stored JSON documents.
        Defaults to ``DEFAULT_KEY_PREFIX``.
    vector_dims:
        Dimensionality of the prompt embedding vectors.
        Must match the injected vectorizer's output dimension.
        Defaults to ``DEFAULT_VECTOR_DIMS`` (768).
    custom_fields:
        Optional list of additional field definition dicts to merge into
        the default schema.  Each dict must contain at least
        ``{"name": str, "type": "tag" | "numeric"}``.

    Raises
    ------
    SchemaValidationError
        If ``vector_dims`` is not a positive integer, if a custom field
        definition is structurally invalid, or if a duplicate field name
        is detected.
    """

    # Names owned by the built-in schema — duplicates in custom fields
    # are rejected immediately at registration time, not silently overwritten.
    _BUILTIN_FIELD_NAMES: frozenset = frozenset(
        {"prompt", "response", "source_id", "prompt_vector"}
    )

    # The only field types accepted via custom_fields / add_*_field helpers.
    _ALLOWED_CUSTOM_TYPES: frozenset = frozenset({"tag", "numeric"})

    def __init__(
        self,
        index_name: str = DEFAULT_INDEX_NAME,
        key_prefix: str = DEFAULT_KEY_PREFIX,
        vector_dims: int = DEFAULT_VECTOR_DIMS,
        custom_fields: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Store configuration and eagerly validate inputs.

        Validation runs here — before any network call — so the developer
        gets immediate feedback during ``GliaManager`` initialisation.
        A ``SchemaBuilder`` that survives ``__init__`` is guaranteed to
        produce a valid ``IndexSchema`` when ``build()`` is called.
        """
        if not isinstance(vector_dims, int) or vector_dims <= 0:
            raise SchemaValidationError(
                f"vector_dims must be a positive integer; got {vector_dims!r}."
            )

        self.index_name: str = index_name
        self.key_prefix: str = key_prefix
        self.vector_dims: int = vector_dims

        # Internal registry: list of validated field dicts ready for
        # injection into the redisvl schema dict at build() time.
        self._custom_fields: List[Dict[str, Any]] = []

        # Validate and register any fields supplied at construction time.
        for field_def in custom_fields or []:
            self._validate_and_append(field_def)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _all_registered_names(self) -> frozenset:
        """Return the union of built-in and currently registered custom names."""
        custom_names = frozenset(f["name"] for f in self._custom_fields)
        return self._BUILTIN_FIELD_NAMES | custom_names

    def _validate_and_append(self, field_def: Dict[str, Any]) -> None:
        """
        Validate *field_def* and append it to ``_custom_fields``.

        Centralises the three checks shared by ``__init__``, ``add_tag_field``,
        and ``add_numeric_field``:

        1. The dict must have a non-empty ``"name"`` key of type ``str``.
        2. The ``"type"`` key must be one of ``_ALLOWED_CUSTOM_TYPES``.
        3. The name must not collide with a built-in or already-registered field.

        Parameters
        ----------
        field_def:
            Raw field definition dict from the caller.

        Raises
        ------
        SchemaValidationError
            On any of the three validation failures above.
        """
        name = field_def.get("name")
        if not name or not isinstance(name, str):
            raise SchemaValidationError(
                f"Custom field definition must include a non-empty string 'name'; "
                f"got {field_def!r}."
            )

        field_type = field_def.get("type")
        if field_type not in self._ALLOWED_CUSTOM_TYPES:
            raise SchemaValidationError(
                f"Custom field '{name}' has unsupported type {field_type!r}. "
                f"Allowed types: {sorted(self._ALLOWED_CUSTOM_TYPES)}."
            )

        if name in self._all_registered_names():
            raise SchemaValidationError(
                f"Field name '{name}' conflicts with a built-in or already-registered "
                f"field.  Built-in names: {sorted(self._BUILTIN_FIELD_NAMES)}."
            )

        # Store a clean, minimal copy — strip any unexpected keys from caller.
        self._custom_fields.append({"name": name, "type": field_type})

    # ------------------------------------------------------------------
    # Public field-builder helpers
    # ------------------------------------------------------------------

    def add_tag_field(self, name: str) -> None:
        """
        Append a TAG field definition to the custom fields list.

        TAG fields support exact-match filtering in ``check()`` via the
        ``filter=`` parameter and targeted invalidation in
        ``CacheInvalidator.delete_by_tag()``.

        Parameters
        ----------
        name:
            Field name.  Must be unique across all schema fields.

        Raises
        ------
        SchemaValidationError
            If *name* conflicts with a built-in field or a previously
            added custom field.
        """
        self._validate_and_append({"name": name, "type": "tag"})

    def add_numeric_field(self, name: str) -> None:
        """
        Append a NUMERIC field definition to the custom fields list.

        Numeric fields support range-filter queries in ``check()``
        via the ``filter=`` parameter.

        Parameters
        ----------
        name:
            Field name.  Must be unique across all schema fields.

        Raises
        ------
        SchemaValidationError
            If *name* conflicts with a built-in field or a previously
            added custom field.
        """
        self._validate_and_append({"name": name, "type": "numeric"})

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> IndexSchema:
        """
        Assemble and return the ``redisvl.IndexSchema`` object.

        Constructs the full schema dict and passes it to
        ``IndexSchema.from_dict()``.  The four built-in fields are always
        included first; custom fields follow in registration order.

        Returns
        -------
        IndexSchema
            A fully configured RedisVL schema ready to be passed to
            ``GliaManager`` for index creation.

        Raises
        ------
        SchemaValidationError
            If redisvl's own Pydantic validation rejects the assembled
            configuration (e.g. an unrecognised algorithm string).  The
            original ``ValidationError`` is chained via ``__cause__``.
        """
        # ------------------------------------------------------------------
        # 1. Built-in field definitions
        # ------------------------------------------------------------------
        builtin_fields: List[Dict[str, Any]] = [
            # Full-text searchable prompt string.
            {
                "name": "prompt",
                "type": "text",
            },
            # Exact cached response — stored as a TAG so it survives
            # round-trips through RediSearch without tokenisation.
            {
                "name": "response",
                "type": "tag",
            },
            # Dependency tag: the targeted invalidation key.
            # Stored as TAG so delete_by_tag() can issue an exact-match query.
            {
                "name": "source_id",
                "type": "tag",
            },
            # Prompt embedding vector for KNN similarity search.
            {
                "name": "prompt_vector",
                "type": "vector",
                "attrs": {
                    "algorithm":       DEFAULT_VECTOR_ALGORITHM,
                    "dims":            self.vector_dims,
                    "distance_metric": DEFAULT_DISTANCE_METRIC,
                    "datatype":        DEFAULT_VECTOR_DATATYPE,
                },
            },
        ]

        # ------------------------------------------------------------------
        # 2. Merge custom fields (already validated at registration time)
        # ------------------------------------------------------------------
        all_fields: List[Dict[str, Any]] = builtin_fields + list(self._custom_fields)

        # ------------------------------------------------------------------
        # 3. Assemble the top-level schema dict and delegate to redisvl
        # ------------------------------------------------------------------
        schema_dict: Dict[str, Any] = {
            "index": {
                "name":         self.index_name,
                "prefix":       self.key_prefix,
                "storage_type": "json",   # BDUF: entries stored as JSON docs
            },
            "fields": all_fields,
        }

        try:
            return IndexSchema.from_dict(schema_dict)
        except Exception as exc:
            raise SchemaValidationError(
                f"redisvl rejected the assembled schema for index "
                f"'{self.index_name}': {exc}"
            ) from exc