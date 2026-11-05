"""OpenAPI documentation helpers: error responses, tag metadata, and schema post-processing.

FastAPI generates OpenAPI 3.1 (JSON Schema 2020-12). The published spec targets OpenAPI 3.0.3 for
wider tool support, so the generated document is converted (see `_to_openapi_30`).
"""

import json
import re
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from rag_service.models.schemas import (
    ErrorResponse,
    QueryStreamDelta,
    QueryStreamDone,
    QueryStreamSources,
)

# Models that are not response bodies but are documented as components (SSE event payloads)
_EXTRA_COMPONENTS = (QueryStreamSources, QueryStreamDelta, QueryStreamDone)

API_DESCRIPTION = """\
REST API of rag-service: manage documents in **datasets**, index them with a selectable parser
(Docling or MinerU), and search them.

**Authorization** is not performed by this service. An external gateway authenticates callers and
authorizes per dataset; every dataset-scoped endpoint carries the dataset id in its path.

**Model API key**: requests that call LiteLLM send the caller's key in `X-Model-API-Key`: dataset
creation, indexing, and search for datasets with a `litellm` embedding model, and every /query.
The key is never stored. The LiteLLM base URL is server configuration and never comes from requests.

**Errors** always have the shape `{"error": {"code": "...", "message": "..."}}`.

See API.md in the repository for a guide with end-to-end examples.
"""

OPENAPI_VERSION = "3.0.3"

TAGS: list[dict[str, str]] = [
    {"name": "datasets", "description": "Datasets: the unit of access control; each fixes its embedding model."},
    {"name": "documents", "description": "Upload, list, download, and delete documents in a dataset."},
    {"name": "indexing", "description": "Indexing jobs (parse → chunk → embed → store) and parser status."},
    {"name": "search", "description": "Vector search and question answering within one dataset."},
    {"name": "ops", "description": "Health and readiness probes."},
]

_STATUS_TEXT = {
    400: "Bad request",
    404: "Not found",
    409: "Conflict",
    413: "Payload too large",
    422: "Invalid request",
    502: "Bad gateway (LiteLLM / model provider error)",
    503: "Service unavailable",
    504: "Gateway timeout (model provider)",
}

_EXAMPLE_MESSAGES = {
    "missing_model_api_key": "X-Model-API-Key header is required",
    "dataset_not_found": "dataset '52967db904d44a09b33f88b3c9f64a6a' not found",
    "document_not_found": "document 'bbd713912b614848a8f0def3f73aa304' not found",
    "job_not_found": "job '2d18380d5e6548ed868383387a606e3b' not found",
    "content_missing": "stored file is missing",
    "not_parsed": "document has not been parsed yet",
    "external_ref_conflict": "external_ref 'team-legal-001' already exists",
    "duplicate_document": "identical file already exists as document 'bbd713912b614848a8f0def3f73aa304'",
    "job_in_progress": "document 'bbd713912b614848a8f0def3f73aa304' already has an active indexing job",
    "job_not_active": "job is already succeeded",
    "file_too_large": "file exceeds 100 MB limit",
    "validation_error": "query: String should have at least 1 character",
    "embedding_model_required": "embedding.model is required for litellm",
    "invalid_embedding_model": "model gateway returned HTTP 400: Invalid model name passed in model=titan-embed-v9",
    "embedding_dim_too_large": "model returns 3072-dimensional vectors; at most 2000 are supported",
    "invalid_chunk_overlap": "chunk_overlap must be smaller than chunk_size",
    "invalid_metadata": "metadata key 'job_id' is reserved",
    "invalid_filename": "uploaded file must have a filename",
    "invalid_json": "form field 'metadata' is not valid JSON",
    "unsupported_parser_or_format": "parser 'docling' does not support '.exe' files",
    "invalid_parser_options": "unsupported docling options: ['to_formats'] (allowed: [...])",
    "invalid_filter": "invalid metadata filter key 'a b'",
    "top_k_too_large": "top_k must be <= 50",
    "max_source_tokens_too_large": "max_source_tokens must be <= 100000",
    "parser_unavailable": "parser 'docling' is unavailable",
    "db_unavailable": "database is not reachable",
    "upstream_auth_failed": "model provider rejected the API key",
    "upstream_error": "model gateway returned HTTP 429: rate limit exceeded",
    "upstream_unreachable": "could not reach the model gateway",
    "litellm_not_configured": "LITELLM_BASE_URL is not configured",
    "upstream_timeout": "model provider did not respond within 120s",
}


def errors(spec: dict[int, list[str]]) -> dict[int | str, dict[str, Any]]:
    """Build a FastAPI `responses=` mapping for error statuses and their error codes."""
    out: dict[int | str, dict[str, Any]] = {}
    for status, codes in spec.items():
        out[status] = {
            "model": ErrorResponse,
            "description": f"{_STATUS_TEXT.get(status, 'Error')}. Error codes: "
            + ", ".join(f"`{c}`" for c in codes),
            "content": {
                "application/json": {
                    "examples": {
                        c: {"value": {"error": {"code": c, "message": _EXAMPLE_MESSAGES.get(c, c)}}}
                        for c in codes
                    }
                }
            },
        }
    return out


def install_openapi(app: FastAPI) -> None:
    """Replace FastAPI's default validation error schema with the service's error envelope."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            openapi_version=OPENAPI_VERSION,
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=TAGS,
            servers=[{"url": "http://localhost:8000", "description": "Local docker compose"}],
        )
        error_ref = "#/components/schemas/ErrorResponse"
        for path_item in schema.get("paths", {}).values():
            for op in path_item.values():
                resp = op.get("responses", {}).get("422")
                if not resp:
                    continue
                content = resp.setdefault("content", {}).setdefault("application/json", {})
                if content.get("schema", {}).get("$ref", "").endswith("HTTPValidationError"):
                    content["schema"] = {"$ref": error_ref}
                    resp["description"] = "Invalid request. Error codes: `validation_error`"
                    content["examples"] = {
                        "validation_error": {
                            "value": {
                                "error": {
                                    "code": "validation_error",
                                    "message": _EXAMPLE_MESSAGES["validation_error"],
                                }
                            }
                        }
                    }
        components = schema.get("components", {}).get("schemas", {})
        for model in _EXTRA_COMPONENTS:
            extra = model.model_json_schema(ref_template="#/components/schemas/{model}")
            for name, sub in extra.pop("$defs", {}).items():
                components.setdefault(name, sub)
            components[model.__name__] = extra
        components.pop("HTTPValidationError", None)
        components.pop("ValidationError", None)
        schema = _to_openapi_30(_rename_schemas(schema))
        app.openapi_schema = schema
        return schema

    app.openapi = custom_openapi  # type: ignore[method-assign]


# Generated component names → readable names
_RENAMES = {
    "Body_upload_document": "DocumentUploadForm",
    "Page_DatasetOut_": "DatasetPage",
    "Page_DocumentOut_": "DocumentPage",
    "Page_JobOut_": "JobPage",
}


def _rename_schemas(schema: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(schema)
    for old, new in _RENAMES.items():
        text = re.sub(rf'(#/components/schemas/){re.escape(old)}"', rf'\g<1>{new}"', text)
    out = json.loads(text)
    comps = out.get("components", {}).get("schemas", {})
    for old, new in _RENAMES.items():
        if old in comps:
            comps[new] = comps.pop(old)
            comps[new]["title"] = new
    out["components"]["schemas"] = dict(sorted(comps.items()))
    return out


# ---------------------------------------------------------------------------
# OpenAPI 3.1 → 3.0.3 conversion
# ---------------------------------------------------------------------------

_SUBSCHEMA_LISTS = ("allOf", "anyOf", "oneOf")
_SUBSCHEMA_KEYS = ("items", "not", "additionalProperties")


def _convert_schema(schema: Any) -> Any:
    """Convert one JSON Schema 2020-12 schema to an OpenAPI 3.0 Schema Object (recursively).

    - anyOf/oneOf with {"type": "null"} → `nullable: true` (a lone $ref is wrapped in allOf,
      since 3.0 ignores siblings of $ref)
    - `examples: [...]` → `example: <first>`
    - `const` → single-value `enum`
    - `contentMediaType` (file uploads) → `format: binary`
    """
    if not isinstance(schema, dict):
        return schema
    s = dict(schema)

    for key in _SUBSCHEMA_LISTS:
        if key in s:
            s[key] = [_convert_schema(x) for x in s[key]]
    for key in _SUBSCHEMA_KEYS:
        if isinstance(s.get(key), dict):
            s[key] = _convert_schema(s[key])
    if isinstance(s.get("properties"), dict):
        s["properties"] = {k: _convert_schema(v) for k, v in s["properties"].items()}

    for key in ("anyOf", "oneOf"):
        variants = s.get(key)
        if not variants:
            continue
        # sub-schemas are already converted, so a null variant may appear as {"nullable": True}
        non_null = [v for v in variants if v not in ({"type": "null"}, {"nullable": True})]
        if len(non_null) == len(variants):
            continue
        s.pop(key)
        s["nullable"] = True
        if len(non_null) == 1:
            only = non_null[0]
            if "$ref" in only:
                s["allOf"] = [only]
            else:
                s = {**only, **s}
        elif non_null:
            s[key] = non_null

    if s.get("type") == "null":
        s.pop("type")
        s["nullable"] = True
    if "examples" in s and isinstance(s["examples"], list):
        examples = s.pop("examples")
        if examples:
            s.setdefault("example", examples[0])
    if "const" in s:
        s["enum"] = [s.pop("const")]
    if "contentMediaType" in s:
        s.pop("contentMediaType")
        s.pop("contentEncoding", None)
        s["format"] = "binary"
    return s


def _convert_content(content: dict[str, Any]) -> None:
    for media in content.values():
        if "schema" in media:
            media["schema"] = _convert_schema(media["schema"])


def _to_openapi_30(spec: dict[str, Any]) -> dict[str, Any]:
    spec["openapi"] = OPENAPI_VERSION
    comps = spec.get("components", {}).get("schemas", {})
    for name, schema in comps.items():
        comps[name] = _convert_schema(schema)
    for path_item in spec.get("paths", {}).values():
        for op in path_item.values():
            for param in op.get("parameters", []):
                if "schema" in param:
                    param["schema"] = _convert_schema(param["schema"])
            if "requestBody" in op:
                _convert_content(op["requestBody"].get("content", {}))
            for resp in op.get("responses", {}).values():
                _convert_content(resp.get("content", {}))
    return spec
