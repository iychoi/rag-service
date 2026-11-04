"""openapi.yaml must match the spec generated from the code."""

import importlib.util
from pathlib import Path

import yaml
from openapi_spec_validator import OpenAPIV30SpecValidator, validate

from rag_service.api.docs import _convert_schema

ROOT = Path(__file__).resolve().parents[1]


def _render() -> str:
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render()


def test_openapi_yaml_is_up_to_date() -> None:
    committed = (ROOT / "openapi.yaml").read_text(encoding="utf-8")
    assert committed == _render(), "openapi.yaml is stale; run: python scripts/export_openapi.py"


def test_openapi_yaml_is_valid_openapi_30() -> None:
    spec = yaml.safe_load((ROOT / "openapi.yaml").read_text(encoding="utf-8"))
    assert spec["openapi"] == "3.0.3"
    validate(spec, cls=OpenAPIV30SpecValidator)


def test_convert_schema_31_to_30() -> None:
    assert _convert_schema({"anyOf": [{"type": "string"}, {"type": "null"}], "title": "X"}) == {
        "type": "string",
        "title": "X",
        "nullable": True,
    }
    assert _convert_schema({"anyOf": [{"$ref": "#/c/A"}, {"type": "null"}]}) == {
        "nullable": True,
        "allOf": [{"$ref": "#/c/A"}],
    }
    assert _convert_schema(
        {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
    ) == {"anyOf": [{"type": "string"}, {"type": "integer"}], "nullable": True}
    assert _convert_schema({"type": "object", "examples": [{"a": 1}]}) == {
        "type": "object",
        "example": {"a": 1},
    }
    assert _convert_schema({"const": "x"}) == {"enum": ["x"]}
    assert _convert_schema({"type": "string", "contentMediaType": "application/octet-stream"}) == {
        "type": "string",
        "format": "binary",
    }
