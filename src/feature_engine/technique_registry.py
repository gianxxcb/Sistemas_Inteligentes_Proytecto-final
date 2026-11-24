"""Validated reader for the declarative trading-technique registry."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union


DEFAULT_REGISTRY_PATH = Path(__file__).with_name("technique_registry.json")


def load_technique_registry(
    registry_path: Optional[Union[Path, str]] = None,
) -> dict[str, Any]:
    path = Path(registry_path) if registry_path is not None else DEFAULT_REGISTRY_PATH
    with path.open(encoding="utf-8") as registry_file:
        registry = json.load(registry_file)

    techniques = registry.get("techniques")
    if registry.get("version") != "2.0" or not isinstance(techniques, list) or not techniques:
        raise ValueError(f"Registro de técnicas inválido o versión no soportada: {path}")

    seen = set()
    required = {"id", "expert", "role", "formula_ref", "params", "outputs", "applicability", "invalidation"}
    for technique in techniques:
        if not isinstance(technique, dict) or not required.issubset(technique):
            raise ValueError(f"Definición de técnica incompleta en {path}")
        technique_id = technique["id"]
        if not isinstance(technique_id, str) or not technique_id or technique_id in seen:
            raise ValueError(f"Identificador de técnica inválido o duplicado: {technique_id}")
        seen.add(technique_id)
        if not isinstance(technique["params"], dict):
            raise ValueError(f"Parámetros inválidos en la técnica {technique_id}")
        for name, spec in technique["params"].items():
            if not isinstance(spec, dict) or "default" not in spec or not isinstance(spec.get("search"), list):
                raise ValueError(f"Parámetro {name} incompleto en la técnica {technique_id}")

    return registry


def technique_defaults(registry: dict[str, Any], technique_id: str) -> dict[str, Any]:
    """Return a copy of the declared defaults for one registered technique."""
    for technique in registry["techniques"]:
        if technique["id"] == technique_id:
            return {
                name: deepcopy(spec["default"])
                for name, spec in technique["params"].items()
            }
    raise KeyError(f"Técnica no registrada: {technique_id}")
