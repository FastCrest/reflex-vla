"""Reflex — Deploy any VLA model to any edge hardware. One command."""

__version__ = "0.3.2"

# Heavy submodules (validate_roundtrip pulls in torch) are lazy-loaded so that
# `reflex --version`, `reflex --help`, `reflex chat`, etc. don't pay the
# 700ms+ torch-import cost on every invocation. Importers like
# `from reflex import ValidateRoundTrip` still work — the __getattr__ hook
# imports on first access.
__all__ = [
    "__version__",
    "ValidateRoundTrip",
    "SUPPORTED_MODEL_TYPES",
    "UNSUPPORTED_MODEL_MESSAGE",
    "load_fixtures",
]


def __getattr__(name: str):
    if name in {"ValidateRoundTrip", "SUPPORTED_MODEL_TYPES", "UNSUPPORTED_MODEL_MESSAGE"}:
        from reflex.validate_roundtrip import (
            SUPPORTED_MODEL_TYPES,
            UNSUPPORTED_MODEL_MESSAGE,
            ValidateRoundTrip,
        )
        return {
            "ValidateRoundTrip": ValidateRoundTrip,
            "SUPPORTED_MODEL_TYPES": SUPPORTED_MODEL_TYPES,
            "UNSUPPORTED_MODEL_MESSAGE": UNSUPPORTED_MODEL_MESSAGE,
        }[name]
    if name == "load_fixtures":
        from reflex.fixtures import load_fixtures
        return load_fixtures
    raise AttributeError(f"module 'reflex' has no attribute {name!r}")
