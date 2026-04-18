"""Regression marker for `docker-image-smoke-test` GOALS.yaml gate.

Publishing + pullability evidence lives in
`reflex_context/docker_smoke_last_run.json`. That receipt is populated
by a successful run of `.github/workflows/docker-publish.yml` plus a
Modal `from_registry` pull smoke.

Full in-container `docker run` smoke is tracked as a v0.3 follow-up
because Modal's from_registry can't introspect an image whose
ENTRYPOINT intercepts subprocess execs (which is the correct customer-
facing behavior here). v0.3 will add a GitHub Actions docker-run smoke
step that runs after each release tag.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RECEIPT = Path(__file__).parent.parent / "reflex_context" / "docker_smoke_last_run.json"


def test_docker_smoke_receipt_passes():
    if not RECEIPT.exists():
        pytest.skip(
            "No docker smoke receipt. After `git tag vX.Y.Z && git push --tags`, "
            "check that docker-publish.yml succeeded and drop the result at "
            f"{RECEIPT}."
        )
    data = json.loads(RECEIPT.read_text())
    assert data.get("passed") is True, f"docker smoke did not pass: {data}"
    checks = data.get("checks", {})
    for name, entry in checks.items():
        assert entry.get("passed") is True, f"{name} failed: {entry}"
