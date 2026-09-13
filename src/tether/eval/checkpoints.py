"""Checkpoint resolution and immutable identity for task-success evaluation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


class CheckpointError(ValueError):
    """The requested checkpoint cannot be identified without guessing."""


@dataclass(frozen=True)
class CheckpointSpec:
    kind: str
    source: str
    identity: str
    base: str | None = None
    revision: str | None = None
    files: tuple[dict, ...] = ()
    base_revision: str | None = None

    def to_dict(self) -> dict:
        value = asdict(self)
        value["files"] = list(self.files)
        return value


def _file_manifest(path: Path) -> tuple[dict, ...]:
    rows = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.is_symlink():
            continue
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append({
            "path": str(item.relative_to(path)),
            "bytes": item.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    return tuple(rows)


def resolve_checkpoint(
    source: str | Path,
    *,
    kind: str = "auto",
    base: str | None = None,
    revision: str | None = None,
    base_revision: str | None = None,
) -> CheckpointSpec:
    """Resolve a checkpoint without downloading it or selecting a fallback."""
    raw = str(source)
    path = Path(raw).expanduser()
    if path.exists():
        if not path.is_dir():
            raise CheckpointError("A local checkpoint must be a directory.")
        adapter_config = path / "adapter_config.json"
        adapter_weights = list(path.glob("adapter_model.*"))
        detected = "smolvla-lora" if adapter_config.is_file() and adapter_weights else "full"
        if kind != "auto" and kind != detected:
            raise CheckpointError(f"Checkpoint kind {kind!r} does not match detected kind {detected!r}.")
        resolved_base = base
        if detected == "smolvla-lora" and not resolved_base:
            try:
                resolved_base = json.loads(adapter_config.read_text()).get("base_model_name_or_path")
            except (OSError, ValueError):
                resolved_base = None
        if detected == "smolvla-lora" and not resolved_base:
            raise CheckpointError("A SmolVLA LoRA adapter requires an explicit base checkpoint.")
        if detected == "smolvla-lora" and resolved_base and not Path(resolved_base).expanduser().is_dir() and not base_revision:
            raise CheckpointError("A remote LoRA base requires --adapter-base-revision for reproducible evidence.")
        files = _file_manifest(path)
        if not files:
            raise CheckpointError("The local checkpoint directory contains no files.")
        identity_input = json.dumps({"kind": detected, "base": resolved_base, "base_revision": base_revision, "files": files}, sort_keys=True)
        identity = "sha256:" + hashlib.sha256(identity_input.encode()).hexdigest()
        return CheckpointSpec(detected, str(path.resolve()), identity, resolved_base, revision, files, base_revision)

    if raw.startswith(("/", "./", "../", "~")):
        raise CheckpointError(f"Checkpoint path not found: {raw}")

    if kind == "smolvla-lora":
        if not base:
            raise CheckpointError("A remote SmolVLA LoRA adapter requires an explicit base checkpoint.")
        if not Path(base).expanduser().is_dir() and not base_revision:
            raise CheckpointError("A remote LoRA base requires --adapter-base-revision for reproducible evidence.")
        detected = kind
    else:
        detected = "full"
    remote = raw.removeprefix("hf://")
    if "/" not in remote:
        raise CheckpointError(f"Checkpoint path does not exist and is not a Hugging Face repository: {raw}")
    if not revision:
        raise CheckpointError("A Hugging Face checkpoint requires --checkpoint-revision for reproducible evidence.")
    identity = (
        f"hf-lora:{remote}@{revision}+{base}@{base_revision}"
        if detected == "smolvla-lora" else f"hf:{remote}@{revision}"
    )
    return CheckpointSpec(detected, remote, identity, base, revision, base_revision=base_revision)
