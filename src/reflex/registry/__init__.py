"""Curated model registry for `reflex models {list,pull,info}`.

Indexes VLA model checkpoints we have verified work with Reflex serving — saves
customers the "5-tab research session" of figuring out which HF repo to use.

The registry is shipped IN-PACKAGE (`src/reflex/registry/data.py`) rather than
queried from HF Hub at runtime. Reasons:

- Curation: every entry has been verified against our parity tests; HF tags can
  be applied by anyone, ours can't be spoofed
- Offline: `reflex models list` works without internet
- Pinning: each entry has a specific revision (commit sha) for reproducibility
- Zero rate-limit risk

Pull operations (`reflex models pull <id>`) DO hit HF Hub via huggingface_hub —
that's where the actual weights live. The registry only stores metadata.
"""

from reflex.registry.models import (
    ModelEntry,
    ModelBenchmark,
    REGISTRY,
    by_id,
    filter_models,
    list_families,
    list_devices,
)

__all__ = [
    "ModelEntry",
    "ModelBenchmark",
    "REGISTRY",
    "by_id",
    "filter_models",
    "list_families",
    "list_devices",
]
