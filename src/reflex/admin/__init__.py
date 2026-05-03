"""Reflex admin CLI — license issuance + revocation against the license worker.

Operator-only commands; not exposed via the public `reflex` CLI. Run as
Python modules:

    python -m reflex.admin.issue_license --customer-id alice@bigco.com --tier pro --expires-in 30
    python -m reflex.admin.revoke_license --license-id lic_xxx --reason refund
    python -m reflex.admin.list_licenses

Auth: set REFLEX_ADMIN_TOKEN to the bearer token configured on the worker
(via `wrangler secret put ADMIN_TOKEN`). Set REFLEX_LICENSE_ENDPOINT to the
deployed worker URL (e.g., https://reflex-licenses.fastcrest.workers.dev).
"""
from __future__ import annotations

__all__: list[str] = []
