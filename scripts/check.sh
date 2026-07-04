#!/usr/bin/env bash
# One-command quality gate — the same checks CI runs, bundled for local/agent use.
# Exits non-zero on the first failure so it's a single green/red signal.
#
#   ./scripts/check.sh            # full gate
#   HODLIN_TEST_DATABASE_URL=... ./scripts/check.sh   # include DB integration tests
#
# Integration tests skip automatically unless Docker or HODLIN_TEST_DATABASE_URL
# is available (see tests/integration/conftest.py).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff (lint) =="       && uv run ruff check .
echo "== ruff (format) =="     && uv run ruff format --check .
echo "== mypy =="              && uv run mypy
echo "== import-linter =="     && uv run lint-imports
echo "== pytest =="            && uv run pytest

echo "✅ all checks passed"
