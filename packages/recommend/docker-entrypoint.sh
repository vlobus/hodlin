#!/bin/sh
# Apply schema migrations, then run whatever command was given (serve by
# default; `... python -m hodlin_recommend.demo` for the one-shot demo).
# Migrations are idempotent, so running on every boot is safe.
set -e

echo "hodlin: applying migrations…"
alembic upgrade head

echo "hodlin: starting → $*"
exec "$@"
