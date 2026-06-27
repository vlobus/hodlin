"""Smoke test: the three workspace packages import and the boundary is real.

Replaced by domain-level suites as T2-T10 land; for T1 it just proves the
skeleton is wired and gives CI a green test to run.
"""

import importlib


def test_packages_import() -> None:
    for name in ("hodlin_contracts", "hodlin_recommend", "hodlin_execute"):
        assert importlib.import_module(name) is not None
