"""Ingestion orchestration — the layer that owns I/O and transactions.

Pulls data through the connector Protocols, persists it via repositories, and
runs the pure domain math over what's stored. Backfill (T5) handles cold start;
the scheduler and per-tick jobs arrive in T8.
"""
