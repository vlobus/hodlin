"""External data-source adapters behind swappable Protocols (D4/D13).

Each connector turns one provider's HTTP API into domain models
(``PriceBar`` / ``NewsItem``), rate-limited and retried, raising
``SourceUnavailable`` on exhaustion so ingestion jobs can degrade gracefully
instead of crashing. Downstream code depends on the Protocols in ``base``,
never on a concrete provider — swapping a source is a one-file change.
"""
