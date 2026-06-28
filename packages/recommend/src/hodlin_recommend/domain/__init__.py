"""Pure domain logic — models and math with no I/O and no framework imports.

Nothing here knows about SQLAlchemy, FastAPI, or HTTP. The store translates
between these models and ORM rows; that boundary is what keeps the logic
testable in isolation.
"""
