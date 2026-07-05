"""FastAPI serving surface — model-agnostic HTTP endpoints.

Thin handlers only: routes validate DTOs, dispatch to injected domain objects,
and translate results back to DTOs. Anything model-shaped arrives through a
Protocol on ``app.state`` (wired by the composition root), so a classifier, a
transformer, or an LLM all serve through the same pattern.
"""
