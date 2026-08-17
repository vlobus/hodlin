"""Serve entry point — build the graph, run the ASGI app.

The wiring lives in ``composition.build_components`` (shared with the demo);
this module only turns those components into a FastAPI app and hands it to
uvicorn. Bind address comes from settings so local runs stay on loopback while
the container sets HOST=0.0.0.0. Run with ``python -m hodlin_recommend.main``.
"""

import uvicorn

from hodlin_recommend.composition import build_components
from hodlin_recommend.config import Settings
from hodlin_recommend.serving.app import create_app


def main() -> None:
    # Fields arrive from the environment at runtime; mypy can't see that.
    settings = Settings()  # type: ignore[call-arg]
    components = build_components(settings)

    app = create_app(
        sentiment_model=components.sentiment_model,
        inference_executor=components.inference_executor,
        scheduler=components.scheduler,
        poller=components.poller,
        session_factory=components.session_factory,
        resources=components.resources,
    )
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
