"""Composition root — the one place concretes are constructed and wired.

Builds the expensive pieces exactly once (FinBERT downloads ~440 MB on first
run, then loads from the HF cache) and hands them to the app factory. Grows
through T8-T10: settings/DB wiring, the scheduler, and the Telegram bot all
get composed here. Run with ``python -m hodlin_recommend.main``.
"""

import uvicorn

from hodlin_recommend.domain.sentiment import FinBertModel
from hodlin_recommend.serving.app import create_app


def main() -> None:
    app = create_app(sentiment_model=FinBertModel())
    # Loopback bind for local runs; T10's compose/Dockerfile overrides this.
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
