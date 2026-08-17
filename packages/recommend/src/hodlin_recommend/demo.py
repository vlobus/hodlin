"""Scripted demo (T10): run the pipeline once, immediately, end-to-end.

The serving app (``main.py``) does this on a schedule; the demo does it now so
a reviewer sees the whole flow in one command without waiting for ticks:

    backfill (seed bars -> detect the demo anomaly)
        -> explain (Anthropic writes the "why")
        -> notify (Telegram delivers it to the allowlisted chat)

Same wiring as production via ``build_components``; demo mode supplies the
offline seed bars, so only Anthropic + Telegram need real credentials. Assumes
migrations are already applied (the container entrypoint runs them). Run with
``python -m hodlin_recommend.demo``.
"""

import asyncio

from hodlin_recommend.composition import build_components
from hodlin_recommend.config import Settings
from hodlin_recommend.domain.asset_config import DEFAULT_ASSETS
from hodlin_recommend.ingest import jobs
from hodlin_recommend.ingest.jobs import JobOutcome


def _line(step: str, outcome: JobOutcome) -> str:
    detail = f" — {outcome.detail}" if outcome.detail else ""
    return f"[{outcome.status:>5}] {step}: {outcome.items} item(s){detail}"


async def run_demo(settings: Settings) -> None:
    components = build_components(settings)
    try:
        backfill = await jobs.run_backfill(
            components.session_factory, components.bar_source, DEFAULT_ASSETS
        )
        print(_line("backfill", backfill))

        explain = await jobs.explain_anomalies(
            components.session_factory,
            llm=components.llm,
            sentiment_model=components.sentiment_model,
            inference_executor=components.inference_executor,
        )
        print(_line("explain ", explain))

        notify = await jobs.notify_anomalies(
            components.session_factory,
            messenger=components.messenger,
            chat_id=components.chat_id,
        )
        print(_line("notify  ", notify))

        if notify.items:
            print(f"\nDelivered {notify.items} anomaly alert(s) to chat {components.chat_id}.")
        else:
            print("\nNothing delivered — see above (already notified? LLM/Telegram down?).")
    finally:
        components.inference_executor.shutdown(wait=True)
        await components.resources.aclose()


def main() -> None:
    settings = Settings()  # type: ignore[call-arg]  # env-provided at runtime
    asyncio.run(run_demo(settings))


if __name__ == "__main__":
    main()
