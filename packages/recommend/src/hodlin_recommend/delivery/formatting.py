"""Message formatting — pure functions, and the escaping boundary (T9).

Messages use Telegram HTML parse mode, so every dynamic field is escaped
*here*, at the one place text becomes markup. The reasoning is LLM prose over
untrusted headlines (T7): it may quote anything, and none of it may become
tags. What stays literal markup is only what this module writes itself.
"""

from html import escape

from hodlin_recommend.domain.models import Anomaly, Explanation


def format_anomaly(anomaly: Anomaly, explanation: Explanation) -> str:
    """One alert: the move in numbers, then the LLM's why, then lineage."""
    arrow = (
        "\N{CHART WITH UPWARDS TREND}"
        if anomaly.direction == "up"
        else ("\N{CHART WITH DOWNWARDS TREND}")
    )
    news_cited = sum(1 for ref in explanation.evidence if ref.kind == "news")
    lines = [
        f"{arrow} <b>{escape(anomaly.symbol)}</b> {escape(anomaly.interval)} "
        f"bar {anomaly.bar_ts:%Y-%m-%d %H:%M} UTC",
        f"move {anomaly.return_pct:+f}% \N{MIDDLE DOT} z-score {anomaly.z_score} "
        f"\N{MIDDLE DOT} baseline {anomaly.window} bars",
        "",
        escape(explanation.reasoning),
        "",
        f"<i>{news_cited} news source(s) cited \N{MIDDLE DOT} "
        f"{escape(explanation.model_version)}</i>",
    ]
    return "\n".join(lines)


def format_status(latest: tuple[Anomaly, Explanation] | None) -> str:
    """The reply to any allowlisted inbound message: the newest explained
    anomaly, or an honest 'nothing yet'."""
    if latest is None:
        return "No explained anomalies yet \N{EM DASH} you'll be notified here when one lands."
    anomaly, explanation = latest
    return "Latest anomaly:\n\n" + format_anomaly(anomaly, explanation)
