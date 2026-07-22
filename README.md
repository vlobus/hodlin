# hodlin

Market-context intelligence with a human approval gate. The **recommend** domain
ingests price bars and news, flags statistically unusual moves with a rolling
z-score, explains each one with a single LLM call over structured evidence, and
delivers the alert to Telegram. It holds no keys and moves no money — a separate
**execute** domain (M2) owns that authority, reachable only through a signed
proposal. *Not financial advice, but HODL.*

## Run it

You need Docker, and — for the live explanation and delivery — an
[Anthropic API key](https://console.anthropic.com) and a Telegram bot token
(from [@BotFather](https://t.me/botfather)) plus your numeric chat id (from
[@userinfobot](https://t.me/userinfobot)).

```sh
cp .env.example .env      # fill in ANTHROPIC_API_KEY and the TELEGRAM_* values
docker compose up         # builds the image, applies migrations, starts serving
```

In the default demo mode, prices come from a committed seed CSV and news is
skipped, so **only Anthropic and Telegram need real credentials** (the
Finnhub/Massive placeholders can stay as-is). On startup the app backfills the
seed bars, detects the demo anomaly (a sharp late-June BTC drop), explains it,
and messages it to your chat. The first boot downloads the FinBERT model
(~440 MB, cached in a volume afterward), so give it a couple of minutes; once
`http://localhost:8000/health/ready` returns `200`, the pipeline is live.

**See it immediately** (one-shot, no waiting for the scheduler):

```sh
docker compose run --rm app python -m hodlin_recommend.demo
```

This runs backfill → explain → notify once and prints each step, delivering the
anomaly to Telegram in one command.

## Develop

The project is a [uv](https://docs.astral.sh/uv/) workspace of three packages —
`contracts` (shared frozen types), `recommend`, and `execute` (M2 stub). The
full local gate — ruff, mypy (strict), import-linter, and pytest (unit +
integration against a real Postgres via testcontainers) — runs with:

```sh
uv sync
./scripts/check.sh
```

Integration tests need Docker for Postgres; they skip cleanly if it is
unavailable, or point them at an existing database with
`HODLIN_TEST_DATABASE_URL`.

## Health

- `GET /health/live` — the process is up (no dependencies checked).
- `GET /health/ready` — the database answers and the scheduler is running.
- `POST /v1/sentiment` — score one text with the served FinBERT model.
