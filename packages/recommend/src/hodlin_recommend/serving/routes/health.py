"""Health probes — liveness and readiness are different questions.

``/health/live`` = "the process is up, don't restart me": no dependencies
touched. ``/health/ready`` = "my dependencies answer, send me traffic": the
database must respond and the scheduler (when one is wired) must be running.
An orchestrator restarts on failed liveness but only routes around failed
readiness — conflating them turns a database blip into a restart loop.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import text

router = APIRouter(prefix="/health")


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    problems: list[str] = []
    session_factory = request.app.state.session_factory
    if session_factory is None:
        problems.append("no database configured")
    else:
        try:
            async with session_factory() as session:
                await session.execute(text("SELECT 1"))
        except Exception as exc:  # any failure to answer means "not ready"
            problems.append(f"database: {type(exc).__name__}")
    scheduler = request.app.state.scheduler
    if scheduler is not None and not scheduler.running:
        problems.append("scheduler not running")
    if problems:
        return JSONResponse(status_code=503, content={"status": "unready", "problems": problems})
    return JSONResponse(content={"status": "ok"})
