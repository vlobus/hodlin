"""Liveness probe. Readiness (DB/scheduler checks) arrives with T8/T10."""

from fastapi import APIRouter

router = APIRouter(prefix="/health")


@router.get("/live")
async def live() -> dict[str, str]:
    return {"status": "ok"}
