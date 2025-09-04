"""
FastAPI middleware that logs basic request/response info to events.jsonl
"""
from __future__ import annotations
import time
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from .events import record_event

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t0 = time.perf_counter()
        path = request.url.path
        method = request.method
        try:
            response: Response = await call_next(request)
            return response
        finally:
            dt = (time.perf_counter() - t0) * 1000.0
            record_event("http", {"path": path, "method": method, "ms": round(dt, 2)})
