"""
Lightweight event sink → JSONL at backend/data/logs/events.jsonl
"""
from __future__ import annotations
from pathlib import Path
import json
import time
from typing import Any, Dict

LOG_DIR = Path(__file__).resolve().parents[2] / "backend" / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "events.jsonl"

def record_event(kind: str, payload: Dict[str, Any]) -> None:
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "kind": kind,
        "payload": payload,
    }
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
