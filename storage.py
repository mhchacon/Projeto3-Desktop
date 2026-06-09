from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
PENDING_EVENTS_FILE = LOG_DIR / "pending_events.jsonl"


def _append_jsonl(filename: str, payload: dict[str, Any]) -> None:
    path = LOG_DIR / filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                items.append(payload)
    return items

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def save_alert(
    motivo: str,
    faces_detectadas: int,
    detalhes: Optional[dict] = None,
    imagem_bytes: Optional[bytes] = None,
) -> None:
    _append_jsonl(
        "alerts.jsonl",
        {
            "tipo": motivo,
            "faces_detectadas": faces_detectadas,
            "detalhes": detalhes or {},
            "imagem_bytes": len(imagem_bytes or b""),
            "criado_em": datetime.utcnow().isoformat(),
        },
    )


def save_ponto(
    tipo: str,
    usuario_id_reconhecido: Optional[str] = None,
    score: Optional[float] = None,
    imagem_bytes: Optional[bytes] = None,
) -> None:
    _append_jsonl(
        "pontos.jsonl",
        {
            "tipo": tipo,
            "usuario_id_reconhecido": usuario_id_reconhecido,
            "score": score,
            "imagem_bytes": len(imagem_bytes or b""),
            "criado_em": datetime.utcnow().isoformat(),
        },
    )

def queue_pending_event(event_type: str, payload: dict[str, Any]) -> None:
    _append_jsonl(
        "pending_events.jsonl",
        {
            "event_type": event_type,
            "payload": payload,
            "queued_at": datetime.utcnow().isoformat(),
        },
    )

def load_pending_events() -> list[dict[str, Any]]:
    return _read_jsonl(PENDING_EVENTS_FILE)

def save_pending_events(events: list[dict[str, Any]]) -> None:
    PENDING_EVENTS_FILE.parent.mkdir(exist_ok=True)
    _write_jsonl(PENDING_EVENTS_FILE, events)
