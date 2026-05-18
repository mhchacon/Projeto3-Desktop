from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def _append_jsonl(filename: str, payload: dict[str, Any]) -> None:
    path = LOG_DIR / filename
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


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
