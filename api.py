from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://projeto3-api.onrender.com").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "").strip()
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT", "20"))


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_TOKEN:
        headers["Authorization"] = f"Bearer {API_TOKEN}"
    return headers


def _json_or_error(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except Exception:
        payload = {"error": response.text}

    if response.ok:
        return payload

    if isinstance(payload, dict):
        payload.setdefault("error", payload.get("detail") or response.text)
    else:
        payload = {"error": response.text}
    return payload


def fetch_galeria_faces() -> List[dict]:
    if not API_TOKEN:
        return []

    try:
        response = requests.get(
            f"{API_URL}/faces/galeria",
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        payload = _json_or_error(response)
        return payload.get("galeria", []) if isinstance(payload, dict) else []
    except Exception as error:
        return [{"error": str(error)}]


def post_alert(
    motivo: str,
    faces_detectadas: int,
    detalhes: Optional[dict] = None,
    imagem_bytes: Optional[bytes] = None,
) -> dict:
    payload = {
        "motivo": motivo,
        "faces_detectadas": faces_detectadas,
        "score": None,
        "usuario_id_reconhecido": None,
    }

    if detalhes:
        payload["score"] = detalhes.get("score")
        payload["usuario_id_reconhecido"] = detalhes.get("usuario_id_reconhecido")

    try:
        response = requests.post(
            f"{API_URL}/seguranca/alerta",
            json=payload,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        return _json_or_error(response)
    except Exception as error:
        return {"error": str(error)}


def post_ponto(
    tipo: str,
    imagem_bytes: Optional[bytes] = None,
    usuario_id_reconhecido: Optional[str] = None,
    score: Optional[float] = None,
) -> dict:
    payload = {
        "tipo": tipo,
        "score_reconhecimento": score,
        "usuario_id_reconhecido": usuario_id_reconhecido,
    }

    try:
        response = requests.post(
            f"{API_URL}/ponto",
            json=payload,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )
        return _json_or_error(response)
    except Exception as error:
        return {"error": str(error)}
