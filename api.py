from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://projeto3-api.onrender.com").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "").strip()
API_LOGIN_EMAIL = os.getenv("API_LOGIN_EMAIL", "").strip()
API_LOGIN_PASSWORD = os.getenv("API_LOGIN_PASSWORD", "").strip()
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT", "20"))


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    token = _ensure_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _login_token() -> str:
    if not API_LOGIN_EMAIL or not API_LOGIN_PASSWORD:
        return ""

    try:
        response = requests.post(
            f"{API_URL}/login",
            json={"email": API_LOGIN_EMAIL, "senha": API_LOGIN_PASSWORD},
            timeout=REQUEST_TIMEOUT,
        )
        payload = _json_or_error(response)
        if response.ok and isinstance(payload, dict):
            return str(payload.get("token") or "").strip()
    except Exception:
        return ""

    return ""


def _ensure_token(force_refresh: bool = False) -> str:
    global API_TOKEN

    if API_TOKEN and not force_refresh:
        return API_TOKEN

    refreshed = _login_token()
    if refreshed:
        API_TOKEN = refreshed
    return API_TOKEN


def _request(method: str, path: str, *, json: Optional[dict] = None) -> requests.Response:
    response = requests.request(
        method,
        f"{API_URL}{path}",
        json=json,
        headers=_headers(),
        timeout=REQUEST_TIMEOUT,
    )

    if response.status_code == 401 and API_LOGIN_EMAIL and API_LOGIN_PASSWORD:
        _ensure_token(force_refresh=True)
        response = requests.request(
            method,
            f"{API_URL}{path}",
            json=json,
            headers=_headers(),
            timeout=REQUEST_TIMEOUT,
        )

    return response


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
    try:
        response = _request("GET", "/faces/galeria")
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
        response = _request("POST", "/seguranca/alerta", json=payload)
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
        response = _request("POST", "/ponto", json=payload)
        return _json_or_error(response)
    except Exception as error:
        return {"error": str(error)}
