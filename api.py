from __future__ import annotations

import base64
import os
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "https://projeto3-api.onrender.com").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "").strip()
API_LOGIN_EMAIL = os.getenv("API_LOGIN_EMAIL", "").strip()
API_LOGIN_PASSWORD = os.getenv("API_LOGIN_PASSWORD", "").strip()
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT", "20"))
_ACTIVE_TOKEN = API_TOKEN


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
    global _ACTIVE_TOKEN

    if _ACTIVE_TOKEN and not force_refresh:
        return _ACTIVE_TOKEN

    refreshed = _login_token()
    if refreshed:
        _ACTIVE_TOKEN = refreshed
    return _ACTIVE_TOKEN


def set_api_token(token: str) -> None:
    global _ACTIVE_TOKEN
    _ACTIVE_TOKEN = token.strip()


def get_api_token() -> str:
    return _ensure_token()


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


def register_device(payload: dict[str, Any]) -> dict:
    try:
        response = _request("POST", "/desktop/devices/register", json=payload)
        return _json_or_error(response)
    except Exception as error:
        return {"error": str(error)}


def send_device_heartbeat(payload: dict[str, Any]) -> dict:
    device_id = str(payload.get("device_id") or "").strip()
    if not device_id:
        return {"error": "device_id ausente"}

    try:
        response = _request("POST", f"/desktop/devices/{device_id}/heartbeat", json=payload)
        return _json_or_error(response)
    except Exception as error:
        return {"error": str(error)}


def fetch_device_commands(device_id: str, last_command_id: Optional[str] = None) -> dict:
    device_id = str(device_id).strip()
    if not device_id:
        return {"error": "device_id ausente", "commands": []}

    path = f"/desktop/devices/{device_id}/commands"
    if last_command_id:
        path += f"?after={last_command_id}"

    try:
        response = _request("GET", path)
        payload = _json_or_error(response)
        if isinstance(payload, dict):
            payload.setdefault("commands", [])
            return payload
        return {"commands": []}
    except Exception as error:
        return {"error": str(error), "commands": []}


def acknowledge_device_command(device_id: str, command_id: str, result: Optional[dict] = None) -> dict:
    device_id = str(device_id).strip()
    command_id = str(command_id).strip()
    if not device_id or not command_id:
        return {"error": "device_id ou command_id ausente"}

    payload = {"command_id": command_id, "result": result or {}}

    try:
        response = _request("POST", f"/desktop/devices/{device_id}/commands/{command_id}/ack", json=payload)
        return _json_or_error(response)
    except Exception as error:
        return {"error": str(error)}


def post_alert(
    motivo: str,
    faces_detectadas: int,
    detalhes: Optional[dict] = None,
    imagem_bytes: Optional[bytes] = None,
    embedding: Optional[Iterable[float]] = None,
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

    if embedding is not None:
        payload["embedding"] = list(embedding)

    if imagem_bytes:
        payload["foto_rosto_base64"] = base64.b64encode(imagem_bytes).decode("ascii")

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
    embedding: Optional[Iterable[float]] = None,
) -> dict:
    payload = {
        "tipo": tipo,
        "score_reconhecimento": score,
        "usuario_id_reconhecido": usuario_id_reconhecido,
    }

    if embedding is not None:
        payload["embedding"] = list(embedding)

    if imagem_bytes:
        payload["foto_rosto_base64"] = base64.b64encode(imagem_bytes).decode("ascii")

    try:
        response = _request("POST", "/ponto", json=payload)
        return _json_or_error(response)
    except Exception as error:
        return {"error": str(error)}
