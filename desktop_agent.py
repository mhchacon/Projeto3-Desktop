from __future__ import annotations

import json
import os
import platform
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from api import (
    acknowledge_device_command,
    fetch_device_commands,
    get_api_token,
    register_device,
    send_device_heartbeat,
    set_api_token,
)


def _default_config_dir() -> Path:
    if os.name == "nt":
        root = os.getenv("APPDATA")
        if root:
            return Path(root) / "VERIFIQ"
    return Path.home() / ".config" / "verifiq"


def _default_lock_path() -> Path:
    return Path(tempfile.gettempdir()) / "verifiq-desktop.lock"


@dataclass
class DesktopAgentState:
    device_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_token: str = ""
    hostname: str = field(default_factory=platform.node)
    machine: str = field(default_factory=platform.machine)
    os_name: str = field(default_factory=platform.platform)
    paired_at: str = ""
    last_heartbeat_at: str = ""
    last_command_id: str = ""

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "DesktopAgentState":
        path = path or (_default_config_dir() / "agent_state.json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                device_id=str(data.get("device_id") or uuid.uuid4()),
                agent_token=str(data.get("agent_token") or ""),
                hostname=str(data.get("hostname") or platform.node()),
                machine=str(data.get("machine") or platform.machine()),
                os_name=str(data.get("os_name") or platform.platform()),
                paired_at=str(data.get("paired_at") or ""),
                last_heartbeat_at=str(data.get("last_heartbeat_at") or ""),
                last_command_id=str(data.get("last_command_id") or ""),
            )
        except FileNotFoundError:
            return cls()
        except Exception:
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        path = path or (_default_config_dir() / "agent_state.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "device_id": self.device_id,
            "agent_token": self.agent_token,
            "hostname": self.hostname,
            "machine": self.machine,
            "os_name": self.os_name,
            "paired_at": self.paired_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "last_command_id": self.last_command_id,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def device_payload(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "hostname": self.hostname,
            "machine": self.machine,
            "os_name": self.os_name,
            "agent_version": os.getenv("VERIFIQ_AGENT_VERSION", "1.0.0"),
        }

    def apply_registration_response(self, payload: dict[str, Any]) -> None:
        token = str(
            payload.get("agent_token")
            or payload.get("device_token")
            or payload.get("token")
            or payload.get("access_token")
            or ""
        ).strip()
        if token:
            self.agent_token = token
            set_api_token(token)

        paired_at = str(payload.get("paired_at") or payload.get("created_at") or "").strip()
        if paired_at:
            self.paired_at = paired_at

    def sync_api_token(self) -> None:
        if self.agent_token:
            set_api_token(self.agent_token)
        else:
            set_api_token(get_api_token())


class SingleInstanceGuard:
    def __init__(self, lock_path: Optional[Path] = None) -> None:
        self.lock_path = lock_path or _default_lock_path()
        self._handle: Any = None
        self._acquired = False

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("a+b")

        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except Exception:
            handle.close()
            return False

        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii", errors="ignore"))
        handle.flush()
        self._handle = handle
        self._acquired = True
        return True

    def release(self) -> None:
        if not self._acquired or self._handle is None:
            return

        try:
            if os.name == "nt":
                import msvcrt

                self._handle.seek(0)
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            self._handle.close()
        except Exception:
            pass
        self._handle = None
        self._acquired = False

    def __enter__(self) -> "SingleInstanceGuard":
        if not self.acquire():
            raise RuntimeError("Já existe uma instância do VERIFIQ em execução.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


CommandHandler = Callable[[str, dict[str, Any]], dict[str, Any] | None]


class AgentSyncService:
    def __init__(
        self,
        state: DesktopAgentState,
        on_command: CommandHandler,
        heartbeat_interval_s: float = 30.0,
        command_interval_s: float = 10.0,
        state_path: Optional[Path] = None,
    ) -> None:
        self.state = state
        self.on_command = on_command
        self.heartbeat_interval_s = heartbeat_interval_s
        self.command_interval_s = command_interval_s
        self.state_path = state_path
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="verifiq-agent-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _save(self) -> None:
        try:
            self.state.save(self.state_path)
        except Exception:
            pass

    def _ensure_registered(self) -> None:
        if self.state.agent_token:
            self.state.sync_api_token()
            return

        response = register_device(self.state.device_payload())
        if isinstance(response, dict) and not response.get("error"):
            self.state.apply_registration_response(response)
            self._save()

    def _heartbeat(self) -> None:
        payload = self.state.device_payload() | {
            "agent_token": self.state.agent_token,
            "last_command_id": self.state.last_command_id,
        }
        response = send_device_heartbeat(payload)
        if isinstance(response, dict) and not response.get("error"):
            heartbeat_at = str(response.get("heartbeat_at") or response.get("server_time") or "").strip()
            if heartbeat_at:
                self.state.last_heartbeat_at = heartbeat_at
            next_token = str(response.get("agent_token") or response.get("device_token") or "").strip()
            if next_token and next_token != self.state.agent_token:
                self.state.agent_token = next_token
                set_api_token(next_token)
            self._save()

    def _poll_commands(self) -> None:
        response = fetch_device_commands(self.state.device_id, self.state.last_command_id or None)
        if not isinstance(response, dict) or response.get("error"):
            return

        commands = response.get("commands") or []
        if not isinstance(commands, list):
            return

        for command in commands:
            if not isinstance(command, dict):
                continue
            command_id = str(command.get("command_id") or command.get("id") or "").strip()
            action = str(command.get("action") or command.get("tipo") or "").strip().lower()
            payload = command.get("payload") or command.get("data") or {}
            if not command_id or not action:
                continue

            result = self.on_command(action, payload if isinstance(payload, dict) else {}) or {}
            acknowledge_device_command(self.state.device_id, command_id, result=result)
            self.state.last_command_id = command_id
            self._save()

    def _run(self) -> None:
        import time

        self._ensure_registered()
        heartbeat_timer = time.monotonic()
        command_timer = time.monotonic()

        while not self._stop.is_set():
            now = time.monotonic()
            if now - heartbeat_timer >= self.heartbeat_interval_s:
                self._heartbeat()
                heartbeat_timer = now
            if now - command_timer >= self.command_interval_s:
                self._poll_commands()
                command_timer = now
            self._stop.wait(1.0)