from __future__ import annotations

import ctypes
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from plyer import notification

try:
    import winsound
except ImportError:  # pragma: no cover - not available on Linux
    winsound = None

from api import fetch_galeria_faces, post_alert, post_ponto
from desktop_agent import AgentSyncService, DesktopAgentState, SingleInstanceGuard
from face_engine import FaceDetection, FaceEngine, FaceGallery
from storage import load_pending_events, save_alert, save_ponto, save_pending_events, queue_pending_event

load_dotenv()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _lock_workstation_windows() -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.user32.LockWorkStation())
    except Exception:
        return False


def _encode_jpeg(image: Optional[np.ndarray]) -> Optional[bytes]:
    if image is None or image.size == 0:
        return None
    ok, encoded = cv2.imencode(".jpg", image)
    if not ok:
        return None
    return encoded.tobytes()


def _draw_faces(frame: np.ndarray, faces: list[FaceDetection]) -> None:
    for face in faces:
        x1, y1, x2, y2 = [int(value) for value in face.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 2)


def _safe_notify(title: str, message: str) -> None:
    try:
        notification.notify(title=title, message=message)
    except Exception:
        pass


def _safe_sound() -> None:
    try:
        sound = os.getenv("SOUND_ALERT_PATH", "")
        if sound:
            if winsound is not None and os.path.splitext(sound)[1].lower() == ".wav":
                winsound.PlaySound(sound, winsound.SND_FILENAME | winsound.SND_ASYNC)
    except Exception:
        pass


@dataclass
class RecognitionResult:
    frame_index: int
    captured_at: float
    crop: Optional[np.ndarray]
    embedding: Optional[np.ndarray]


class RecognitionWorker:
    def __init__(self, engine: FaceEngine) -> None:
        self._engine = engine
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="verifiq-face")
        self._future: Optional[Future[RecognitionResult]] = None

    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def submit(self, crop: Optional[np.ndarray], frame_index: int, captured_at: float) -> bool:
        if crop is None or crop.size == 0 or self.busy():
            return False
        crop_copy = np.ascontiguousarray(crop)
        self._future = self._executor.submit(self._run_with_engine, crop_copy, frame_index, captured_at)
        return True

    @staticmethod
    def _run(crop: np.ndarray, frame_index: int, captured_at: float) -> RecognitionResult:
        raise RuntimeError("RecognitionWorker._run should not be called directly.")

    def _run_with_engine(self, crop: np.ndarray, frame_index: int, captured_at: float) -> RecognitionResult:
        embedding = self._engine.encode_crop(crop)
        return RecognitionResult(frame_index=frame_index, captured_at=captured_at, crop=crop, embedding=embedding)

    def poll(self) -> Optional[RecognitionResult]:
        if self._future is None or not self._future.done():
            return None

        future = self._future
        self._future = None
        return future.result()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class SecurityState:
    last_faces: list[FaceDetection]
    primary_face: Optional[FaceDetection]
    latest_result: Optional[RecognitionResult]
    latest_match: Any
    status_text: str
    status_color: tuple[int, int, int]
    faces_count: int


def _post_alert_event(
    motivo: str,
    faces_detectadas: int,
    crop: Optional[np.ndarray] = None,
    score: Optional[float] = None,
    usuario_id: Optional[str] = None,
) -> None:
    imagem_bytes = _encode_jpeg(crop)
    detalhes = {"score": score, "usuario_id_reconhecido": usuario_id}
    try:
        resp = post_alert(
            motivo,
            faces_detectadas,
            detalhes=detalhes,
            imagem_bytes=imagem_bytes,
        )
        if resp.get("error"):
            print(f"[API] Falha ao enviar alerta: {resp['error']}")
            queue_pending_event(
                "alert",
                {
                    "motivo": motivo,
                    "faces_detectadas": faces_detectadas,
                    "detalhes": detalhes,
                    "imagem_bytes": imagem_bytes.decode("latin1") if imagem_bytes else None,
                },
            )
    except Exception as error:
        print(f"[API] Erro ao enviar alerta: {error}")
        queue_pending_event(
            "alert",
            {
                "motivo": motivo,
                "faces_detectadas": faces_detectadas,
                "detalhes": detalhes,
                "imagem_bytes": imagem_bytes.decode("latin1") if imagem_bytes else None,
            },
        )
    try:
        save_alert(motivo, faces_detectadas, detalhes=detalhes, imagem_bytes=imagem_bytes)
    except Exception:
        pass
    _safe_sound()


def _post_ponto_event(
    usuario_id: Optional[str],
    score: Optional[float],
    crop: Optional[np.ndarray],
    embedding: Optional[np.ndarray],
) -> None:
    imagem_bytes = _encode_jpeg(crop)
    try:
        resp = post_ponto(
            tipo="entrada",
            imagem_bytes=imagem_bytes,
            usuario_id_reconhecido=usuario_id,
            score=score,
            embedding=None if embedding is None else embedding.tolist(),
        )
        if resp.get("error"):
            print(f"[API] Falha ao enviar ponto: {resp['error']}")
            queue_pending_event(
                "ponto",
                {
                    "tipo": "entrada",
                    "usuario_id_reconhecido": usuario_id,
                    "score": score,
                    "embedding": None if embedding is None else embedding.tolist(),
                    "imagem_bytes": imagem_bytes.decode("latin1") if imagem_bytes else None,
                },
            )
        else:
            print(f"[API] Ponto confirmado: {resp}")
    except Exception as error:
        print(f"[API] Erro ao enviar ponto: {error}")
        queue_pending_event(
            "ponto",
            {
                "tipo": "entrada",
                "usuario_id_reconhecido": usuario_id,
                "score": score,
                "embedding": None if embedding is None else embedding.tolist(),
                "imagem_bytes": imagem_bytes.decode("latin1") if imagem_bytes else None,
            },
        )
    try:
        save_ponto(
            tipo="entrada",
            usuario_id_reconhecido=usuario_id,
            score=score,
            imagem_bytes=imagem_bytes,
        )
    except Exception:
        pass
    _safe_sound()


def _open_camera() -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if os.name == "nt" else 0
    capture = cv2.VideoCapture(0, backend)
    try:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return capture


def _bytes_from_serialized_image(value: Optional[str]) -> Optional[bytes]:
    if not value:
        return None
    return value.encode("latin1")


def _flush_pending_events() -> None:
    pending = load_pending_events()
    if not pending:
        return

    remaining: list[dict[str, Any]] = []
    for event in pending:
        event_type = str(event.get("event_type") or "").strip().lower()
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue

        try:
            if event_type == "alert":
                response = post_alert(
                    str(payload.get("motivo") or ""),
                    int(payload.get("faces_detectadas") or 0),
                    detalhes=payload.get("detalhes") if isinstance(payload.get("detalhes"), dict) else None,
                    imagem_bytes=_bytes_from_serialized_image(payload.get("imagem_bytes")),
                )
            elif event_type == "ponto":
                embedding = payload.get("embedding") or None
                response = post_ponto(
                    tipo=str(payload.get("tipo") or "entrada"),
                    imagem_bytes=_bytes_from_serialized_image(payload.get("imagem_bytes")),
                    usuario_id_reconhecido=payload.get("usuario_id_reconhecido"),
                    score=payload.get("score"),
                    embedding=embedding,
                )
            else:
                remaining.append(event)
                continue

            if response.get("error"):
                remaining.append(event)
        except Exception:
            remaining.append(event)

    save_pending_events(remaining)


def iniciar_vigia() -> None:
    threshold = _float_env("FACE_MATCH_THRESHOLD", 0.38)
    debounce_unknown = _int_env("ALERT_DEBOUNCE_FRAMES_UNKNOWN", 3)
    debounce_multi = _int_env("ALERT_DEBOUNCE_FRAMES_MULTI", 2)
    alert_cooldown_s = _float_env("ALERT_COOLDOWN_SEGUNDOS", 45.0)
    point_cooldown_s = _float_env("PONTO_COOLDOWN_SEGUNDOS", 60.0)
    gallery_refresh_s = _float_env("GALERIA_REFRESH_SEGUNDOS", 120.0)
    detect_stride = max(1, _int_env("FRAME_PROCESS_STRIDE", 4))
    auto_lock_on_intruder = _bool_env("AUTO_LOCK_ON_INTRUSO", True)
    lock_cooldown_s = _float_env("LOCK_COOLDOWN_SEGUNDOS", 120.0)
    det_w = max(320, min(640, _int_env("FACE_DET_SIZE", 480)))
    headless = _bool_env("HEADLESS_MODE", False)

    print("Carregando modelo de reconhecimento local...")
    engine = FaceEngine(det_size=(det_w, det_w))
    galeria = FaceGallery()
    worker = RecognitionWorker(engine)
    agent_state = DesktopAgentState.load()
    agent_state.sync_api_token()
    sync_service: Optional[AgentSyncService] = None

    monitoring_enabled = threading.Event()
    monitoring_enabled.set()
    stop_requested = threading.Event()

    def recarregar_galeria() -> None:
        rows = fetch_galeria_faces()
        galeria.load_rows(rows)
        print(f"[GALERIA] {galeria.size} rosto(s) autorizado(s).")

    def on_agent_command(action: str, payload: dict[str, Any]) -> dict[str, Any]:
        if action in {"start_monitoring", "start", "resume"}:
            monitoring_enabled.set()
            return {"status": "monitoring_enabled"}
        if action in {"stop_monitoring", "stop", "pause"}:
            monitoring_enabled.clear()
            return {"status": "monitoring_disabled"}
        if action in {"refresh_gallery", "reload_gallery", "sync_gallery"}:
            recarregar_galeria()
            return {"status": "gallery_refreshed", "size": galeria.size}
        if action in {"lock", "lock_workstation"}:
            if _lock_workstation_windows():
                return {"status": "locked"}
            return {"status": "lock_unavailable"}
        if action in {"shutdown", "exit", "quit"}:
            stop_requested.set()
            return {"status": "stop_requested"}
        return {"status": "ignored", "action": action, "payload": payload}

    def tentar_bloquear_tela(motivo: str) -> None:
        nonlocal ultimo_bloqueio_mono
        if not auto_lock_on_intruder:
            return
        agora = time.monotonic()
        if agora - ultimo_bloqueio_mono < lock_cooldown_s:
            return
        if _lock_workstation_windows():
            ultimo_bloqueio_mono = agora
            print(f"[SEGURANCA] Tela bloqueada: {motivo}")
        else:
            print("[SEGURANCA] Bloqueio indisponível nesta sessão.")

    try:
        with SingleInstanceGuard():
            recarregar_galeria()
            ultima_galeria = time.monotonic()
            print("Iniciando sistema de segurança...")
            print("  [q] sair  |  [p] bater ponto manual  |  [r] recarregar galeria")

            sync_service = AgentSyncService(agent_state, on_agent_command)
            sync_service.start()

            cap = _open_camera()
            if not cap.isOpened():
                print("Erro: não foi possível abrir a câmera local.")
                return

            frame_index = 0
            streak_unknown = 0
            streak_multi = 0
            ultimo_alerta_mono = 0.0
            ultimo_bloqueio_mono = 0.0
            ultimo_ponto_mono = 0.0
            ultimo_flush_pendente_mono = 0.0
            ultimo_ponto_usuario: Optional[str] = None
            latest_faces: list[FaceDetection] = []
            latest_primary_face: Optional[FaceDetection] = None
            latest_result: Optional[RecognitionResult] = None
            latest_match: Any = None
            latest_status = "Monitorando..."
            latest_color = (200, 200, 200)

            while not stop_requested.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("Erro: falha ao ler frame da câmera.")
                    break

                frame_index += 1
                now = time.monotonic()

                if now - ultima_galeria >= gallery_refresh_s:
                    recarregar_galeria()
                    ultima_galeria = now

                if now - ultimo_flush_pendente_mono >= max(30.0, point_cooldown_s / 2.0):
                    _flush_pending_events()
                    ultimo_flush_pendente_mono = now

                if monitoring_enabled.is_set():
                    if frame_index % detect_stride == 0 or latest_primary_face is None:
                        latest_faces = engine.detect_faces(frame)
                        latest_primary_face = engine.largest_face(latest_faces)
                        if latest_primary_face is not None and latest_primary_face.crop is None:
                            latest_primary_face.crop = engine.crop_face(frame, latest_primary_face.bbox)
                        if len(latest_faces) > 1:
                            streak_multi += 1
                        else:
                            streak_multi = 0
                        if latest_primary_face is not None and len(latest_faces) == 1:
                            worker.submit(latest_primary_face.crop, frame_index, now)
                else:
                    latest_faces = []
                    latest_primary_face = None
                    latest_status = "Monitoramento pausado"
                    latest_color = (200, 200, 80)

                result = worker.poll()
                if result is not None:
                    latest_result = result
                    if result.embedding is None:
                        latest_match = None
                        latest_status = "Rosto sem embedding local"
                        latest_color = (0, 140, 255)
                    else:
                        latest_match = galeria.best_match(result.embedding)
                        if galeria.size == 0:
                            latest_status = "Galeria vazia - cadastre faces na API"
                            latest_color = (0, 165, 255)
                        elif latest_match.score >= threshold:
                            display_name = latest_match.nome or f"ID {latest_match.usuario_id}"
                            latest_status = f"Autorizado: {display_name}"
                            latest_color = (0, 220, 0)
                            streak_unknown = 0
                            if (
                                latest_match.usuario_id
                                and (
                                    ultimo_ponto_usuario != latest_match.usuario_id
                                    or now - ultimo_ponto_mono >= point_cooldown_s
                                )
                            ):
                                print(
                                    f"[PONTO] Registrando entrada para {display_name} "
                                    f"(score={latest_match.score:.3f})"
                                )
                                _post_ponto_event(
                                    usuario_id=latest_match.usuario_id,
                                    score=latest_match.score,
                                    crop=result.crop,
                                    embedding=result.embedding,
                                )
                                ultimo_ponto_usuario = latest_match.usuario_id
                                ultimo_ponto_mono = now
                        else:
                            latest_status = f"Desconhecido (sim={latest_match.score:.2f})"
                            latest_color = (0, 0, 255)
                            streak_unknown += 1
                            if (
                                streak_unknown >= debounce_unknown
                                and now - ultimo_alerta_mono >= alert_cooldown_s
                            ):
                                print(
                                    f"[ALERTA] Rosto desconhecido detectado "
                                    f"(score={latest_match.score:.3f})"
                                )
                                _post_alert_event(
                                    motivo="rosto_desconhecido",
                                    faces_detectadas=1,
                                    crop=result.crop,
                                    score=latest_match.score,
                                    usuario_id=latest_match.usuario_id,
                                )
                                tentar_bloquear_tela("rosto_desconhecido")
                                ultimo_alerta_mono = now
                                streak_unknown = 0

                if monitoring_enabled.is_set() and len(latest_faces) > 1:
                    latest_status = f"ALERTA: {len(latest_faces)} pessoas no quadro"
                    latest_color = (0, 0, 255)
                    streak_unknown = 0
                    if streak_multi >= debounce_multi and now - ultimo_alerta_mono >= alert_cooldown_s:
                        print(f"[ALERTA] Múltiplas pessoas detectadas: {len(latest_faces)}")
                        _post_alert_event(
                            motivo="multiplas_pessoas",
                            faces_detectadas=len(latest_faces),
                        )
                        tentar_bloquear_tela("multiplas_pessoas")
                        ultimo_alerta_mono = now
                        streak_multi = 0
                elif monitoring_enabled.is_set() and len(latest_faces) == 0:
                    latest_status = "Nenhum rosto detectado"
                    latest_color = (180, 180, 180)
                    streak_unknown = 0
                    streak_multi = 0
                elif monitoring_enabled.is_set() and latest_result is None:
                    latest_status = "Analisando rosto localmente..."
                    latest_color = (220, 220, 220)

                if not headless:
                    _draw_faces(frame, latest_faces)
                    cv2.putText(
                        frame,
                        latest_status,
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        latest_color,
                        2,
                    )
                    if latest_match is not None and len(latest_faces) <= 1:
                        cv2.putText(
                            frame,
                            f"limiar={threshold:.2f}  score={latest_match.score:.3f}",
                            (10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (220, 220, 220),
                            1,
                        )

                    cv2.imshow("VERIFIQ OS - Scanner Ativo", frame)
                    key = cv2.waitKey(1) & 0xFF
                else:
                    key = 255
                    time.sleep(0.01)

                if key == ord("q"):
                    stop_requested.set()
                elif key == ord("r"):
                    recarregar_galeria()
                    ultima_galeria = time.monotonic()
                elif key == ord("p"):
                    if latest_primary_face is None:
                        print("Ponto: detecte exatamente um rosto na câmera.")
                        continue
                    if galeria.size == 0:
                        print("Ponto: galeria vazia. Cadastre embeddings no servidor.")
                        continue

                    crop = (
                        latest_primary_face.crop
                        if latest_primary_face.crop is not None
                        else engine.crop_face(frame, latest_primary_face.bbox)
                    )
                    embedding = engine.encode_crop(crop)
                    if embedding is None:
                        print("Ponto: não foi possível obter embedding.")
                        continue

                    match = galeria.best_match(embedding)
                    if match.score < threshold:
                        print(
                            f"Ponto: rosto não reconhecido com confiança "
                            f"(score={match.score:.3f} < {threshold})."
                        )
                        continue

                    display_name = match.nome or f"ID {match.usuario_id}"
                    print(f"Ponto de entrada registrado: {display_name} (score={match.score:.3f})")
                    _post_ponto_event(
                        usuario_id=match.usuario_id,
                        score=match.score,
                        crop=crop,
                        embedding=embedding,
                    )

    except Exception as error:
        print(f"Erro ao iniciar vigia: {error}")
    finally:
        print("Encerrando turno e desligando câmera...")
        if sync_service is not None:
            sync_service.stop()
        worker.shutdown()
        if 'cap' in locals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    iniciar_vigia()
