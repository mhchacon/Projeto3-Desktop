import os
import time
from typing import Any, Dict, Optional
import ctypes

import cv2
import numpy as np
from dotenv import load_dotenv

from face_engine import FaceEngine, FaceGallery
import base64
import platform

from api import fetch_galeria_faces, post_alert, post_ponto
from storage import save_alert, save_ponto
from plyer import notification
from playsound import playsound

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





def _draw_faces(frame: np.ndarray, faces: list) -> None:
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 2)


def iniciar_vigia() -> None:

    threshold = _float_env("FACE_MATCH_THRESHOLD", 0.38)
    debounce_unknown = _int_env("ALERT_DEBOUNCE_FRAMES_UNKNOWN", 15)
    debounce_multi = _int_env("ALERT_DEBOUNCE_FRAMES_MULTI", 8)
    alert_cooldown_s = _float_env("ALERT_COOLDOWN_SEGUNDOS", 45.0)
    gallery_refresh_s = _float_env("GALERIA_REFRESH_SEGUNDOS", 120.0)
    process_stride = max(1, _int_env("FRAME_PROCESS_STRIDE", 1))
    auto_lock_on_intruder = _bool_env("AUTO_LOCK_ON_INTRUSO", True)
    lock_cooldown_s = _float_env("LOCK_COOLDOWN_SEGUNDOS", 120.0)

    print("Carregando modelo de reconhecimento (primeira execução pode baixar pesos)...")
    det_w = max(320, min(640, _int_env("FACE_DET_SIZE", 640)))
    cap: Optional[cv2.VideoCapture] = None
    engine = FaceEngine(det_size=(det_w, det_w))
    galeria = FaceGallery()

    def recarregar_galeria() -> None:
        rows = fetch_galeria_faces()
        galeria.load_rows(rows)
        print(f"Galeria: {galeria.size} rosto(s) autorizado(s).")

    recarregar_galeria()
    ultima_galeria = time.monotonic()

    print("Iniciando sistema de segurança...")
    print("  [q] sair  |  [p] bater ponto  |  [r] recarregar galeria")
    # TODO: Notificar API que câmera está ligada
    # api.post_camera_status("LIGADA")

    cap = cv2.VideoCapture(0)
    frame_index = 0
    streak_unknown = 0
    streak_multi = 0
    ultimo_alerta_mono = 0.0
    ultimo_bloqueio_mono = 0.0
    last_faces: list = []

    def tentar_bloquear_tela(motivo: str) -> None:
        nonlocal ultimo_bloqueio_mono
        if not auto_lock_on_intruder:
            return

        agora = time.monotonic()
        if agora - ultimo_bloqueio_mono < lock_cooldown_s:
            return

        if _lock_workstation_windows():
            ultimo_bloqueio_mono = agora
            print(f"[SEGURANCA] Tela bloqueada pelo motivo: {motivo}")
        else:
            print("[SEGURANCA] Falha ao bloquear a tela (somente Windows com sessão interativa).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            now = time.monotonic()

            if now - ultima_galeria >= gallery_refresh_s:
                recarregar_galeria()
                ultima_galeria = now

            if frame_index % process_stride == 0:
                last_faces = engine.analyze(frame)
            faces = last_faces

            _draw_faces(frame, faces)
            n = len(faces)

            status_txt = "Monitorando..."
            cor = (200, 200, 200)
            match_nome = ""
            match_score = 0.0
            match_uid: Any = None

            if n == 0:
                streak_unknown = 0
                streak_multi = 0
                status_txt = "Nenhum rosto detectado"
                cor = (180, 180, 180)

            elif n > 1:
                streak_unknown = 0
                streak_multi += 1
                status_txt = f"ALERTA: {n} pessoas no quadro"
                cor = (0, 0, 255)
                if (
                    streak_multi >= debounce_multi
                    and now - ultimo_alerta_mono >= alert_cooldown_s
                ):
                    print(f"[ALERTA] Múltiplas pessoas detectadas: {n}")
                    # enviar alerta à API com um frame
                    try:
                        _, jpg = cv2.imencode('.jpg', frame)
                        resp = post_alert('multiplas_pessoas', n, detalhes={"score": None}, imagem_bytes=jpg.tobytes())
                        if resp.get('error'):
                            print('Aviso: falha ao enviar alerta:', resp['error'])
                        # salvar no MongoDB Atlas (assíncrono)
                        try:
                            save_alert('multiplas_pessoas', n, detalhes={"score": None}, imagem_bytes=jpg.tobytes())
                        except Exception:
                            pass
                        # notificar sistema
                        try:
                            notification.notify(title='Alerta de Segurança', message=f'Múltiplas pessoas detectadas: {n}')
                        except Exception:
                            pass
                        # tocar som (opcional, definir SOUND_ALERT_PATH env)
                        try:
                            sound = os.getenv('SOUND_ALERT_PATH', '')
                            if sound:
                                playsound(sound)
                        except Exception:
                            pass
                    except Exception as e:
                        print('Erro ao enviar alerta:', e)
                    tentar_bloquear_tela("multiplas_pessoas")
                    ultimo_alerta_mono = now
                    streak_multi = 0

            else:
                streak_multi = 0
                face = engine.largest_face(faces)
                emb = FaceEngine.embedding(face) if face is not None else None
                if emb is None:
                    status_txt = "Rosto sem embedding"
                    cor = (0, 140, 255)
                else:
                    m = galeria.best_match(emb)
                    match_score = m.score
                    match_nome = m.nome
                    match_uid = m.usuario_id

                    if galeria.size == 0:
                        status_txt = "Galeria vazia — cadastre faces na API"
                        cor = (0, 165, 255)
                    elif match_score >= threshold:
                        status_txt = f"Autorizado: {match_nome or 'ID ' + str(match_uid)}"
                        cor = (0, 220, 0)
                        streak_unknown = 0
                    else:
                        status_txt = f"Desconhecido (sim={match_score:.2f})"
                        cor = (0, 0, 255)
                        streak_unknown += 1
                        if (
                            streak_unknown >= debounce_unknown
                            and now - ultimo_alerta_mono >= alert_cooldown_s
                        ):
                            print(f"[ALERTA] Rosto desconhecido detectado (score={match_score:.3f})")
                            try:
                                _, jpg = cv2.imencode('.jpg', frame)
                                resp = post_alert('rosto_desconhecido', 1, detalhes={"score": match_score}, imagem_bytes=jpg.tobytes())
                                if resp.get('error'):
                                    print('Aviso: falha ao enviar alerta:', resp['error'])
                                try:
                                    save_alert('rosto_desconhecido', 1, detalhes={"score": match_score}, imagem_bytes=jpg.tobytes())
                                except Exception:
                                    pass
                                try:
                                    notification.notify(title='Alerta de Segurança', message=f'Rosto desconhecido (score={match_score:.2f})')
                                except Exception:
                                    pass
                                try:
                                    sound = os.getenv('SOUND_ALERT_PATH', '')
                                    if sound:
                                        playsound(sound)
                                except Exception:
                                    pass
                            except Exception as e:
                                print('Erro ao enviar alerta:', e)
                            tentar_bloquear_tela("rosto_desconhecido")
                            ultimo_alerta_mono = now
                            streak_unknown = 0

            y0 = 28
            cv2.putText(
                frame,
                status_txt,
                (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                cor,
                2,
            )
            if n == 1:
                cv2.putText(
                    frame,
                    f"limiar={threshold:.2f}  score={match_score:.3f}",
                    (10, y0 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (220, 220, 220),
                    1,
                )

            cv2.imshow("VERIFIQ OS - Scanner Ativo", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            if key == ord("r"):
                recarregar_galeria()
                ultima_galeria = time.monotonic()
            if key == ord("p"):
                if n != 1:
                    print("Ponto: detecte exatamente um rosto na câmera.")
                else:
                    # ensure gallery loaded; try reload once
                    if galeria.size == 0:
                        print("Galeria vazia. Tentando recarregar...")
                        recarregar_galeria()
                        # wait briefly for gallery to load
                        t0 = time.monotonic()
                        while galeria.size == 0 and time.monotonic() - t0 < 3.0:
                            time.sleep(0.2)
                    if galeria.size == 0:
                        print("Ponto: galeria vazia. Cadastre embeddings no servidor.")
                        continue

                    face = engine.largest_face(faces)
                    emb = FaceEngine.embedding(face) if face else None
                    if emb is None:
                        print("Ponto: não foi possível obter embedding.")
                    else:
                        m = galeria.best_match(emb)
                        if m.score < threshold:
                            print(
                                f"Ponto: rosto não reconhecido com confiança "
                                f"(score={m.score:.3f} < {threshold})."
                            )
                        else:
                            # Enviar ponto para API
                            print(f"Ponto de entrada registrado: {m.nome} (score={m.score:.3f})")
                            try:
                                # enviar frame atual ao servidor como ponto
                                _, jpg = cv2.imencode('.jpg', frame)
                                resp = post_ponto(
                                    tipo="entrada",
                                    imagem_bytes=jpg.tobytes(),
                                    usuario_id_reconhecido=m.usuario_id,
                                    score=m.score,
                                )
                                if resp.get('error'):
                                    print('Aviso: falha ao enviar ponto:', resp['error'])
                                else:
                                    print('Resposta do servidor:', resp)
                                try:
                                    save_ponto(tipo='entrada', usuario_id_reconhecido=m.usuario_id, score=m.score, imagem_bytes=jpg.tobytes())
                                except Exception:
                                    pass
                            except Exception as e:
                                print('Erro ao enviar ponto:', e)

    except Exception as e:
        print(f"Erro ao iniciar vigia: {e}")

    finally:
        print("Encerrando turno e desligando câmera...")
        # TODO: Notificar API que câmera está desligada
        # try:
        #     api.post_camera_status("DESLIGADA")
        # except Exception:
        #     pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # TODO: Implementar autenticação via API
    iniciar_vigia()
