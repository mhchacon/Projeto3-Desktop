from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import cv2
import face_recognition
import numpy as np


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    encoding: Optional[np.ndarray] = None


@dataclass
class FaceMatch:
    usuario_id: Optional[str]
    nome: str
    score: float


class FaceEngine:
    def __init__(self, det_size: tuple[int, int] = (640, 640)):
        self.det_size = det_size

    def analyze(self, frame: np.ndarray) -> list[FaceDetection]:
        if frame is None or frame.size == 0:
            return []

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, locations)

        faces: list[FaceDetection] = []
        for location, encoding in zip(locations, encodings):
            top, right, bottom, left = location
            faces.append(
                FaceDetection(
                    bbox=(left, top, right, bottom),
                    encoding=np.asarray(encoding, dtype=np.float32),
                )
            )
        return faces

    @staticmethod
    def largest_face(faces: Iterable[FaceDetection]) -> Optional[FaceDetection]:
        faces_list = list(faces)
        if not faces_list:
            return None

        def area(face: FaceDetection) -> int:
            x1, y1, x2, y2 = face.bbox
            return max(0, x2 - x1) * max(0, y2 - y1)

        return max(faces_list, key=area)

    @staticmethod
    def embedding(face: Optional[FaceDetection]) -> Optional[np.ndarray]:
        if face is None:
            return None
        return face.encoding


class FaceGallery:
    def __init__(self):
        self.rows: list[dict] = []
        self.embeddings: list[np.ndarray] = []
        self.names: list[str] = []
        self.user_ids: list[str] = []

    @property
    def size(self) -> int:
        return len(self.embeddings)

    def load_rows(self, rows: list[dict]) -> None:
        self.rows = rows or []
        self.embeddings = []
        self.names = []
        self.user_ids = []

        for row in self.rows:
            embedding = row.get("embedding") or []
            if not embedding:
                continue
            self.embeddings.append(np.asarray(embedding, dtype=np.float32))
            self.names.append(str(row.get("nome") or ""))
            self.user_ids.append(str(row.get("usuario_id") or ""))

    def best_match(self, embedding: np.ndarray) -> FaceMatch:
        if embedding is None or not self.embeddings:
            return FaceMatch(usuario_id=None, nome="", score=0.0)

        distances = face_recognition.face_distance(self.embeddings, embedding)
        melhor_indice = int(np.argmin(distances))
        distancia = float(distances[melhor_indice])
        score = max(0.0, 1.0 - distancia)

        return FaceMatch(
            usuario_id=self.user_ids[melhor_indice],
            nome=self.names[melhor_indice],
            score=score,
        )
