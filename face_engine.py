from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]
    encoding: Optional[np.ndarray] = None
    crop: Optional[np.ndarray] = None
    score: float = 0.0


@dataclass
class FaceMatch:
    usuario_id: Optional[str]
    nome: str
    score: float


class FaceEngine:
    def __init__(self, det_size: tuple[int, int] = (640, 640)):
        self.det_size = det_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._detector = cv2.CascadeClassifier(cascade_path)
        if self._detector.empty():
            raise RuntimeError("Falha ao carregar o detector de faces do OpenCV.")

        eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
        self._eye_detector = cv2.CascadeClassifier(eye_cascade_path)
        if self._eye_detector.empty():
            raise RuntimeError("Falha ao carregar o detector de olhos do OpenCV.")

        self._hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)

    def detect_faces(self, frame: np.ndarray) -> list[FaceDetection]:
        if frame is None or frame.size == 0:
            return []

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return []

        det_w, det_h = self.det_size
        det_frame = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
        rects = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )

        faces: list[FaceDetection] = []
        scale_x = frame_w / float(det_w)
        scale_y = frame_h / float(det_h)
        for x, y, width, height in rects:
            left = max(0, int(round(x * scale_x)))
            top = max(0, int(round(y * scale_y)))
            right = min(frame_w, int(round((x + width) * scale_x)))
            bottom = min(frame_h, int(round((y + height) * scale_y)))
            crop = self.crop_face(frame, (left, top, right, bottom), padding=0.12)
            faces.append(
                FaceDetection(
                    bbox=(left, top, right, bottom),
                    crop=crop,
                    score=1.0,
                )
            )

        faces.sort(key=lambda face: self._face_priority(frame.shape[:2], face), reverse=True)
        return faces

    def analyze(self, frame: np.ndarray) -> list[FaceDetection]:
        faces = self.detect_faces(frame)
        for face in faces:
            face.encoding = self.encode_crop(face.crop)
        return faces

    @staticmethod
    def crop_face(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        padding: float = 0.12,
    ) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        if width == 0 or height == 0:
            return None

        pad_x = int(round(width * padding))
        pad_y = int(round(height * padding))
        left = max(0, x1 - pad_x)
        top = max(0, y1 - pad_y)
        right = min(frame_w, x2 + pad_x)
        bottom = min(frame_h, y2 + pad_y)
        if right <= left or bottom <= top:
            return None
        return frame[top:bottom, left:right]

    @staticmethod
    def _face_priority(frame_size: tuple[int, int], face: FaceDetection) -> tuple[int, float]:
        frame_h, frame_w = frame_size
        x1, y1, x2, y2 = face.bbox
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        area = width * height
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0
        dist_to_center = abs(center_x - frame_w / 2.0) / max(frame_w, 1) + abs(center_y - frame_h / 2.0) / max(frame_h, 1)
        return area, -dist_to_center

    def encode_crop(self, crop: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None

        aligned = self._align_crop(crop)
        mirrored = cv2.flip(aligned, 1)

        vectors: list[np.ndarray] = []
        for variant in (aligned, mirrored):
            features = self._hog_vector(variant)
            if features is not None:
                vectors.append(features)

        if not vectors:
            return None

        vector = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def _align_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop

        if crop.shape[0] < 64 or crop.shape[1] < 64:
            return cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        eye_region = gray[: max(1, int(gray.shape[0] * 0.7)), :]
        eyes = self._eye_detector.detectMultiScale(
            eye_region,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(18, 18),
        )

        if len(eyes) < 2:
            return cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

        eye_centers = []
        for x, y, w, h in eyes:
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            eye_centers.append((center_x, center_y, w * h))

        eye_centers.sort(key=lambda item: item[2], reverse=True)
        left_eye = min(eye_centers[:2], key=lambda item: item[0])
        right_eye = max(eye_centers[:2], key=lambda item: item[0])

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        if abs(dx) < 1e-6:
            return cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)

        angle = np.degrees(np.arctan2(dy, dx))
        center = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
        rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(
            crop,
            rotation,
            (crop.shape[1], crop.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return cv2.resize(aligned, (128, 128), interpolation=cv2.INTER_AREA)

    def _hog_vector(self, crop: np.ndarray) -> Optional[np.ndarray]:
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        features = self._hog.compute(gray)
        if features is None:
            return None

        return features.reshape(-1).astype(np.float32)

    def close(self) -> None:
        return

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

        query = np.asarray(embedding, dtype=np.float32)
        query = query.reshape(-1)
        query_norm = float(np.linalg.norm(query))
        if query_norm == 0.0:
            return FaceMatch(usuario_id=None, nome="", score=0.0)

        scores = []
        for stored in self.embeddings:
            stored_vec = np.asarray(stored, dtype=np.float32).reshape(-1)
            stored_norm = float(np.linalg.norm(stored_vec))
            if stored_norm == 0.0:
                scores.append(-1.0)
                continue
            cosine = float(np.dot(stored_vec, query) / (stored_norm * query_norm))
            scores.append(cosine)

        melhor_indice = int(np.argmax(scores))
        cosine_score = float(scores[melhor_indice])
        score = max(0.0, min(1.0, (cosine_score + 1.0) / 2.0))

        return FaceMatch(
            usuario_id=self.user_ids[melhor_indice],
            nome=self.names[melhor_indice],
            score=score,
        )
