import cv2

from api import _request
from face_engine import FaceEngine


def capturar_e_cadastrar():
    print("Iniciando câmera para cadastro. Olhe para a lente e aperte 'c' para capturar.")
    cap = cv2.VideoCapture(0)
    engine = FaceEngine()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = engine.detect_faces(frame)

            # Desenha um quadrado para guiar o usuário
            for face in faces:
                x1, y1, x2, y2 = [int(value) for value in face.bbox]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow("Cadastro de Rosto", frame)

            if cv2.waitKey(1) & 0xFF == ord("c"):
                if len(faces) == 1:
                    print("Rosto detectado! Extraindo embedding local...")
                    face = engine.largest_face(faces)
                    embedding = engine.encode_crop(face.crop if face else None)

                    if embedding is None:
                        print(" Não foi possível extrair o embedding do rosto.")
                        break

                    
                    payload = {"embedding": embedding.tolist()}
                    print("Enviando para a API...")

                    resp = _request("POST", "/rosto/cadastrar", json=payload)
                    if resp.ok:
                        print(" Rosto cadastrado com sucesso no banco de dados!")
                    else:
                        print(f" Erro na API: {resp.text}")
                    break
                else:
                    print("Atenção: Garanta que apenas UM rosto esteja na câmera.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capturar_e_cadastrar()