import cv2
import requests
import socket
import uuid

API_URL = "http://127.0.0.1:8000"

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def obter_id_maquina():
    nome_pc = socket.gethostname()

    mac=uuid.getnode()
    mac_string= ':'.join(("%012X" % mac)[i:i+2] for i in range(0, 12, 2))

    return f"{nome_pc}_{mac_string}"

def iniciar_vigia():
    machine_id = obter_id_maquina()
    print(f"impressão digital desta maquina: {machine_id}")
    print(f"solicitando acesso na API: {API_URL}...")
    try:
        login_req = requests.post(f"{API_URL}/login-maquina", json={"machine_id": machine_id})
        if login_req.status_code == 401:
            print("Acesso negado. Máquina não autorizada.")
            return False
        login_req.raise_for_status()

        dados = login_req.json()
        token = dados["token"]
        headers = {"Authorization": f"Bearer {token}"}  
        print(f"Acesso Liberado, Máquina reconhecida. Operador: {dados['usuario']}")
        
        requests.post(f"{API_URL}/camera/status", json={"status": "LIGADA"}, headers=headers)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            rostos = face_cascade.detectMultiScale(gray, 1.3, 5)

            #detecção
            status_txt = "monitorando..." if len(rostos) == 1 else "ALERTA!"
            cor = (0, 255, 0) if len(rostos) == 1 else (0, 0, 255)
            cv2.putText(frame, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
            cv2.imshow('VERIFIQ OS - Scanner', frame)
            
            if cv2.waitKey(1) & 0XFF == ord('q') : break

    except Exception as e:
        print(f" falha na integração: {e}")
    finally:
        if 'headers' in locals():
            requests.post(f"{API_URL}/camera/status", json={"status": "DESLIGADA"}, headers=headers)
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
    
if __name__ == "__main__":
    iniciar_vigia()