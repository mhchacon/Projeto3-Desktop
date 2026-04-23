import cv2
import requests
import os
import getpass
from dotenv import load_dotenv

load_dotenv()   

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

face_Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def fazer_login():
    print("VERIFIQ OS - PORTAL DO FUNCIONÁRIO")

    email_digitado = input(" digite seu e-mail: ")
    senha_digitada = getpass.getpass(" digite sua senha: ")

    print("Verificando credenciais...")

    try:
        login_req = requests.post(f"{API_URL}/login", json={"email": email_digitado, "senha": senha_digitada})

        if login_req.status_code == 401:
            print("Credenciais inválidas. Tente novamente.")
            return None
        
        login_req.raise_for_status()

        dados = login_req.json()
        print(f"Bem-vindo, {dados['usuario']}!")  

        return dados["token"]
    
    except Exception as e:
        print(f"Erro ao fazer login: {e}")
        return None
    
def iniciar_vigia(token):

    headers = {"Authorization": f"Bearer {token}"} 

    print("Iniciando sistema de segurança...")

    try:
        requests.post(f"{API_URL}/cameras/status", json={"status": "LIGADA"}, headers=headers)
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret: break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            rostos = face_Cascade.detectMultiScale(gray, 1.3, 5)  

            status_txt = "Ponto Ativo / Monitorando..." if len(rostos) == 1 else "ALERTA DE SEGURANCA!"
            cor = (0, 255, 0) if len(rostos) == 1 else (0, 0, 255)

            cv2.putText(frame, status_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2)
            cv2.imshow('VERIFIQ OS - Scanner Ativo', frame)

            if cv2.waitKey(1) & 0XFF == ord('q'): break

    except Exception as e:
        print(f"Erro ao iniciar vigia: {e}")
        
    finally:
        print("Encerrando turno e desligando câmera...")
        requests.post(f"{API_URL}/cameras/status", json={"status": "DESLIGADA"}, headers=headers)  
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    
    token_recebido = fazer_login()
    
    if token_recebido:
        iniciar_vigia(token_recebido)