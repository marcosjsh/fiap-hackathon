#pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# Carregue seu modelo treinado (pode ser um .pt personalizado)
model = YOLO('.\\runs\\detect\\sharped-vs-nonsharped-v13\\weights\\best.pt')  # Ex: "best.pt"

# Abre a webcam (0 é a câmera padrão)
cap = cv2.VideoCapture(0)

# Verifica se a câmera abriu corretamente
if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

while True:
    # Captura um frame da webcam
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o frame.")
        break

    # Usa o modelo YOLO para inferência
    results = model(frame, stream=True)

    # Desenha as caixas de detecção no frame
    for r in results:
        annotated_frame = r.plot()

        # Exibe o frame com anotações
        cv2.imshow("YOLO Webcam", annotated_frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha a janela
cap.release()
cv2.destroyAllWindows()