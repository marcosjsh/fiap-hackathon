import cv2
import os
from ultralytics import YOLO

# ===== CONFIGURAÇÕES =====
key = 1
modelo_path = "runs/detect/sharped-vs-nonsharped-v13/weights/best.pt"  # caminho para o modelo .pt
video_input_path = f"teste/video{key}.mp4"  # caminho do vídeo de entrada
output_dir = "teste/output"          # pasta onde o vídeo será salvo
output_video_name = f"video{key}_processado.mp4"

# Cria a pasta de saída se não existir
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_video_name)

# ===== Carrega o modelo =====
model = YOLO(modelo_path)

# ===== Abre o vídeo de entrada =====
cap = cv2.VideoCapture(video_input_path)
if not cap.isOpened():
    print(f"Erro ao abrir vídeo: {video_input_path}")
    exit()

# Pega as propriedades do vídeo
fps = cap.get(cv2.CAP_PROP_FPS)
largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
codec = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
out = cv2.VideoWriter(output_path, codec, fps, (largura, altura))

# ===== Processa o vídeo frame a frame =====
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecção com o YOLO
    results = model(frame)

    # Pega o frame anotado com bounding boxes
    annotated_frame = results[0].plot()

    # Escreve o frame no novo vídeo
    out.write(annotated_frame)

    frame_count += 1
    print(f"Frame {frame_count} processado")

# ===== Libera recursos =====
cap.release()
out.release()
print(f"Vídeo salvo em: {output_path}")
