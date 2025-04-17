# 📌 Importar
from ultralytics import YOLO
import os

# 📌 Definir caminho absoluto do dataset
dataset_path = os.path.abspath('dataset/data.yaml')

# 📌 Carregar modelo YOLOv5s
model = YOLO('yolov5s.yaml')  # ou 'yolov8s.pt' para transfer learning

# 📌 Treinar o modelo
model.train(
    data=dataset_path,  # Caminho absoluto para o arquivo YAML
    epochs=100,
    imgsz=640,
    batch=32,
    device='cpu',
    workers=2,
    name='sharped-vs-nonsharped-v1',
    pretrained=True,  # usa pesos pré-treinados
    augment=True,     # ativa augmentations básicas
    mosaic=1.0,       # intensidade do Mosaic
    mixup=0.2,        # intensidade do Mixup
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,  # variações de cor
    flipud=0.3, fliplr=0.5,  # flips verticais e horizontais
    degrees=10.0, translate=0.1, scale=0.5, shear=2.0,  # variações geométricas
    patience=20  # early stopping
)
