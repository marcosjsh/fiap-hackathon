# 📌 Importar
from ultralytics import YOLO
import os

# 📌 Definir caminho absoluto do dataset
dataset_path = os.path.abspath('dataset/data.yaml')

# 📌 Carregar modelo YOLOv5s
model = YOLO('yolov5s.yaml') 

def main():
    # 📌 Treinar o modelo
    model.train(
        data=dataset_path,
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        workers=2,
        name='sharped-vs-nonsharped-v1',
        pretrained=True,
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        flipud=0.3, fliplr=0.5,
        degrees=10.0, translate=0.1, scale=0.5, shear=2.0,
        patience=20
    )

if __name__ == '__main__':
    main()

