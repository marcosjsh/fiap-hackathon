#pip install ultralytics opencv-python

from ultralytics import YOLO

key = 1
ext = 'mp4'
# Carregar o modelo treinado
model = YOLO('.\\runs\\detect\\sharped-vs-nonsharped-v13\\weights\\best.pt')

results = model(f'.\\teste\\imagem{key}.png', conf=0.4)  # conf = limiar de confiança

# # Exibir resultados
results[0].show()   # Visualização com bounding boxes
results[0].save(f'./teste/output/imagem_detectado{key}.{ext}')


