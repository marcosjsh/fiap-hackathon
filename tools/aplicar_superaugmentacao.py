import os
import cv2
from pathlib import Path
from albumentations import Compose, HorizontalFlip, RandomBrightnessContrast, MotionBlur, Affine, CoarseDropout, BboxParams
from collections import defaultdict

# Classes minoritárias alvo (índices)
CLASSES_ALVO = [1, 2, 4, 5, 6, 7]
MAX_IMGS_POR_CATEGORIA = 1000
N_AUGS = 3

# Contador de imagens por categoria
contador_imgs = defaultdict(int)

# Caminhos
IMGS_DIR = Path("/content/drive")
LABELS_DIR = Path("/content/drive")

# Augmentações definidas
transform = Compose([
    HorizontalFlip(p=0.5),
    RandomBrightnessContrast(p=0.5),
    MotionBlur(blur_limit=3, p=0.2),
    Affine(translate_percent=0.05, scale=1.1, rotate=15, p=0.5),
    CoarseDropout(
        num_holes_range=(1, 4),
        hole_height_range=(0.05, 0.2),
        hole_width_range=(0.05, 0.2),
        fill=0,
        p=0.3)
],
bbox_params=BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.1,
    clip=True,
    filter_invalid_bboxes=True
))

def carrega_labels(label_path):
    with open(label_path, 'r') as f:
        linhas = f.readlines()
    boxes = []
    class_ids = []
    for linha in linhas:
        linha = linha.strip()
        if not linha:
            continue
        partes = linha.split()
        if len(partes) < 5:
            continue
        try:
            cid = int(partes[0])
            bbox = list(map(float, partes[1:5]))
            boxes.append(bbox)
            class_ids.append(cid)
        except ValueError:
            continue
    return boxes, class_ids

def salvar_labels(bboxes, class_ids, out_path):
    with open(out_path, 'w') as f:
        for cls, box in zip(class_ids, bboxes):
            if box[2] <= 0 or box[3] <= 0:
                continue
            f.write(f"{int(cls)} {' '.join(f'{x:.6f}' for x in box)}\n")

# Processar imagens
for img_file in IMGS_DIR.glob("*.jpg"):
    label_file = LABELS_DIR / img_file.with_suffix('.txt').name
    if not label_file.exists():
        continue

    bboxes, class_ids = carrega_labels(label_file)

    # Só considera se houver classe de interesse
    if not any(cls in CLASSES_ALVO for cls in class_ids):
        continue

    # Pula se todas as classes da imagem já bateram o limite
    if all(contador_imgs[cls] >= MAX_IMGS_POR_CATEGORIA for cls in class_ids if cls in CLASSES_ALVO):
        continue

    image = cv2.imread(str(img_file))
    h, w = image.shape[:2]

    for i in range(N_AUGS):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_ids)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_classes = augmented['class_labels']

        if len(aug_bboxes) == 0:
            continue

        # Verifica se pode salvar (pelo menos uma classe da imagem está abaixo do limite)
        if all(contador_imgs[cls] >= MAX_IMGS_POR_CATEGORIA for cls in aug_classes if cls in CLASSES_ALVO):
            continue

        new_img_name = img_file.stem + f"_aug{i}.jpg"
        new_lbl_name = label_file.stem + f"_aug{i}.txt"

        cv2.imwrite(str(IMGS_DIR / new_img_name), aug_img)
        salvar_labels(aug_bboxes, aug_classes, LABELS_DIR / new_lbl_name)

        # Atualiza contadores apenas das classes realmente salvas
        for cls in set(aug_classes):
            if cls in CLASSES_ALVO:
                contador_imgs[cls] += 1

print("✅ Superaugmentações concluídas com sucesso.")
