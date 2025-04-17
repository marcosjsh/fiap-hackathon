import cv2
from pathlib import Path
from albumentations import (
    Compose, HorizontalFlip, VerticalFlip, RandomRotate90,
    CoarseDropout, RandomResizedCrop, Affine, BboxParams
)

N_AUGS = 3

# Caminhos
IMGS_DIR = Path(r"C:\Users\marco\OneDrive\Documents\GitHub\fiap-hackathon\content\drive\dataset\cutter\images")
LABELS_DIR = Path(r"C:\Users\marco\OneDrive\Documents\GitHub\fiap-hackathon\content\drive\dataset\cutter\labels")
AUG_IMG_DIR = Path(IMGS_DIR) / "augmented"
AUG_LABEL_DIR = Path(LABELS_DIR) / "augmented"
AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)
AUG_LABEL_DIR.mkdir(parents=True, exist_ok=True)

transform = Compose([
    # Espelhamentos aumentam variação posicional
    HorizontalFlip(p=0.6),            # Aumentei para 60% — comum em dataset real
    VerticalFlip(p=0.4),              # Aumentei para 40% — objetos podem estar invertidos
    RandomRotate90(p=0.4),            # Rotações fortes melhoram orientação geral

    # Zoom extremo e cortes agressivos
    RandomResizedCrop(
        size=(512, 512),               # Tamanho fixo para manter consistência  
        scale=(0.2, 0.6),             # Zoom forte — até 20% da imagem original visível
        ratio=(0.5, 1.5),             # Permite cortes verticais e horizontais variados
        p=0.6                         # Alta chance, simula objetos muito próximos
    ),

    # Transformação afim agressiva
    Affine(
        translate_percent=0.15,       # Translação até 15% — pega pedaços diferentes
        scale=(0.9, 1.3),             # Permite reduzir e aumentar escala
        rotate=(-30, 30),             # Roda até 30° para ambos os lados
        p=0.6                         # Aumentei para aplicar com frequência
    ),

    # Simulação de obstruções (ex: mãos, sombras, dedos)
    CoarseDropout(
        num_holes_range=(4, 8),                  # Número de buracos (simula múltiplas oclusões)
        hole_height_range=(0.1, 0.3),            # Altura de cada buraco: entre 10% e 30% da imagem
        hole_width_range=(0.1, 0.3),             # Largura de cada buraco: entre 10% e 30%
        fill='random',                           # Preenchimento de cor aleatória para simular oclusões por tecidos e outros objetos
        p=0.6                                    # Aplica em 60% das imagens
    )
],
bbox_params=BboxParams(
    format='yolo',
    label_fields=['class_labels'],
    min_visibility=0.15,              # Exige pelo menos 15% do bbox visível
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


    image = cv2.imread(str(img_file))
    h, w = image.shape[:2]

    for i in range(N_AUGS):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_ids)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_classes = augmented['class_labels']

        if len(aug_bboxes) == 0:
            continue

        new_img_name = img_file.stem + f"_aug{i}.jpg"
        new_lbl_name = label_file.stem + f"_aug{i}.txt"

        cv2.imwrite(str(AUG_IMG_DIR / new_img_name), aug_img)
        salvar_labels(aug_bboxes, aug_classes, AUG_LABEL_DIR / new_lbl_name)


print("✅ Superaugmentações concluídas com sucesso.")
