import os
import shutil
import random
from pathlib import Path

# Classes e quantidades a mover
CLASSES_A_REBALANCEAR = {
    4: {"nome": "saw", "valid": 130, "test": 65},
    6: {"nome": "chisel", "valid": 97, "test": 48},
    7: {"nome": "sickle", "valid": 108, "test": 54}
}

# Diretórios base
BASE_PATH = Path("/content/drive")
LABELS_TRAIN = BASE_PATH / "train" / "labels"
IMAGES_TRAIN = BASE_PATH / "train" / "images"

def get_all_labels_for_class(class_id, label_dir):
    matching = []
    for label_file in label_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith(f"{class_id} "):
                    matching.append(label_file)
                    break
    return matching

# Processar cada classe
for class_id, meta in CLASSES_A_REBALANCEAR.items():
    nome_classe = meta["nome"]
    total_valid = meta["valid"]
    total_test = meta["test"]

    # Coletar e embaralhar arquivos que contêm a classe
    arquivos_classe = get_all_labels_for_class(class_id, LABELS_TRAIN)
    random.shuffle(arquivos_classe)

    mover_valid = arquivos_classe[:total_valid]
    mover_test = arquivos_classe[total_valid:total_valid + total_test]

    for destino, arquivos in [("valid", mover_valid), ("test", mover_test)]:
        for label_file in arquivos:
            image_file = IMAGES_TRAIN / label_file.with_suffix(".jpg").name

            dst_label = BASE_PATH / destino / "labels" / label_file.name
            dst_image = BASE_PATH / destino / "images" / image_file.name

            dst_label.parent.mkdir(parents=True, exist_ok=True)
            dst_image.parent.mkdir(parents=True, exist_ok=True)

            shutil.move(label_file, dst_label)
            shutil.move(image_file, dst_image)

        print(f"✅ {len(arquivos)} arquivos de '{nome_classe}' movidos para {destino}/")

print("\n🎉 Rebalanceamento finalizado com sucesso.")
