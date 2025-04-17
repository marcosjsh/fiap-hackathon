import os
import shutil
import random

# 🔁 Mapeamento da classe -> índice no dataset unificado e limite por categoria
class_map = {
    "knife": {"index": 0, "limite": 1000},
    "scissor": {"index": 1, "limite": None},
    "scalpel": {"index": 2, "limite": None},
    "axe": {"index": 3, "limite": 1000},
    "saw": {"index": 4, "limite": None},
    "chainsaw": {"index": 5, "limite": None},
    "chisel": {"index": 6, "limite": None},
    "sickle": {"index": 7, "limite": None}
}
labels_com_defeito = []

base_src = "/content/drive"
base_dst = "/content/drive"

# Criar estrutura de diretórios
splits = ["train", "valid", "test"]
for split in splits:
    os.makedirs(os.path.join(base_dst, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(base_dst, split, "labels"), exist_ok=True)

# Processar classes
for class_name, props in class_map.items():
    class_index = props["index"]
    limite_maximo = props["limite"]

    for split in splits:
        images_path = os.path.join(base_src, class_name, split, "images")
        labels_path = os.path.join(base_src, class_name, split, "labels")

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            continue

        todos_os_arquivos = [f for f in os.listdir(labels_path) if os.path.isfile(os.path.join(labels_path, f))]
        # Seleciona até 1000 arquivos aleatoriamente (ou todos, se tiver menos de 1000)
        arquivos_validos = random.sample(todos_os_arquivos, min(1000, len(todos_os_arquivos)))

        for label_filename in arquivos_validos:
            label_src = os.path.join(labels_path, label_filename)
            image_filename = label_filename.replace('.txt', '.jpg')
            image_src = os.path.join(images_path, image_filename)

            if not os.path.exists(image_src):
                print(f"❌ Imagem não encontrada: {image_src}")
                continue


            # Destinos
            dst_label = os.path.join(base_dst, split, "labels", label_filename)
            dst_image = os.path.join(base_dst, split, "images", image_filename)
            
            print(f"-------------- COPIANDO ARQUIVOS --------------")
            print(f"\nOrigem IMG: {image_src}")
            print(f"\nDestino IMG: {dst_image}")
            print(f"\nOrigem LABEL: {label_src}")
            print(f"\nDestino LABEL: {dst_label}")

            # Copiar arquivos
            shutil.copy2(label_src, dst_label)
            shutil.copy2(image_src, dst_image)

            print(f"\nAtualizando índice da classe")
            # Corrigir label
            with open(dst_label, 'r') as f:
                linhas = f.readlines()

            novas_linhas = []
            for linha in linhas:
                partes = linha.strip().split()
                if not partes:
                    labels_com_defeito += dst_label
                    continue
                partes[0] = str(class_index)
                novas_linhas.append(" ".join(partes))

            with open(dst_label, 'w') as f:
                f.write("\n".join(novas_linhas) + "\n")

            print(f"✅ {split.upper()} | Classe: {class_name} | {image_filename}")

print(f"\n ❌ Labels com defeito: {len(labels_com_defeito)}")
print(labels_com_defeito)
print("\n🎉 Processo finalizado com sucesso.")
