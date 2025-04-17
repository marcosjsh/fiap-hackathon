import os
from glob import glob
from collections import defaultdict

# 📁 Caminho base onde estão os diretórios train/valid/test
dataset_base = "/content/drive"

class_names = ['knife', 'scissor', 'scalpel', 'axe', 'saw', 'chainsaw', 'chisel', 'sickle']

# 🔁 Inicializa contadores por split
image_counts_by_split = {
    "train": defaultdict(set),
    "valid": defaultdict(set),
    "test": defaultdict(set)
}

# 🔍 Função para processar cada split
def count_images_per_category(split_name):
    label_dir = os.path.join(dataset_base, split_name, "labels")
    for label_file in glob(f"{label_dir}/**/*.txt", recursive=True):
        with open(label_file, 'r') as f:
            content = f.read()
            categories_in_image = set()
            for line in content.strip().splitlines():
                parts = line.strip().split()
                if parts and parts[0].isdigit():
                    class_id = int(parts[0])
                    if 0 <= class_id < len(class_names):
                        categories_in_image.add(class_names[class_id])
            for category in categories_in_image:
                image_counts_by_split[split_name][category].add(label_file)

# 🧭 Percorre os splits
for split_key in ['train', 'valid', 'test']:
    print(f"🔍 Contando imagens em: {split_key}")
    count_images_per_category(split_key)

# 📊 Exibe resultados como tabela Markdown por split
print("\n### 📸 Quantidade de imagens por categoria e por split\n")

for split_key in ['train', 'valid', 'test']:
    print(f"\n#### 📂 {split_key.capitalize()}")
    print("| Categoria    | Imagens únicas |")
    print("|--------------|----------------|")
    for class_name in class_names:
        count = len(image_counts_by_split[split_key][class_name])
        print(f"| {class_name:<12} | {count:<14} |")
