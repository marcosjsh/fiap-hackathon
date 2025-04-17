import os
from glob import glob
from collections import defaultdict

# 📁 Caminho base onde estão os diretórios das classes
dataset_base = r"C:\Users\marco\OneDrive\Documents\GitHub\fiap-hackathon\content\drive\dataset"

class_names = ['knife', 'scissor', 'cutter']
splits = ['train', 'valid', 'test']

# 🔁 Inicializa contadores
image_counts = {
    split: defaultdict(int) for split in splits
}

# 🔍 Função para contar imagens por categoria e split
def count_images_per_category():
    for class_name in class_names:
        class_dir = os.path.join(dataset_base, class_name)
        if os.path.exists(class_dir):
            for split in splits:
                split_dir = os.path.join(class_dir, split)
                if os.path.exists(split_dir):
                    # Conta todos os arquivos de imagem
                    images = glob(f"{split_dir}/**/*.jpg", recursive=True) + \
                            glob(f"{split_dir}/**/*.jpeg", recursive=True) + \
                            glob(f"{split_dir}/**/*.png", recursive=True)
                    image_counts[split][class_name] = len(images)

# 🧭 Conta as imagens
print("🔍 Contando imagens em cada categoria e split...")
count_images_per_category()

# 📊 Exibe resultados como tabela Markdown
print("\n### 📸 Quantidade de imagens por categoria e split\n")

for split in splits:
    print(f"\n#### 📂 {split.capitalize()}")
    print("| Categoria    | Imagens |")
    print("|--------------|---------|")
    for class_name in class_names:
        count = image_counts[split][class_name]
        print(f"| {class_name:<12} | {count:<7} |")
