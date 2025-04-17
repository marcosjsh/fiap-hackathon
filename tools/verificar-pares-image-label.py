import os

def verificar_correspondencia_labels(pasta_imagens, pasta_labels):
    imagens_sem_label = []
    labels_sem_imagem = []
    total_imagens = 0
    total_labels = 0

    # Verificar imagens que não têm labels
    for root, _, files in os.walk(pasta_imagens):
        for file in files:
            if file.endswith(".jpg"):
                total_imagens += 1
                nome_base = os.path.splitext(file)[0]
                label_path = os.path.join(pasta_labels, nome_base + ".txt")
                if not os.path.exists(label_path):
                    imagens_sem_label.append(os.path.join(root, file))

    # Verificar labels que não têm imagens
    for root, _, files in os.walk(pasta_labels):
        for file in files:
            if file.endswith(".txt"):
                total_labels += 1
                nome_base = os.path.splitext(file)[0]
                imagem_path = os.path.join(pasta_imagens, nome_base + ".jpg")
                if not os.path.exists(imagem_path):
                    labels_sem_imagem.append(os.path.join(root, file))

    print(f"Total de imagens analisadas: {total_imagens}")
    print(f"Total de labels analisadas: {total_labels}")
    return imagens_sem_label, labels_sem_imagem


# 🔧 Substitua pelos caminhos corretos
pasta_imagens = "/content/drive"
pasta_labels = "/content/drive"

# Rodar a verificação
imgs_sem_lbl, lbls_sem_img = verificar_correspondencia_labels(pasta_imagens, pasta_labels)

# Mostrar os resultados
print("📷 Imagens sem label correspondente:")
for img in imgs_sem_lbl:
    print(" -", img)

print("\n📝 Labels sem imagem correspondente:")
for lbl in lbls_sem_img:
    print(" -", lbl)
