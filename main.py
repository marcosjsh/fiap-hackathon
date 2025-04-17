import cv2
import torch
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import shutil

# --- CONFIGURAÇÕES INICIAIS ---

# Carrega o modelo YOLOv5
modelo = torch.hub.load('ultralytics/yolov5', 'custom', path='./best.pt')
modelo.conf = 0.5  # Confiabilidade mínima
modelo.iou = 0.5   # IOU para NMS
# modelo.max_det = 10   # Número máximo de detecções por imagem

# Configuração do e-mail
email_user = "j1C6o@example.com"  # Endereço do remetente
# Para criar seu código de app, crie em https://myaccount.google.com/apppassword
email_cod = ""   # Código do app do Gmail

# Variáveis globais
opcao = ""

# --- FUNÇÕES PRINCIPAIS ---

def selec_arquivo():
    global opcao
    caminho = filedialog.askopenfilename()
    if caminho:
        opcao = caminho

def limpar_pasta_frames(folder='frames_detectados'):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def detectar(path_arquivo):
    if not path_arquivo or not os.path.exists(path_arquivo):
        print("Arquivo inválido ou inexistente.")
        return False

    limpar_pasta_frames()
    _, ext = os.path.splitext(path_arquivo.lower())
    
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        return detectar_imagem(path_arquivo)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        return detectar_video(path_arquivo)
    else:
        print("Formato de arquivo não suportado.")
        return False

def detectar_imagem(path_img, output_folder='frames_detectados'):
    os.makedirs(output_folder, exist_ok=True)
    img = cv2.imread(path_img)
    results = modelo(img)
    resultado = results.render()[0]

    if len(results.xywh[0]) > 0:
        filename = os.path.basename(path_img)
        cv2.imwrite(os.path.join(output_folder, f"detect_{filename}"), resultado)
        print("Imagem salva com detecção.")
        return True
    else:
        print("Nenhuma detecção.")
        return False

def detectar_video(path_video, output_folder='frames_detectados'):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(path_video)
    houve_deteccao = False
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = modelo(img_rgb)
        resultado = results.render()[0]

        if len(results.xywh[0]) > 0:
            houve_deteccao = True
            frame_count += 1
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.jpg"), resultado)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return houve_deteccao

def enviar_email(email_send, folder_path='frames_detectados'):
    msg = MIMEMultipart()
    msg["From"] = email_user
    msg["To"] = email_send
    msg["Subject"] = "Alerta de Detecção de Objeto"
    msg.attach(MIMEText("Objetos suspeitos foram detectados. Veja os anexos.", "plain"))

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with open(filepath, "rb") as f:
                msg.attach(MIMEImage(f.read(), name=filename))

    try:
        loading_label.config(text="Enviando email...")
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(email_user, email_cod)
        server.sendmail(email_user, email_send, msg.as_string())
        server.quit()
        loading_label.config(text="E-mail enviado com sucesso!")
    except Exception as e:
        loading_label.config(text=f"Erro ao enviar: {e}")

# --- FUNÇÕES DE THREADS / INTERFACE ---

def thread_verificacao():
    global opcao
    email_destino = email_entry.get()
    if not email_destino:
        messagebox.showwarning("Atenção", "Por favor, insira um e-mail.")
        return
    if not opcao:
        messagebox.showwarning("Atenção", "Por favor, selecione um arquivo.")
        return

    progress_bar.start()
    loading_label.config(text="Detectando objetos...")
    select_button.config(state="disabled")
    start_button.config(state="disabled")

    def processo_completo():
        try:
            if detectar(opcao):
                enviar_email(email_destino)
            else:
                loading_label.config(text="Nenhuma detecção encontrada.")
        finally:
            select_button.config(state="normal")
            start_button.config(state="normal")
            progress_bar.stop()

    threading.Thread(target=processo_completo).start()

# --- INTERFACE TKINTER ---

root = tk.Tk()
root.title("IA para DEVs - FIAP/ALURA")
root.configure(bg="#f4f4f4")

largura_janela = 500
altura_janela = 300

def center(root):
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (largura_janela // 2)
    y = (root.winfo_screenheight() // 2) - (altura_janela // 2)
    root.geometry(f"{largura_janela}x{altura_janela}+{x}+{y}")

center(root)

label_font = ("Segoe UI", 10)
entry_font = ("Segoe UI", 10)
button_font = ("Segoe UI", 10, "bold")

main_frame = tk.Frame(root, bg="#f4f4f4", padx=20, pady=20)
main_frame.pack(expand=True, fill="both")

header_frame = tk.Frame(main_frame, bg="#f4f4f4")
header_frame.pack(fill="x", pady=(0, 20))

# Cabeçalho
title_label = tk.Label(header_frame, text="Sistema de Detecção de Objetos Cortantes",
                       font=("Segoe UI", 14, "bold"), bg="#f4f4f4", fg="#2c3e50")
title_label.pack(anchor="center")

desc_label = tk.Label(header_frame, text="Selecione um arquivo e insira o e-mail para notificação.",
                      font=("Segoe UI", 10), bg="#f4f4f4", fg="#555")
desc_label.pack(anchor="center", pady=(5, 0))

# Campos de entrada

tk.Label(main_frame, text="E-mail de destino:", font=label_font, bg="#f4f4f4").pack(anchor="w")
email_entry = tk.Entry(main_frame, font=entry_font, width=40, relief="solid", bd=1)
email_entry.pack(pady=(0, 10))

select_button = tk.Button(main_frame, text="Selecionar Arquivo", font=button_font,
                          bg="#3498db", fg="white", padx=10, pady=5, relief="flat", command=selec_arquivo)
select_button.pack(fill="x", pady=(0, 10))

start_button = tk.Button(main_frame, text="Iniciar Detecção", font=button_font,
                         bg="#27ae60", fg="white", padx=10, pady=5, relief="flat", command=thread_verificacao)
start_button.pack(fill="x", pady=(10, 0))

progress_bar = ttk.Progressbar(main_frame, length=200, mode="indeterminate")

loading_label = tk.Label(main_frame, text="", font=("Segoe UI", 10), bg="#f4f4f4", fg="#27ae60")
loading_label.pack(pady=(0, 20))

root.mainloop()