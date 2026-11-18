import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import numpy as np
from model import VisionTransformer
from inference import get_emnist_char, preprocess_image

# Učitavanje uređaja i modela
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(
    in_channels=1, patch_size=4, emb_dim=64, img_size=28, 
    num_classes=62, depth=4, num_heads=4, dropout=0.1
).to(device)

checkpoint = torch.load('models/vit_emnist.pt', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

# Tkinter GUI
root = tk.Tk()
root.title("ViT EMNIST - Uncertainty Estimation")

canvas_size = 280
canvas = tk.Canvas(root, width=canvas_size, height=canvas_size, bg="black")
canvas.pack(pady=10)

app_image = Image.new("L", (canvas_size, canvas_size), color=0)
app_draw = ImageDraw.Draw(app_image)

def paint(event):
    x, y = event.x, event.y
    r = 15
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
    app_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

canvas.bind("<B1-Motion>", paint)

label_result = tk.Label(root, text="Nacrtaj nešto i klikni Prepoznaj...", font=("Arial", 14, "bold"))
label_result.pack(pady=5)

def predict_drawing():
    tensor_img = preprocess_image(app_image).to(device)
    
    # Monte Carlo Dropout rad u train mod-u
    model.train()
    n_passes = 20
    all_probs = []

    with torch.no_grad():
        for _ in range(n_passes):
            outputs = model(tensor_img)
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.vstack(all_probs)
    mean_probs = np.mean(all_probs, axis=0)
    uncertainty = np.std(all_probs, axis=0)

    top5_ind = np.argsort(mean_probs)[::-1][:5]
    best_idx = top5_ind[0]
    best_conf = mean_probs[best_idx] * 100
    best_unc = uncertainty[best_idx] * 100
    best_char = get_emnist_char(best_idx)

    CONFIDENCE_THRESHOLD = 50.0
    UNCERTAINTY_THRESHOLD = 15.0

    if best_conf < CONFIDENCE_THRESHOLD or best_unc > UNCERTAINTY_THRESHOLD:
        label_result.config(text=f"Nepoznato / Nesigurno (Conf: {best_conf:.1f}%, Std: ±{best_unc:.1f}%)")
    else:
        label_result.config(text=f"Predikcija: '{best_char}' ({best_conf:.1f}%, ±{best_unc:.1f}%)")

def clear_canvas():
    canvas.delete("all")
    global app_image, app_draw
    app_image = Image.new("L", (canvas_size, canvas_size), color=0)
    app_draw = ImageDraw.Draw(app_image)
    label_result.config(text="Nacrtaj nešto i klikni Prepoznaj...")

btn_predict = tk.Button(root, text="Prepoznaj slovo", command=predict_drawing, font=("Arial", 12), bg="lightgreen")
btn_predict.pack(side=tk.LEFT, padx=30, pady=10)

btn_clear = tk.Button(root, text="Obriši platno", command=clear_canvas, font=("Arial", 12), bg="lightcoral")
btn_clear.pack(side=tk.RIGHT, padx=30, pady=10)

root.mainloop()