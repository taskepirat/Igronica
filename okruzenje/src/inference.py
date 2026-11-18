import torch
import torch.nn.functional as F
import numpy as np

def get_emnist_char(label):
    if label < 10:
        return str(label)
    elif label < 36:
        return chr(ord('A') + (label - 10))
    else:
        return chr(ord('a') + (label - 36))

def preprocess_image(app_image):
    img_resized = app_image.resize((28, 28), 3) # Resampling.LANCZOS
    img_resized = img_resized.rotate(270, expand=True)
    
    from PIL import ImageOps
    img_resized = ImageOps.mirror(img_resized)

    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    if img_np.sum() > 0:
        from scipy.ndimage import center_of_mass
        cy, cx = center_of_mass(img_np)
        rows, cols = img_np.shape
        shift_y = int(rows / 2.0 - cy)
        shift_x = int(cols / 2.0 - cx)
        img_np = np.roll(img_np, shift_y, axis=0)
        img_np = np.roll(img_np, shift_x, axis=1)

    tensor_img = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor_img = torch.rot90(tensor_img, 1, [2, 3])
    tensor_img = torch.flip(tensor_img, [3])
    tensor_img = (tensor_img - 0.5) / 0.5
    return tensor_img