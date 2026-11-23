import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import center_of_mass
from torch.utils.data import ConcatDataset, DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    DATA_DIR,
    N_CALIB,
    N_FASHION,
    N_ID_EVAL,
    N_KMNIST,
    N_SCRIBBLES,
    SEED,
)

emnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda img: torch.rot90(img, 1, [1, 2])),
    transforms.Lambda(lambda img: torch.flip(img, [2])),
    transforms.Normalize((0.5,), (0.5,)),
])

ood_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])


def build_loaders(seed: int = SEED) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    train_dataset = datasets.EMNIST(
        root=str(DATA_DIR),
        split="byclass",
        train=True,
        download=True,
        transform=emnist_transform,
    )
    test_dataset = datasets.EMNIST(
        root=str(DATA_DIR),
        split="byclass",
        train=False,
        download=True,
        transform=emnist_transform,
    )

    n_test_total = len(test_dataset)
    assert N_CALIB + N_ID_EVAL <= n_test_total, "Nema dovoljno uzoraka u test_dataset-u"

    perm = torch.randperm(n_test_total, generator=generator).tolist()
    calib_ind = perm[:N_CALIB]
    id_eval_ind = perm[N_CALIB : N_CALIB + N_ID_EVAL]
    rest_test_ind = perm[N_CALIB + N_ID_EVAL :]

    calib_dataset = Subset(test_dataset, calib_ind)
    id_eval_dataset = Subset(test_dataset, id_eval_ind)
    rest_test_dataset = Subset(test_dataset, rest_test_ind)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(rest_test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    calib_loader = DataLoader(calib_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    id_eval_loader = DataLoader(id_eval_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    kmnist_test = datasets.KMNIST(
        root=str(DATA_DIR), train=False, download=False, transform=ood_transform
    )
    fashion_test = datasets.FashionMNIST(
        root=str(DATA_DIR), train=False, download=True, transform=ood_transform
    )

    k_idx = torch.randperm(len(kmnist_test), generator=generator)[:N_KMNIST].tolist()
    f_idx = torch.randperm(len(fashion_test), generator=generator)[:N_FASHION].tolist()

    kmnist_sub = TensorDataset(
        torch.stack([kmnist_test[i][0] for i in k_idx]),
        torch.full((N_KMNIST,), -1, dtype=torch.long),
    )
    fashion_sub = TensorDataset(
        torch.stack([fashion_test[i][0] for i in f_idx]),
        torch.full((N_FASHION,), -1, dtype=torch.long),
    )

    scribble_tensors = []
    for _ in range(N_SCRIBBLES):
        img = Image.new("L", (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        n_strokes = int(rng.integers(2, 6))
        for _ in range(n_strokes):
            pts = [
                (int(rng.integers(2, 26)), int(rng.integers(2, 26)))
                for _ in range(int(rng.integers(2, 5)))
            ]
            draw.line(pts, fill=int(rng.integers(180, 256)), width=int(rng.integers(1, 3)))
        scribble_tensors.append(ood_transform(img))

    scribble_sub = TensorDataset(
        torch.stack(scribble_tensors),
        torch.full((N_SCRIBBLES,), -1, dtype=torch.long),
    )

    ood_dataset = ConcatDataset([kmnist_sub, fashion_sub, scribble_sub])
    ood_loader = DataLoader(ood_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "calib_loader": calib_loader,
        "id_eval_loader": id_eval_loader,
        "ood_loader": ood_loader,
    }


def preprocess_canvas_image(pil_image: Image.Image, device: torch.device) -> torch.Tensor:
    img_resized = pil_image.resize((28, 28), Image.Resampling.LANCZOS)
    img_resized = img_resized.rotate(270, expand=True)
    img_resized = ImageOps.mirror(img_resized)

    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    if img_np.sum() > 0:
        cy, cx = center_of_mass(img_np)
        rows, cols = img_np.shape
        shift_y = int(rows / 2.0 - cy)
        shift_x = int(cols / 2.0 - cx)
        img_np = np.roll(img_np, shift_y, axis=0)
        img_np = np.roll(img_np, shift_x, axis=1)

    tensor_img = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    tensor_img = torch.rot90(tensor_img, 1, [2, 3])
    tensor_img = torch.flip(tensor_img, [3])
    tensor_img = (tensor_img - 0.5) / 0.5
    return tensor_img