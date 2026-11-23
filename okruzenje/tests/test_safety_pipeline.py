import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in (str(BASE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import get_emnist_char
from data import preprocess_canvas_image
from inference import (
    calibrate_ood_threshold,
    compute_energy_ood_score,
    enable_mc_dropout,
    evaluate_machine,
    predict_single_all_machines,
    run_mc_temp_ood_vit,
    run_vanilla_vit,
)
from model import VisionTransformer


@pytest.fixture
def cpu_device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def small_vit(cpu_device: torch.device) -> VisionTransformer:
    model = VisionTransformer(
        in_channels=1,
        patch_size=4,
        emb_dim=32,
        img_size=28,
        num_classes=62,
        depth=2,
        num_heads=4,
        dropout=0.2,
    ).to(cpu_device)
    model.eval()
    return model


def test_get_emnist_char_boundaries():
    assert get_emnist_char(0) == "0"
    assert get_emnist_char(9) == "9"
    assert get_emnist_char(10) == "A"
    assert get_emnist_char(35) == "Z"
    assert get_emnist_char(36) == "a"
    assert get_emnist_char(61) == "z"


def test_vit_forward_shape_and_finiteness(small_vit: VisionTransformer, cpu_device: torch.device):
    x = torch.randn(3, 1, 28, 28, device=cpu_device)
    with torch.no_grad():
        logits = small_vit(x)

    assert logits.shape == (3, 62)
    assert torch.isfinite(logits).all()


def test_preprocess_canvas_image_output(cpu_device: torch.device):
    img = Image.new("L", (280, 280), color=0)
    draw = ImageDraw.Draw(img)
    draw.line([(60, 60), (220, 220)], fill=255, width=20)

    tensor = preprocess_canvas_image(img, cpu_device)

    assert tensor.shape == (1, 1, 28, 28)
    assert tensor.device == cpu_device
    assert torch.isfinite(tensor).all()
    assert tensor.min().item() >= -1.0001
    assert tensor.max().item() <= 1.0001


def test_enable_mc_dropout_only_toggles_dropout(small_vit: VisionTransformer):
    enable_mc_dropout(small_vit)

    dropout_modules = [m for m in small_vit.modules() if isinstance(m, nn.Dropout)]
    non_dropout_modules = [
        m for m in small_vit.modules() if not isinstance(m, (nn.Dropout, VisionTransformer))
    ]

    assert len(dropout_modules) > 0
    assert all(m.training for m in dropout_modules)
    assert all(not m.training for m in non_dropout_modules)


def test_energy_score_sign_convention(cpu_device: torch.device):
    # Oštri/jaki logiti daju nižu (više negativnu) slobodnu energiju od slabih/ravnih logita
    confident_logits = torch.tensor([[12.0, -2.0, -2.0]], device=cpu_device)
    weak_logits = torch.tensor([[0.5, 0.4, 0.3]], device=cpu_device)

    e_confident = compute_energy_ood_score(confident_logits, temp=1.5).item()
    e_weak = compute_energy_ood_score(weak_logits, temp=1.5).item()

    assert e_confident < e_weak


def test_calibrate_ood_threshold_returns_float_percentile(
    small_vit: VisionTransformer, cpu_device: torch.device
):
    x = torch.randn(16, 1, 28, 28)
    y = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    thresh = calibrate_ood_threshold(small_vit, loader, cpu_device, percentile=75.0, temp=1.0)
    assert isinstance(thresh, float)
    assert np.isfinite(thresh)


def test_vanilla_no_rejection_and_m4_rejection_union(
    small_vit: VisionTransformer, cpu_device: torch.device
):
    x = torch.randn(6, 1, 28, 28)
    y = torch.arange(6, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)

    _, _, m1_rej, _, _ = run_vanilla_vit(small_vit, loader, cpu_device)
    assert not np.any(m1_rej)

    # Strogi pragovi da garantujemo odbijanje u M4
    _, _, m4_rej, _, _ = run_mc_temp_ood_vit(
        small_vit,
        loader,
        cpu_device,
        temperature=1.0,
        ood_threshold=-1e9,
        n_passes=5,
        unc_threshold=0.0,
    )
    assert np.all(m4_rej)


def test_evaluate_machine_metrics_and_auroc():
    id_preds = np.array([0, 1, 2, 3])
    id_labels = np.array([0, 1, 9, 3])  # 3/4 tačno -> sirova tačnost 0.75
    id_rejected = np.array([False, True, False, False])  # pokrivenost 0.75; od [0, 2, 3] tačni su [0, 3] -> 2/3
    id_unc = np.array([0.001, 0.010, 0.002, 0.001])

    ood_rejected = np.array([True, True])
    ood_unc = np.array([0.020, 0.030])

    id_ood_scores = np.array([-13.0, -12.5, -12.8, -13.2])
    ood_ood_scores = np.array([-6.0, -5.5])

    res = evaluate_machine(
        name="Test M4",
        id_preds=id_preds,
        id_labels=id_labels,
        id_rejected=id_rejected,
        id_uncertainty=id_unc,
        ood_rejected=ood_rejected,
        ood_uncertainty=ood_unc,
        id_ood_scores=id_ood_scores,
        ood_ood_scores=ood_ood_scores,
    )

    assert np.isclose(res["id_raw_acc"], 0.75)
    assert np.isclose(res["id_coverage"], 0.75)
    assert np.isclose(res["id_selective_acc"], 2.0 / 3.0)
    assert np.isclose(res["ood_rejection_rate"], 1.0)
    assert np.isclose(res["id_vs_ood_auroc"], 1.0)


def test_predict_single_all_machines_returns_four_entries(
    small_vit: VisionTransformer, cpu_device: torch.device
):
    tensor_img = torch.randn(1, 1, 28, 28, device=cpu_device)
    rows = predict_single_all_machines(
        small_vit,
        tensor_img,
        temperature=1.03,
        ood_threshold=-10.63,
        n_passes=5,
        unc_threshold=0.004,
    )

    assert len(rows) == 4
    for row in rows:
        assert {"machine", "status", "detail"}.issubset(row.keys())
        assert isinstance(row["status"], str) and len(row["status"]) > 0
    assert not small_vit.training