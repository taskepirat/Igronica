import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from config import N_PASSES, OOD_PERCENTILE, UNC_THRESHOLD, get_emnist_char


def enable_mc_dropout(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def fit_temperature(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in calib_loader:
            images = images.to(device)
            logits_list.append(model(images))
            labels_list.append(labels.to(device))

    calib_logits = torch.cat(logits_list, dim=0)
    calib_labels = torch.cat(labels_list, dim=0)

    log_temp = nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temp], lr=lr, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temp = torch.exp(log_temp)
        loss = F.cross_entropy(calib_logits / temp, calib_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temp).detach().item())


def compute_energy_ood_score(logits: torch.Tensor, temp: float = 1.0) -> torch.Tensor:
    return -temp * torch.logsumexp(logits / temp, dim=1)


def calibrate_ood_threshold(
    model: nn.Module,
    calib_loader,
    device: torch.device,
    percentile: float = OOD_PERCENTILE,
    temp: float = 1.0,
) -> float:
    model.eval()
    scores = []
    with torch.no_grad():
        for images, _ in calib_loader:
            logits = model(images.to(device))
            scores.append(compute_energy_ood_score(logits, temp=temp).cpu().numpy())
    return float(np.percentile(np.concatenate(scores), percentile))


def run_vanilla_vit(model: nn.Module, loader, device: torch.device):
    model.eval()
    all_preds, all_labels, all_rejected, all_ood_proxy, all_unc = [], [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            rejected = torch.zeros_like(preds, dtype=torch.bool)
            ood_proxy = 1.0 - max_probs

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_rejected.append(rejected.cpu().numpy())
            all_ood_proxy.append(ood_proxy.cpu().numpy())
            all_unc.append(ood_proxy.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_rejected),
        np.concatenate(all_ood_proxy),
        np.concatenate(all_unc),
    )


def run_mc_dropout_vit(
    model: nn.Module,
    loader,
    device: torch.device,
    n_passes: int = N_PASSES,
    unc_threshold: float = UNC_THRESHOLD,
):
    enable_mc_dropout(model)
    all_preds, all_labels, all_rejected, all_ood_proxy, all_unc = [], [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            mc_probs = torch.stack(
                [F.softmax(model(images), dim=1) for _ in range(n_passes)], dim=0
            )
            mean_probs = mc_probs.mean(dim=0)
            _, preds = torch.max(mean_probs, dim=1)

            uncertainty = mc_probs.std(dim=0).mean(dim=1)
            rejected = uncertainty > unc_threshold
            ood_proxy = uncertainty

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_rejected.append(rejected.cpu().numpy())
            all_ood_proxy.append(ood_proxy.cpu().numpy())
            all_unc.append(uncertainty.cpu().numpy())

    model.eval()
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_rejected),
        np.concatenate(all_ood_proxy),
        np.concatenate(all_unc),
    )


def run_mc_temp_vit(
    model: nn.Module,
    loader,
    device: torch.device,
    temperature: float,
    n_passes: int = N_PASSES,
    unc_threshold: float = UNC_THRESHOLD,
):
    enable_mc_dropout(model)
    all_preds, all_labels, all_rejected, all_unc, all_ood_proxy = [], [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            mc_probs = torch.stack(
                [F.softmax(model(images) / temperature, dim=1) for _ in range(n_passes)],
                dim=0,
            )
            mean_probs = mc_probs.mean(dim=0)
            _, preds = torch.max(mean_probs, dim=1)

            uncertainty = mc_probs.std(dim=0).mean(dim=1)
            rejected = uncertainty > unc_threshold
            ood_proxy = uncertainty

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_rejected.append(rejected.cpu().numpy())
            all_unc.append(uncertainty.cpu().numpy())
            all_ood_proxy.append(ood_proxy.cpu().numpy())

    model.eval()
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_rejected),
        np.concatenate(all_unc),
        np.concatenate(all_ood_proxy),
    )


def run_mc_temp_ood_vit(
    model: nn.Module,
    loader,
    device: torch.device,
    temperature: float,
    ood_threshold: float,
    n_passes: int = N_PASSES,
    unc_threshold: float = UNC_THRESHOLD,
):
    enable_mc_dropout(model)
    all_preds, all_labels, all_rejected, all_unc, all_ood_scores = [], [], [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            mc_logits = torch.stack([model(images) for _ in range(n_passes)], dim=0)
            mc_probs = F.softmax(mc_logits / temperature, dim=-1)
            mean_probs = mc_probs.mean(dim=0)
            _, preds = torch.max(mean_probs, dim=1)

            uncertainty = mc_probs.std(dim=0).mean(dim=1)
            ood_score = compute_energy_ood_score(mc_logits.mean(dim=0), temp=temperature)
            rejected = (uncertainty > unc_threshold) | (ood_score > ood_threshold)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_rejected.append(rejected.cpu().numpy())
            all_unc.append(uncertainty.cpu().numpy())
            all_ood_scores.append(ood_score.cpu().numpy())

    model.eval()
    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_rejected),
        np.concatenate(all_unc),
        np.concatenate(all_ood_scores),
    )


def evaluate_machine(
    name: str,
    id_preds: np.ndarray,
    id_labels: np.ndarray,
    id_rejected: np.ndarray,
    id_uncertainty: np.ndarray,
    ood_rejected: np.ndarray,
    ood_uncertainty: np.ndarray,
    id_ood_scores: np.ndarray,
    ood_ood_scores: np.ndarray,
) -> dict:
    id_raw_acc = float(np.mean(id_preds == id_labels))

    accepted_mask = ~id_rejected
    id_coverage = float(np.mean(accepted_mask))
    if np.sum(accepted_mask) > 0:
        id_selective_acc = float(np.mean(id_preds[accepted_mask] == id_labels[accepted_mask]))
    else:
        id_selective_acc = 0.0

    ood_rejection_rate = float(np.mean(ood_rejected))
    y_true_ood = np.concatenate([np.zeros(len(id_ood_scores)), np.ones(len(ood_ood_scores))])
    y_scores_ood = np.concatenate([id_ood_scores, ood_ood_scores])
    auroc = float(roc_auc_score(y_true_ood, y_scores_ood))

    return {
        "machine": name,
        "id_raw_acc": id_raw_acc,
        "id_coverage": id_coverage,
        "id_selective_acc": id_selective_acc,
        "id_unc_mean": float(np.mean(id_uncertainty)),
        "ood_unc_mean": float(np.mean(ood_uncertainty)),
        "ood_rejection_rate": ood_rejection_rate,
        "id_vs_ood_auroc": auroc,
    }


def evaluate_all_machines(
    model: nn.Module,
    id_eval_loader,
    ood_loader,
    device: torch.device,
    temperature: float,
    ood_threshold: float,
    n_passes: int = N_PASSES,
    unc_threshold: float = UNC_THRESHOLD,
) -> list[dict]:
    results = []

    m1_id_p, m1_id_l, m1_id_r, m1_id_u, m1_id_pr = run_vanilla_vit(model, id_eval_loader, device)
    _, _, m1_ood_r, m1_ood_u, m1_ood_pr = run_vanilla_vit(model, ood_loader, device)
    results.append(
        evaluate_machine(
            "Mašina 1: Vanilla ViT",
            m1_id_p, m1_id_l, m1_id_r, m1_id_u, m1_ood_r, m1_ood_u, m1_id_pr, m1_ood_pr,
        )
    )

    m2_id_p, m2_id_l, m2_id_r, m2_id_u, m2_id_pr = run_mc_dropout_vit(
        model, id_eval_loader, device, n_passes=n_passes, unc_threshold=unc_threshold
    )
    _, _, m2_ood_r, m2_ood_u, m2_ood_pr = run_mc_dropout_vit(
        model, ood_loader, device, n_passes=n_passes, unc_threshold=unc_threshold
    )
    results.append(
        evaluate_machine(
            f"Mašina 2: MC Dropout (N={n_passes}, prag={unc_threshold})",
            m2_id_p, m2_id_l, m2_id_r, m2_id_u, m2_ood_r, m2_ood_u, m2_id_pr, m2_ood_pr,
        )
    )

    m3_id_p, m3_id_l, m3_id_r, m3_id_u, m3_id_pr = run_mc_temp_vit(
        model,
        id_eval_loader,
        device,
        temperature=temperature,
        n_passes=n_passes,
        unc_threshold=unc_threshold,
    )
    _, _, m3_ood_r, m3_ood_u, m3_ood_pr = run_mc_temp_vit(
        model,
        ood_loader,
        device,
        temperature=temperature,
        n_passes=n_passes,
        unc_threshold=unc_threshold,
    )
    results.append(
        evaluate_machine(
            f"Mašina 3: MC + Temp (T={temperature:.2f}, prag={unc_threshold})",
            m3_id_p, m3_id_l, m3_id_r, m3_id_u, m3_ood_r, m3_ood_u, m3_id_pr, m3_ood_pr,
        )
    )

    m4_id_p, m4_id_l, m4_id_r, m4_id_u, m4_id_e = run_mc_temp_ood_vit(
        model,
        id_eval_loader,
        device,
        temperature=temperature,
        ood_threshold=ood_threshold,
        n_passes=n_passes,
        unc_threshold=unc_threshold,
    )
    _, _, m4_ood_r, m4_ood_u, m4_ood_e = run_mc_temp_ood_vit(
        model,
        ood_loader,
        device,
        temperature=temperature,
        ood_threshold=ood_threshold,
        n_passes=n_passes,
        unc_threshold=unc_threshold,
    )
    results.append(
        evaluate_machine(
            f"Mašina 4: MC + Temp + Energy OOD (prag={ood_threshold:.2f})",
            m4_id_p, m4_id_l, m4_id_r, m4_id_u, m4_ood_r, m4_ood_u, m4_id_e, m4_ood_e,
        )
    )

    return results


def build_results_table(results_list: list[dict]) -> pd.DataFrame:
    df_results = pd.DataFrame(results_list).set_index("machine")
    return pd.DataFrame({
        "ID Sirova tačnost": (df_results["id_raw_acc"] * 100).round(2).astype(str) + "%",
        "ID Pokrivenost": (df_results["id_coverage"] * 100).round(2).astype(str) + "%",
        "ID Selektivna tačnost": (df_results["id_selective_acc"] * 100).round(2).astype(str) + "%",
        "Prosek unc (ID)": df_results["id_unc_mean"].round(5),
        "Prosek unc (OOD)": df_results["ood_unc_mean"].round(5),
        "Odbijeno OOD": (df_results["ood_rejection_rate"] * 100).round(2).astype(str) + "%",
        "AUROC": df_results["id_vs_ood_auroc"].round(4),
    })


def predict_single_all_machines(
    model: nn.Module,
    tensor_img: torch.Tensor,
    temperature: float,
    ood_threshold: float,
    n_passes: int = N_PASSES,
    unc_threshold: float = UNC_THRESHOLD,
) -> list[dict]:
    out = []

    model.eval()
    with torch.no_grad():
        logits_m1 = model(tensor_img)
        probs_m1 = F.softmax(logits_m1, dim=1)[0]
        max_p1, idx1 = torch.max(probs_m1, dim=0)
        char1 = get_emnist_char(int(idx1.item()))
        out.append({
            "machine": "M1: Vanilla",
            "status": f"'{char1}' ({max_p1.item() * 100:.1f}%)",
            "detail": f"1 - max_p = {1.0 - max_p1.item():.4f}",
        })

    enable_mc_dropout(model)
    with torch.no_grad():
        mc_p2 = torch.stack(
            [F.softmax(model(tensor_img), dim=1)[0] for _ in range(n_passes)], dim=0
        )
        mean_p2 = mc_p2.mean(dim=0)
        unc2 = float(mc_p2.std(dim=0).mean().item())
        prob2, idx2 = torch.max(mean_p2, dim=0)
        char2 = get_emnist_char(int(idx2.item()))
        rej2 = unc2 > unc_threshold
        out.append({
            "machine": "M2: MC Dropout",
            "status": "ODBIJENO" if rej2 else f"'{char2}' ({prob2.item() * 100:.1f}%)",
            "detail": f"unc={unc2:.5f} (prag {unc_threshold})",
        })

    enable_mc_dropout(model)
    with torch.no_grad():
        mc_p3 = torch.stack(
            [F.softmax(model(tensor_img) / temperature, dim=1)[0] for _ in range(n_passes)],
            dim=0,
        )
        mean_p3 = mc_p3.mean(dim=0)
        unc3 = float(mc_p3.std(dim=0).mean().item())
        prob3, idx3 = torch.max(mean_p3, dim=0)
        char3 = get_emnist_char(int(idx3.item()))
        rej3 = unc3 > unc_threshold
        out.append({
            "machine": "M3: MC + Temp",
            "status": "ODBIJENO" if rej3 else f"'{char3}' ({prob3.item() * 100:.1f}%)",
            "detail": f"unc={unc3:.5f} | T={temperature:.2f}",
        })

    enable_mc_dropout(model)
    with torch.no_grad():
        mc_l4 = torch.stack([model(tensor_img) for _ in range(n_passes)], dim=0)
        mc_p4 = F.softmax(mc_l4 / temperature, dim=-1)
        mean_p4 = mc_p4.mean(dim=0)[0]
        unc4 = float(mc_p4.std(dim=0).mean(dim=-1)[0].item())
        energy4 = float(compute_energy_ood_score(mc_l4.mean(dim=0), temp=temperature)[0].item())
        prob4, idx4 = torch.max(mean_p4, dim=0)
        char4 = get_emnist_char(int(idx4.item()))

        rej_unc4 = unc4 > unc_threshold
        rej_ood4 = energy4 > ood_threshold
        rej4 = rej_unc4 or rej_ood4

        reasons = []
        if rej_unc4:
            reasons.append("unc")
        if rej_ood4:
            reasons.append("energija")
        reason_str = "+".join(reasons) if reasons else "OK"

        out.append({
            "machine": "M4: MC+T+Energy",
            "status": f"ODBIJENO ({reason_str})" if rej4 else f"'{char4}' ({prob4.item() * 100:.1f}%)",
            "detail": f"unc={unc4:.5f} | E={energy4:.2f} (prag {ood_threshold:.2f})",
        })

    model.eval()
    return out