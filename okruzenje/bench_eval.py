import sys
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
for p in (str(BASE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import N_PASSES, OOD_PERCENTILE, UNC_THRESHOLD, seed_everything
from data import build_loaders
from model import load_trained_vit
from inference import (
    build_results_table,
    calibrate_ood_threshold,
    evaluate_all_machines,
    fit_temperature,
)


def main() -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Uređaj: {device}")

    loaders = build_loaders()
    model = load_trained_vit(device)

    optimal_T = fit_temperature(model, loaders["calib_loader"], device)
    ood_threshold = calibrate_ood_threshold(
        model, loaders["calib_loader"], device, percentile=OOD_PERCENTILE, temp=optimal_T
    )
    print(f"Optimalna temperatura T: {optimal_T:.4f} | OOD prag: {ood_threshold:.4f}")

    results = evaluate_all_machines(
        model=model,
        id_eval_loader=loaders["id_eval_loader"],
        ood_loader=loaders["ood_loader"],
        device=device,
        temperature=optimal_T,
        ood_threshold=ood_threshold,
        n_passes=N_PASSES,
        unc_threshold=UNC_THRESHOLD,
    )

    df_pretty = build_results_table(results)
    print("\n=== UPOREDNA TABELA: MAŠINE 1–4 ===")
    print(df_pretty.to_string())


if __name__ == "__main__":
    main()