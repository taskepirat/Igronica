import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import torch

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"
for p in (str(BASE_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from config import N_PASSES, OOD_PERCENTILE, UNC_THRESHOLD, seed_everything
from data import build_loaders, preprocess_canvas_image
from model import load_trained_vit
from inference import (
    build_results_table,
    calibrate_ood_threshold,
    evaluate_all_machines,
    fit_temperature,
    predict_single_all_machines,
)

CANVAS_SIZE = 280
BRUSH_RADIUS = 14


class SafetyVitApp:
    def __init__(
        self,
        root: tk.Tk,
        model,
        device: torch.device,
        optimal_T: float,
        ood_threshold: float,
        loaders: dict,
    ):
        self.root = root
        self.root.title("ViT EMNIST — Nesigurnost i Energy OOD (Mašine 1–4)")
        self.model = model
        self.device = device
        self.optimal_T = optimal_T
        self.ood_threshold = ood_threshold
        self.loaders = loaders

        self.app_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.app_draw = ImageDraw.Draw(self.app_image)

        self._build_ui()
        self.refresh_benchmark_table()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(main_frame, text="Crtež (28x28 ulaz)", padding=10)
        left_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        self.canvas = tk.Canvas(
            left_frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1,
        )
        self.canvas.pack()
        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(btn_frame, text="Prepoznaj (M1–M4)", command=self.on_predict).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5)
        )
        ttk.Button(btn_frame, text="Obriši", command=self.on_clear).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0)
        )

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        single_box = ttk.LabelFrame(
            right_frame, text="Rezultat za trenutni crtež (Mašine 1–4)", padding=8
        )
        single_box.pack(fill=tk.X, pady=(0, 10))

        self.single_tree = ttk.Treeview(
            single_box, columns=("machine", "status", "detail"), show="headings", height=4
        )
        self.single_tree.heading("machine", text="Mašina")
        self.single_tree.heading("status", text="Ishod")
        self.single_tree.heading("detail", text="Signali / Pragovi")
        self.single_tree.column("machine", width=140, anchor=tk.W)
        self.single_tree.column("status", width=190, anchor=tk.W)
        self.single_tree.column("detail", width=270, anchor=tk.W)
        self.single_tree.pack(fill=tk.X)

        table_box = ttk.LabelFrame(right_frame, text="Evaluacija na loader-ima (ID vs. OOD)", padding=8)
        table_box.pack(fill=tk.BOTH, expand=True)

        cols = (
            "machine",
            "raw_acc",
            "coverage",
            "sel_acc",
            "unc_id",
            "unc_ood",
            "ood_rej",
            "auroc",
        )
        self.bench_tree = ttk.Treeview(table_box, columns=cols, show="headings", height=5)
        headers = {
            "machine": "Mašina",
            "raw_acc": "Sirova ID",
            "coverage": "Pokrivenost",
            "sel_acc": "Selektivna ID",
            "unc_id": "Unc (ID)",
            "unc_ood": "Unc (OOD)",
            "ood_rej": "Odbijeno OOD",
            "auroc": "AUROC",
        }
        widths = {
            "machine": 230,
            "raw_acc": 85,
            "coverage": 90,
            "sel_acc": 95,
            "unc_id": 80,
            "unc_ood": 85,
            "ood_rej": 95,
            "auroc": 75,
        }
        for c in cols:
            self.bench_tree.heading(c, text=headers[c])
            self.bench_tree.column(
                c, width=widths[c], anchor=tk.CENTER if c != "machine" else tk.W
            )

        self.bench_tree.pack(fill=tk.BOTH, expand=True)
        ttk.Button(
            table_box, text="Ponovo izračunaj tabelu", command=self.refresh_benchmark_table
        ).pack(anchor=tk.E, pady=(8, 0))

    def paint(self, event) -> None:
        x, y = event.x, event.y
        r = BRUSH_RADIUS
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.app_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def on_clear(self) -> None:
        self.canvas.delete("all")
        self.app_image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), color=0)
        self.app_draw = ImageDraw.Draw(self.app_image)
        for row in self.single_tree.get_children():
            self.single_tree.delete(row)

    def on_predict(self) -> None:
        tensor_img = preprocess_canvas_image(self.app_image, self.device)
        rows = predict_single_all_machines(
            self.model,
            tensor_img,
            temperature=self.optimal_T,
            ood_threshold=self.ood_threshold,
            n_passes=N_PASSES,
            unc_threshold=UNC_THRESHOLD,
        )
        for item in self.single_tree.get_children():
            self.single_tree.delete(item)
        for r in rows:
            self.single_tree.insert("", tk.END, values=(r["machine"], r["status"], r["detail"]))

    def refresh_benchmark_table(self) -> None:
        res_all = evaluate_all_machines(
            model=self.model,
            id_eval_loader=self.loaders["id_eval_loader"],
            ood_loader=self.loaders["ood_loader"],
            device=self.device,
            temperature=self.optimal_T,
            ood_threshold=self.ood_threshold,
            n_passes=N_PASSES,
            unc_threshold=UNC_THRESHOLD,
        )
        df_pretty = build_results_table(res_all)

        for item in self.bench_tree.get_children():
            self.bench_tree.delete(item)

        for idx_name, row in df_pretty.iterrows():
            self.bench_tree.insert(
                "",
                tk.END,
                values=(
                    idx_name,
                    row["ID Sirova tačnost"],
                    row["ID Pokrivenost"],
                    row["ID Selektivna tačnost"],
                    row["Prosek unc (ID)"],
                    row["Prosek unc (OOD)"],
                    row["Odbijeno OOD"],
                    row["AUROC"],
                ),
            )


def main() -> None:
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders()
    model = load_trained_vit(device)
    optimal_T = fit_temperature(model, loaders["calib_loader"], device)
    ood_threshold = calibrate_ood_threshold(
        model, loaders["calib_loader"], device, percentile=OOD_PERCENTILE, temp=optimal_T
    )

    root = tk.Tk()
    SafetyVitApp(root, model, device, optimal_T, ood_threshold, loaders)
    root.mainloop()


if __name__ == "__main__":
    main()