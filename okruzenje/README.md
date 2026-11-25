# ViT EMNIST — Uncertainty Estimation & Energy-Based OOD Detection

A modular PyTorch project on model reliability for **EMNIST (`byclass`, 62 classes)** character classification, using a compact **Vision Transformer** (≈0.21M parameters: `patch_size=4`, `emb_dim=64`, `depth=4`, `num_heads=4`, `dropout=0.1`).

Instead of trusting the raw argmax, the pipeline asks when the model should **abstain**: either because its prediction is unstable across stochastic forward passes (MC Dropout), or because the input looks **out-of-distribution** (OOD) according to the free energy of its logits. Four increasingly strict configurations ("Machines 1–4") are compared on the same in-distribution (ID) and OOD sets, and a Tkinter app lets you draw a character and see all four verdicts side by side.

---

## Model & Data

**Model.** `VisionTransformer` in `src/model.py` (patch embedding → CLS token + learned positional embedding → 4 pre-norm transformer blocks → linear head). It was trained in `notebooks/moj MaLi.ipynb` (there is no standalone training script): 10 epochs, AdamW (`lr=3e-4`, `weight_decay=0.01`), batch size 256, cross-entropy. It reaches **86.31%** accuracy on the 112,323 EMNIST test samples that are not used for calibration or evaluation. The trained checkpoint is committed as `models/vit_emnist.pt` (`notebooks/vit_emnist.pt` is an identical copy).

**Data splits** (all built in `data.py`, seed `42`). The EMNIST `byclass` test split is shuffled and divided into:

- `calib_loader` — 2,000 samples, used to fit the temperature and the energy threshold;
- `id_eval_loader` — 2,000 samples, the ID evaluation set;
- the remaining 112,323 samples (`test_loader`).

`ood_loader` holds 500 OOD samples: 200 KMNIST, 150 FashionMNIST and 150 synthetic scribbles (random strokes drawn with PIL).

---

## Evaluation Progression: Machines 1–4

All four configurations share the same trained model and the same ID/OOD sets. Two values are calibrated on `calib_loader`:

- **Temperature** `T = 1.0262`, fitted by minimizing NLL with L-BFGS (calibration NLL 0.3607 → 0.3601, so the model was already close to calibrated).
- **Energy threshold** `-10.63`, the 95th percentile of the ID free-energy scores.

1. **Machine 1 — Vanilla ViT:** one deterministic `eval()` pass, never rejects. `1 - max_p` serves as the baseline uncertainty / OOD score.
2. **Machine 2 — MC Dropout (`N=20`, `threshold=0.004`):** `nn.Dropout` layers stay active at inference. The prediction is the argmax of the mean softmax over passes; uncertainty is the per-class standard deviation across passes, averaged over classes (`mc_probs.std(dim=0).mean(dim=1)`). A sample is rejected if uncertainty exceeds the threshold.
3. **Machine 3 — MC Dropout + Temperature Scaling (`T=1.03`):** same as Machine 2, but logits are divided by `T` before the softmax. Same uncertainty threshold.
4. **Machine 4 — MC Dropout + Temperature Scaling + Energy OOD (`threshold=-10.63`):** adds a free-energy score computed on the MC-averaged logits:

   $$E(z;T) = -T \log \sum_{c} \exp(z_c / T)$$

   Higher (less negative) energy means weaker overall logit support and a higher OOD suspicion. A sample is rejected if `uncertainty > unc_threshold` **or** `energy > ood_threshold`.

**Metrics.** *Raw accuracy* is accuracy on all 2,000 ID samples. *Coverage* is the fraction of ID samples accepted. *Selective accuracy* is accuracy on the accepted ID samples only. *OOD rejected* is the fraction of the 500 OOD samples that are rejected. *AUROC* separates ID from OOD using each machine's own scalar score (see the notes below the table).

---

## Benchmark Results

| Machine | ID Raw Acc | ID Coverage | ID Selective Acc | Mean Unc (ID) | Mean Unc (OOD) | OOD Rejected | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Machine 1: Vanilla ViT** | 86.60% | 100.00% | 86.60% | 0.13064 | 0.46689 | 0.00% | 0.8908 |
| **Machine 2: MC Dropout** | 86.55% | 93.15% | 88.57% | 0.00115 | 0.00856 | 90.60% | 0.9753 |
| **Machine 3: MC + Temp** | 86.45% | 93.45% | 88.60% | 0.00115 | 0.00844 | 90.80% | 0.9759 |
| **Machine 4: MC + Temp + Energy OOD** | 86.30% | 91.20% | 88.65% | 0.00115 | 0.00844 | 96.20% | 0.9882 |

Notes on reading the table:

- Output of `python bench_eval.py` (seed 42, CUDA, 2,000 ID and 500 OOD samples). MC Dropout is stochastic, so the Machine 2–4 numbers vary slightly between runs: an earlier run in the notebook differed by up to about 0.8 pp (e.g. Machine 2 OOD rejection `91.40%` vs. `90.60%`). Machine 1 is deterministic.
- **The uncertainty columns are not comparable across machines:** Machine 1 reports `1 - max_p`, Machines 2–4 report the MC standard deviation.
- **AUROC uses a different score per machine:** `1 - max_p` (Machine 1), MC standard deviation (Machines 2–3) and the **free energy alone** (Machine 4). The rejection rule of Machine 4 combines uncertainty and energy, but its AUROC is that of the energy score.
- The command-line and GUI tables print their labels in Serbian (e.g. "Mašina", "Pokrivenost").

---

## Key Takeaways, Limitations & Future Work

* **MC Dropout is a much better OOD signal than `1 - max_p`:** AUROC rises from `0.8908` to `0.9753`, and Machine 2 rejects `90.60%` of OOD samples at a cost of `6.85%` of ID samples (coverage `93.15%`), lifting selective accuracy to `88.57%` (vs. `86.60%` without rejection). Temperature scaling (Machine 3) changes almost nothing (AUROC `0.9759`, coverage `93.45%`), as expected for `T ≈ 1.03`.
* **Free energy buys extra OOD rejection at the cost of ID coverage, not of selective accuracy:** relative to Machine 3, Machine 4 raises OOD rejection from `90.80%` to `96.20%` (+5.4 pp) and lowers coverage from `93.45%` to `91.20%` (−2.25 pp), while selective accuracy stays essentially unchanged (`88.60%` → `88.65%`, within noise). The energy score alone also separates ID from OOD better than the MC standard deviation (AUROC `0.9882` vs. `0.9759`).
* **Statistical caveat:** with 2,000 ID and 500 OOD samples, one seed and one OOD set, binomial standard errors are roughly 0.7–0.8 pp for the ID metrics and 0.7–1.3 pp for the OOD rejection rates. Differences of about 1 pp should not be over-interpreted, and AUROC depends heavily on how easy the OOD set is (the synthetic scribbles in particular).
* **False acceptance on visually similar out-of-domain symbols:** a scalar energy threshold calibrated at the 95th percentile of ID data does not guarantee rejection of every non-EMNIST input. In the GUI, drawing the Cyrillic character **`ж`** (outside the 62-class label space) can still be accepted as `'X'`. A plausible (untested) explanation is that its intersecting strokes produce strong, consistent activations that keep the free energy below `-10.63` and the MC variance low.
* **Planned improvements:** explore stronger OOD/reliability signals (Mahalanobis/feature-space distance, ensembles, entropy); per-class or stricter calibration regimes; a larger and more varied OOD benchmark; and evaluation on adversarial / perturbed stroke shifts.

---

## Setup

Requires **Python 3.10+**.

```bash
pip install -r requirements.txt
```

- Tkinter ships with Python on Windows/macOS; on Debian/Ubuntu install it with `sudo apt install python3-tk`.
- On the first benchmark/GUI run, `torchvision` downloads EMNIST (`byclass`), FashionMNIST and KMNIST into `notebooks/data/` (git-ignored; EMNIST alone is several hundred MB).
- A CUDA GPU is used automatically if available, otherwise the CPU.

## Running the Project

Run all commands from the project directory (`okruzenje/`).

1. **Unit tests (fast; no dataset download, checkpoint or GUI required):**
   `pytest`

2. **CLI benchmark evaluation (Machines 1–4 table):**
   `python bench_eval.py`

3. **Interactive Tkinter application:**
   `python src/app.py`

   On start-up the app fits the temperature, calibrates the energy threshold and runs the full benchmark once, so the window can take a while to appear on CPU. Draw a character, press **Prepoznaj (M1–M4)** to see each machine's verdict and signals, and use **Ponovo izračunaj tabelu** to recompute the benchmark table.

---

## Project Structure

```text
okruzenje/
├── config.py                  # Paths, seeds, thresholds, loader sizes, EMNIST label mapping
├── data.py                    # ID/OOD DataLoader construction and Tkinter canvas preprocessing
├── bench_eval.py              # CLI evaluation across Machines 1–4 and summary table printout
├── requirements.txt           # Python dependencies
├── pyproject.toml             # Pytest configuration (tests/ only; skips env, data, models, notebooks)
├── models/
│   └── vit_emnist.pt          # Trained ViT checkpoint
├── notebooks/
│   ├── moj MaLi.ipynb         # Model definition, training, and the original M1–M4 experiments
│   └── PrviPokusaji.ipynb     # Early experiments
├── src/
│   ├── __init__.py
│   ├── model.py               # VisionTransformer architecture and checkpoint loader
│   ├── inference.py           # MC Dropout, temperature scaling, energy OOD, metrics, inference
│   └── app.py                 # Tkinter GUI (drawing canvas + M1–M4 predictions + benchmark table)
└── tests/
    └── test_safety_pipeline.py
```
