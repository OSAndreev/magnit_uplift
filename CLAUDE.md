# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

3rd place solution for Magnit Tech x HSE 2026 Hackathon — "Machine Learning for Retail", uplift modeling track. The goal is to predict CATE (heterogeneous treatment effects) on customer spending (`rec_spend`) across 3 marketing channels and rank customers by predicted uplift.

## Environment Setup

```bash
pip install -r requirements.txt
```

Key dependencies: `lightgbm`, `causalml`, `econml`, `scikit-uplift`, `optuna`, `torch` (Apple Silicon MPS target), `shap`.

## Running Notebooks

```bash
jupyter notebook   # or jupyter lab
```

Notebooks must be run in order due to artifact dependencies:
1. `EDA.ipynb` → produces `eda_artifacts/`
2. `pilot_modeling.ipynb` → produces `pilot_artifacts/` (models, OOF predictions, Optuna params)
3. `pilot_modeling_final.ipynb` → loads saved params, trains final models, outputs submission CSV
4. `dl_uplift_benchmark.ipynb` → independent DL benchmark (produces `dl_artifacts/`)

To reproduce the final submission only, run `pilot_modeling_final.ipynb` — it loads pre-saved Optuna params without needing to rerun search.

## Architecture

Notebook-only codebase — all logic is self-contained in four Jupyter notebooks. No Python package/module structure.

### Data

- `train.parquet` — 355,246 rows; `test.parquet` — 118,414 rows
- Key columns: `user_id`, `treatment_flg` (binary, 49.6% treated), `communication_type` (com_type_1/2/3), `rec_spend` (continuous target, ~90% zeros)
- 88 raw feature columns covering demographics, RFM, category affinity (cat5/6/7), marketing history

### Evaluation Protocol

`StratifiedKFold(n_splits=5)` stratified on `treatment_flg × communication_type` (6 strata).
- **Primary metric**: `uplift@10` in rubles — mean spend difference between T=1 and T=0 in top 10% scored users
- **Secondary**: `qini_auc_score` (binary scale from scikit-uplift)
- 80% bootstrap CI (100 resamples, 10th/90th percentile)

**Important**: `pilot_summary.json` stores binary-scale OOF (~0.03–0.06). `model_comparison.csv` and notebook cell outputs use the correct ruble scale (~10–22 ₽). Always use ruble-scale numbers for comparison.

### Feature Engineering

`engineer_features(df)` — defined inline in `pilot_modeling.ipynb` and `dl_uplift_benchmark.ipynb` — adds 10 ratio/interaction features on top of 88 raw (96 total):
- `cat7_share`, `cat6_share`, `cat5_share` — category share of RTO
- `mkt_resp_rate`, `mkt_view_rate` — marketing responsiveness rates
- `spend_cv` — stdtv/atv (coefficient of variation)
- `trn_density` — n_trn/n_days_life × 365 (transactions per year)
- `recency_ratio`, `cat_breadth_ratio`, `cat7_vs_last_visit`

### Modeling: X-Learner (Final Approach)

Two-stage meta-learner implemented with LightGBM:
1. Stage 1: Fit `μ₁(X)` on treated, `μ₀(X)` on control
2. Stage 2: Imputed effects `D₁ = Y₁ − μ₀(X₁)`, `D₀ = μ₁(X₀) − Y₀`; fit `τ₁`, `τ₀`
3. CATE: `g(X)·τ₀(X) + (1−g(X))·τ₁(X)` where `g(X)` = propensity score (logistic regression)

**Final ensemble** (rank blend): `0.7 × perchan_ranks + 0.3 × global_ranks`, 5-seed averaging each.

### Hyperparameter Optimization

Optuna runs on 40% subsample (saves time). Best params saved as JSON in `pilot_artifacts/`:
- `optuna_xl_params.json` — global X-Learner (54 trials, subsample uplift@10: 24.57)
- `perchan_xl_params.json` — per-channel params (3 separate dicts)
- `optuna_xl_v2_params.json` — overlap-validated variant (top25 features)
- `optuna_xl_v3_params.json` — tweedie + MAE objective variant

### OOF Benchmark Results (ruble-scale uplift@10)

| Model | uplift@10 (₽) |
|---|---|
| Blend 0.7/0.3 (final) | 21.24 |
| X-Learner Optuna global | 21.07 |
| Per-Channel X-Learner | 20.61 |
| X-Learner default | 19.96 |
| T-Learner | 18.50 |

### Deep Learning Models (`dl_uplift_benchmark.ipynb`)

PyTorch on Apple Silicon MPS. Architecture: shared MLP encoder (256, 128) + task-specific heads (64).
- `TARNet` — two heads (T=1, T=0)
- `CFRNet` — TARNet + IPM regularizer (Wasserstein or MMD)
- `DragonNet` — TARNet + propensity head
- `RERUM` — DragonNet + ZILN loss + ranking losses + uncertainty weighting

DL results use binary uplift@10 scale (not directly comparable to LGB ruble-scale results).

### Saved Artifacts

- `model_artifacts/` — serialized models (`global_xlearner.pkl` 21MB, `perchan_xlearner.pkl` 45MB)
- `pilot_artifacts/` — Optuna params JSONs, OOF `.npy` arrays, submission CSVs
- `dl_artifacts/` — DL OOF predictions
- `eda_artifacts/` — EDA charts, feature configs, hypothesis results
