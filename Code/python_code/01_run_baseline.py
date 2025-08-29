# -*- coding: utf-8 -*-
"""
Main runner aligned to the latest Rmd logic with CRRA + CARA utilities
+ belief-weighted CE (mixture across precipitation states)
+ risk-increasing vs risk-reducing classification.
"""
# REPL/Jupyter setup — run this first
from pathlib import Path
import sys

REPO_ROOT = Path('/Users/jaeseokhwang/dist_yield').resolve()     # your repo root
CODE_DIR  = REPO_ROOT / 'Code' / 'python_code'                    # where the .py files live

# make your modules importable
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# now imports will work
from corn_n_response import (
    CFG, load_data, make_features,
    train_baselines, build_prediction_grid, predict_all,
    estimate_mbme_for_grid, compute_utilities,
    aggregate_ce_with_beliefs, classify_input_risk, argmax_n
)

# paths for data/output in REPL
DATA_PATH = REPO_ROOT / 'Data' / 'Processed' / 'Analysis_ready' / 'df_il.csv'
OUT_DATA  = REPO_ROOT / 'Results' / 'Data'

# print("DATA exists?", DATA_PATH.exists(), DATA_PATH)

OUT_DATA.mkdir(parents=True, exist_ok=True)

# ---- Config
cfg = CFG(
    p_yield=5.0,
    p_n=0.8,
    cara_r=0.02,
    crra_gamma=2.0,
    support_points=600
)

# ---- Subjective beliefs over precipitation groups
BELIEFS = {
    "Very Low": 0.2,
    "Low": 0.2,
    "Moderate": 0.2,
    "High": 0.2,
    "Very High": 0.2
}

# print(f"Loading data from: {DATA_PATH}")

df = load_data(str(DATA_PATH))
df = make_features(df)

# ---- Train
# print("Training baseline models...")
df_fit, models = train_baselines(df)

# ---- Build grid & predict
print("Building prediction grid & predicting moments...")
grid = build_prediction_grid(df_fit, n_points=50)
grid_pred = predict_all(models, grid)

# ---- MBME densities
print("Estimating MBME densities (GAM, FE)...")
mbme_gam = estimate_mbme_for_grid(grid_pred, family="gam", cfg=cfg)
mbme_fe  = estimate_mbme_for_grid(grid_pred, family="fe",  cfg=cfg)

# ---- Utilities (RN, CARA, CRRA; normal approx + MBME)
print("Computing utilities...")
util_grid = compute_utilities(grid_pred, pd.concat([mbme_gam, mbme_fe], ignore_index=True), cfg)

# ---- Belief-weighted CE curves per family
print("Aggregating CE with subjective beliefs...")
ce_mix_gam = aggregate_ce_with_beliefs(util_grid, "gam", cfg, BELIEFS)
ce_mix_fe  = aggregate_ce_with_beliefs(util_grid, "fe",  cfg, BELIEFS)

# ---- Risk classification via Cov(g, g_N)
print("Classifying input risk (risk-increasing vs risk-reducing)...")
risk_class = classify_input_risk(grid_pred, BELIEFS, use_col="mu1_hat_ct_gam")

# ---- Optimal N (argmax CE) per approach
opt_gam = argmax_n(ce_mix_gam, ["ce_rn","ce_cara_norm","ce_cara_mbme"])
opt_fe  = argmax_n(ce_mix_fe,  ["ce_rn","ce_cara_norm","ce_cara_mbme"])

# ---- Save outputs
print("Saving outputs...")
df_fit.to_csv(OUT_DATA / "dat_com_mutate.csv", index=False)
grid_pred.to_csv(OUT_DATA / "moments_grid_raw.csv", index=False)
mbme_gam.to_csv(OUT_DATA / "mbme_density_gam.csv", index=False)
mbme_fe.to_csv(OUT_DATA / "mbme_density_fe.csv", index=False)
util_grid.to_csv(OUT_DATA / "utility_grid_compare.csv", index=False)
ce_mix_gam.to_csv(OUT_DATA / "ce_mixture_gam.csv", index=False)
ce_mix_fe.to_csv(OUT_DATA / "ce_mixture_fe.csv", index=False)
risk_class.to_csv(OUT_DATA / "risk_classification.csv", index=False)
opt_gam.to_csv(OUT_DATA / "nstar_gam.csv", index=False)
opt_fe.to_csv(OUT_DATA / "nstar_fe.csv", index=False)
with open(OUT_DATA / "beliefs_used.json", "w") as f:
    json.dump(BELIEFS, f, indent=2)

print("Done.")
