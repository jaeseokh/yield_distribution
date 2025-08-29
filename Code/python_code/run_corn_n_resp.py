# -*- coding: utf-8 -*-
"""
Corn N-Response: End-to-end pipeline with rich explanations.

This single file runs the whole analysis in clearly separated "process" blocks.

==========================
### OVERALL PROCESS MAP ###
==========================
(0) Setup & Paths
    - Define repo paths, imports, config, and output directory.

(1) Load Data
    - Read CSV exported from R (or elsewhere).
    - Validate required columns, coerce types.

(2) Feature Engineering
    - Trim outliers in yield and N.
    - Scale yield to [0,1] so all utilities work on a bounded support.
    - Create raw moments y^1, y^2, y^3.

(3) Train Baseline Models (Mean & Moments)
    - μ1 (mean) via GAM           : smooths in n, prcp, gdd, edd (+ linear site vars).
    - μ1 (mean) via Quadratic OLS : polynomial in n, interactions with weather.
    - μ1 (mean) via FE OLS        : like quadratic but includes field fixed effects.
    - Central second/third moments via GAM; also quadratic & FE variants.
    - Raw moment models for y1,y2,y3 (GAM, Quad, FE).
    [Prediction matrices are built using the fitted Patsy design_info to avoid
     shape or intercept/dummy mismatches.]

(4) Build Prediction Grid
    - Create a grid over N and precipitation.
    - Hold other covariates at medians.
    - Label precipitation groups (Very Low ... Very High).

(5) Predict All Moments on the Grid
    - For each model family (GAM, Quadratic, FE), predict μ1, μ2, μ3 (both central/raw).

(6) Maximum-Entropy (MBME) Densities on [0,1]
    - Given raw moments (μ1, μ2, μ3) for a grid point, construct the density f(y)
      on [0,1] that maximizes entropy subject to those moment constraints:
        maximize  H(f) = -∫ f(y) log f(y) dy
        subject to ∫ f(y) dy = 1, ∫ y f(y) dy = μ1, ∫ y^2 f(y) dy = μ2, ∫ y^3 f(y) dy = μ3.
      The solution has the exponential family form:
        f(y) = exp(λ1 y + λ2 y^2 + λ3 y^3) / Z(λ),  y ∈ [0,1]
      where λ are found by solving the moment-matching equations.

(7) Utilities (Risk-Neutral & Risk-Averse)
    - Profit: π = p_yield * y - p_N * n.
    - Risk-neutral EU is just E[π].
    - CARA utility: U(π) = -exp(-r * π).
      * Normal approx closed-form for E[U] when (y ~ Normal):
          E[U] = -exp( -r(a μ) + 0.5 r^2 a^2 Var ) * exp(-r b),
          where a = p_yield, b = -p_N * n.
      * MBME-based EU integrates U over the max-entropy density on [0,1].
    - CRRA utility: U(π) = { (π^(1-γ) - 1) / (1-γ)  if γ≠1;  log(π) if γ=1 }.
      * We integrate numerically under Normal (truncated to [0,1]) or MBME.

(8) Belief-Weighted Certainty Equivalents (CE) across Precip Groups
    - For each N, mix expected utilities across precip groups using subjective
      probabilities (beliefs) π_g. Then convert CARA E[U] mixture to a CE:
        CE = -(1/r) * log(-E[U])   (valid since U is negative for CARA).
    - Also report the risk-neutral CE (which equals expected profit).

(9) Classify N as Risk-Increasing vs Risk-Reducing
    - For each N, compute Cov_g( g, ∂g/∂N ) across groups g using weights π_g.
      If Cov < 0 → "risk-reducing"; else "risk-increasing".
      Intuition: if higher-yield states have lower ∂g/∂N (or vice versa),
      N mitigates variability (reduces risk).

(10) Pick Optimal N (argmax CE)
    - For each CE curve (RN, CARA-Normal, CARA-MBME), report the N that maximizes it.

(11) Save Outputs
    - Write all intermediate/final tables to CSV for inspection.

Each process block below is marked with:  #### ---- name ---- ####
and heavily commented with intuition and (light) math where helpful.

"""



# ========== (0) SETUP & PATHS =================================================
#### ---- setup & paths ---- ####

# --- Standard Library & Typing ------------------------------------------------

from dataclasses import dataclass
# ^ Imports the @dataclass decorator from Python’s standard library.
#   We’ll decorate small "config" classes with @dataclass so Python auto-generates
#   an __init__ and nice repr() for us (cleaner, less boilerplate).

from typing import Dict, Tuple, Optional, Iterable
# ^ Type hints to make function signatures self-documenting:
#   - Dict[str, float], Tuple[...], Optional[T], Iterable[T], etc.
#   These don’t change runtime behavior—they just add clarity and help with IDE linting.

from pathlib import Path
# ^ Modern, object-oriented filesystem paths. Prefer Path over raw strings such as
#   "/Users/...". You can do (REPO_ROOT / "Data" / "file.csv") and it handles separators.

import sys, json, warnings
# ^ sys: access to interpreter settings (e.g., sys.path for module search path).
#   json: read/write JSON (we save beliefs_used.json).
#   warnings: emit or silence warnings (useful during modeling).

# --- Third-Party: Numerical & Data --------------------------------------------

import numpy as np
# ^ NumPy = foundation for numerical arrays, vectorized math, linear algebra, etc.
#   Conventional alias "np". We’ll use np.array, np.linspace, np.trapz, etc.

import pandas as pd
# ^ pandas = tabular data frames (like R data.frames). Conventional alias "pd".
#   We’ll use pd.read_csv, DataFrame, groupby, .to_csv, etc.

# --- Third-Party: Modeling Libraries ------------------------------------------

from pygam import LinearGAM, s, te
# ^ pyGAM provides Generalized Additive Models:
#   - LinearGAM: the model class for continuous outcomes.
#   - s(k): univariate smooth on the k-th feature (e.g., s(0) for "n").
#   - te(i, j): tensor-product smooth for interactions (e.g., te(n, prcp)).
#   We use GAM to flexibly estimate mean yield as a smooth function of inputs.

import statsmodels.api as sm
# ^ statsmodels = econometrics/statistics workhorse.
#   - sm.OLS(y, X).fit() for ordinary least squares (quadratic spec, fixed effects via dummies).
#   We keep the module alias "sm" so calls read cleanly: sm.OLS, sm.add_constant, etc.

import patsy
# ^ patsy builds design matrices from formulas (R-style), e.g.:
#   y, X = patsy.dmatrices("y ~ n + I(n**2) + n:prcp_t + ...", data=df)
#   It mirrors familiar R formula syntax and handles categorical encodings/dummies.

# --- Third-Party: Optimization / Typing ---------------------------------------

from scipy.optimize import root
# ^ Nonlinear equation solver. We use it to solve for maximum-entropy (MBME) density
#   parameters (λ1, λ2, λ3) such that the implied distribution matches target moments.

from numpy.typing import ArrayLike
# ^ Type hint meaning "anything that behaves like a NumPy array" (list, ndarray, etc.).
#   Makes signatures like profit(y: ArrayLike, ...) more informative for readers and IDEs.


# --- repo & IO paths (adjust here if your layout differs)
REPO_ROOT = Path("/Users/jaeseokhwang/dist_yield").resolve()
DATA_PATH = REPO_ROOT / "Data" / "Processed" / "Analysis_ready" / "df_il.csv"
OUT_DIR   = REPO_ROOT / "Results" / "Data"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ========== (0) CONFIGURATION =================================================
#### ---- configuration ---- ####
@dataclass
class CFG:
    """
    Economic parameters live here so you can tweak them in one place.
    - p_yield: output price applied to scaled yield y ∈ [0,1]
    - p_n    : nitrogen price per unit
    - cara_r : CARA risk aversion coefficient (r > 0)
    - crra_gamma: CRRA risk aversion (γ > 0, γ != 1 unless you want log utility)
    - support_points: resolution for numerical integration on [0,1]
    """
    p_yield: float = 5.0
    p_n: float = 0.8
    cara_r: float = 0.02
    crra_gamma: float = 2.0
    support_points: int = 600



# ========== (1) LOAD DATA =====================================================
#### ---- load data ---- ####
def load_data(path: Path) -> pd.DataFrame:
    """
    Loads a CSV with at least these columns:
      yield, n, prcp_t, gdd_t, edd_t, elev, slope, aspect, tpi,
      clay, sand, silt, water_storage, ffy_id
    Notes:
    - We rename `ffy_id` -> `field_id` to be explicit.
    """
    if not str(path).lower().endswith(".csv"):
        raise ValueError("Please export your RDS to CSV and pass a .csv path.")
    df = pd.read_csv(path)

    needed = ["yield","n","prcp_t","gdd_t","edd_t","elev","slope","aspect","tpi",
              "clay","sand","silt","water_storage","ffy_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=needed).copy()
    df.rename(columns={"ffy_id": "field_id"}, inplace=True)

    # Basic type coercions (paranoid-safe)
    for c in ["yield","n","prcp_t","gdd_t","edd_t","elev","slope","tpi",
              "clay","sand","silt","water_storage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["yield","n"])

    # Ensure field_id is str-like (for fixed effects)
    df["field_id"] = df["field_id"].astype(str)
    return df





# ========== (2) FEATURE ENGINEERING ==========================================
#### ---- feature engineering ---- ####
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Trim outliers:
       - Keep observations with yield in [mean±2σ] and n in [mean±1.5σ].
       - Economically: avoids extreme leverage points that distort fits.

    2) Scale yield to [0,1]:
       y_scaled = (y - min(y)) / (max(y) - min(y)).
       - Makes utility comparisons numerical stable, bounded support needed for MBME.

    3) Raw moments of scaled yield:
       y1 = y, y2 = y^2, y3 = y^3 on the scaled space.
       - Later we’ll need μ1, μ2, μ3 (moments) to build MBME densities.
    """
    df = df.copy()

    # --- Outlier rules (simple z-like rules on original scale)
    y_m, y_sd = df["yield"].mean(), df["yield"].std()
    n_m, n_sd = df["n"].mean(), df["n"].std()

    df = df.loc[
        (df["yield"].between(y_m - 2*y_sd, y_m + 2*y_sd)) &
        (df["n"].between(n_m - 1.5*n_sd, n_m + 1.5*n_sd))
    ].copy()

    # --- Scale yield to [0,1]
    y_min = df["yield"].min()
    y_rng = (df["yield"] - y_min).max()
    if y_rng <= 0:
        raise ValueError("Yield has zero range after filtering; cannot scale.")
    df["yield_scaled"] = (df["yield"] - y_min) / y_rng

    # --- Raw moments on scaled yield
    df["y1"] = df["yield_scaled"]
    df["y2"] = df["yield_scaled"]**2
    df["y3"] = df["yield_scaled"]**3

    return df


# ========== (3) TRAIN BASELINE MODELS ========================================
#### ---- model fitting ---- ####
def _design_quadratic(df: pd.DataFrame, yvar: str):
    """
    Quadratic + interactions; mirrors R-style:
      y ~ n + n^2 + prcp + gdd + edd + n:prcp + n:gdd + elev + slope + tpi + clay + sand + water_storage
    Interpretation:
      - n + n^2 allows diminishing returns.
      - Interactions n:prcp and n:gdd let N-response vary with weather.
      - Site covariates linearly shift the production function.
    """
    formula = (
        f"{yvar} ~ n + I(n**2) + prcp_t + gdd_t + edd_t + n:prcp_t + n:gdd_t "
        "+ elev + slope + tpi + clay + sand + water_storage"
    )
    # Patsy provides an Intercept by default.
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    return y.iloc[:, 0], X

def _design_fe(df: pd.DataFrame, yvar: str):
    """
    FE (Fixed Effects) design:
      y ~ n + n^2 + n:prcp + elev + slope + tpi + clay + sand + water_storage + C(field_id)
    Interpretation:
      - C(field_id) captures time-invariant field heterogeneity (soil, management).
      - OLS with FE is equivalent to including a dummy for each field_id.
    """
    formula = (
        f"{yvar} ~ n + I(n**2) + n:prcp_t + elev + slope + tpi + clay + sand + water_storage "
        "+ C(field_id)"
    )
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    return y.iloc[:, 0], X

def fit_gam_mu1(df: pd.DataFrame):
    """
    μ1 (mean) via GAM:
      g(n, prcp, gdd, edd) = s(n) + s(prcp) + te(n, prcp) + s(gdd) + s(edd) + linear(site vars)
    Intuition:
      - Smooth of n handles nonlinear response without picking a polynomial degree.
      - te(n, prcp) captures local interactions (e.g., N is more/less productive given precipitation).
    """
    X_smooth = df[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    X_lin = df[["elev","slope","tpi","clay","sand","water_storage"]].to_numpy()
    X = np.hstack([X_smooth, X_lin])
    gam = LinearGAM(s(0) + s(1) + te(0,1) + s(2) + s(3)).fit(X, df["yield_scaled"].to_numpy())
    return (gam, X_lin.shape[1])  # return (model, #linear cols) for shape context

def predict_gam_mu1(gam_and_k, df_like: pd.DataFrame) -> np.ndarray:
    gam, _ = gam_and_k
    X_smooth = df_like[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    X_lin = df_like[["elev","slope","tpi","clay","sand","water_storage"]].to_numpy()
    X = np.hstack([X_smooth, X_lin])
    return gam.predict(X)

def fit_quad(df: pd.DataFrame, yvar: str):
    y, X = _design_quadratic(df, yvar)
    mod = sm.OLS(y, X).fit()
    # Save Patsy design_info for robust prediction (exact same columns/order later)
    mod._design_info = getattr(X, "design_info", None)
    return mod

def fit_fe(df: pd.DataFrame, yvar: str):
    y, X = _design_fe(df, yvar)
    mod = sm.OLS(y, X).fit()
    mod._design_info = getattr(X, "design_info", None)
    return mod

def fit_raw_moment_gam(df: pd.DataFrame, yvar: str) -> LinearGAM:
    X = df[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    gam = LinearGAM(s(0) + s(1) + te(0,1) + s(2) + s(3)).fit(X, df[yvar].to_numpy())
    return gam

def predict_raw_gam(gam: LinearGAM, df_like: pd.DataFrame) -> np.ndarray:
    X = df_like[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    return gam.predict(X)

def _predict_with_model_design(model, new_df: pd.DataFrame) -> np.ndarray:
    """
    Use the fitted model's Patsy design_info to build X for new_df.
    This guarantees:
      - identical column set and order,
      - correct intercept naming (Intercept vs const),
      - presence/absence of FE dummies as seen during training.
    """
    design_info = getattr(model, "_design_info", None)
    if design_info is None:
        design_info = getattr(getattr(model, "model", None), "data", None)
        design_info = getattr(design_info, "design_info", None)
    if design_info is None:
        raise RuntimeError("Model missing design_info; refit using the fit_* functions in this file.")
    X_new = patsy.build_design_matrices([design_info], new_df, return_type="dataframe")[0]
    return model.predict(X_new)

def train_baselines(df: pd.DataFrame):
    """
    Fit central and raw-moment models:
      - μ1 central (GAM), residual moments y2_ct, y3_ct,
      - central models (GAM/Quad/FE) for y2_ct, y3_ct,
      - raw-moment models (GAM/Quad/FE) for y1,y2,y3.
    """
    models: Dict[str, object] = {}

    # μ1 (mean) central GAM
    gam_mu1_ct, _ = fit_gam_mu1(df)
    df["mu1_hat_ct_gam"] = predict_gam_mu1((gam_mu1_ct, 0), df)

    # Central residual moments around μ1_hat_ct_gam
    # y2_ct = (y - E[y])^2 ; y3_ct = (y - E[y])^3
    df["y2_ct"] = (df["yield_scaled"] - df["mu1_hat_ct_gam"])**2
    df["y3_ct"] = (df["yield_scaled"] - df["mu1_hat_ct_gam"])**3

    # Central GAMs for y2_ct, y3_ct
    gam_y2_ct = fit_raw_moment_gam(df, "y2_ct")
    gam_y3_ct = fit_raw_moment_gam(df, "y3_ct")

    # Quadratic central μ1, then residuals and Quadratic on those residual moments
    quad_mu1_ct = fit_quad(df, "yield_scaled")
    df["mu1_hat_ct_quad"] = _predict_with_model_design(quad_mu1_ct, df)
    df["y2_ct_quad_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_quad"])**2
    df["y3_ct_quad_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_quad"])**3
    quad_y2_ct = fit_quad(df, "y2_ct_quad_resid")
    quad_y3_ct = fit_quad(df, "y3_ct_quad_resid")

    # FE central μ1, then residuals and FE on those residual moments
    fe_mu1_ct = fit_fe(df, "yield_scaled")
    df["mu1_hat_ct_fe"] = _predict_with_model_design(fe_mu1_ct, df)
    df["y2_ct_fe_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_fe"])**2
    df["y3_ct_fe_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_fe"])**3
    fe_y2_ct = fit_fe(df, "y2_ct_fe_resid")
    fe_y3_ct = fit_fe(df, "y3_ct_fe_resid")

    # Raw-moment GAMs (y1,y2,y3)
    gam_y1_raw = fit_raw_moment_gam(df, "y1")
    gam_y2_raw = fit_raw_moment_gam(df, "y2")
    gam_y3_raw = fit_raw_moment_gam(df, "y3")

    # Raw-moment Quadratic
    quad_y1_raw = fit_quad(df, "y1")
    quad_y2_raw = fit_quad(df, "y2")
    quad_y3_raw = fit_quad(df, "y3")

    # Raw-moment FE
    fe_y1_raw = fit_fe(df, "y1")
    fe_y2_raw = fit_fe(df, "y2")
    fe_y3_raw = fit_fe(df, "y3")

    models.update(dict(
        # central mean + central moment models
        gam_mu1_ct=gam_mu1_ct,
        gam_y2_ct=gam_y2_ct, gam_y3_ct=gam_y3_ct,
        quad_mu1_ct=quad_mu1_ct, quad_y2_ct=quad_y2_ct, quad_y3_ct=quad_y3_ct,
        fe_mu1_ct=fe_mu1_ct, fe_y2_ct=fe_y2_ct, fe_y3_ct=fe_y3_ct,
        # raw-moment models
        gam_y1_raw=gam_y1_raw, gam_y2_raw=gam_y2_raw, gam_y3_raw=gam_y3_raw,
        quad_y1_raw=quad_y1_raw, quad_y2_raw=quad_y2_raw, quad_y3_raw=quad_y3_raw,
        fe_y1_raw=fe_y1_raw, fe_y2_raw=fe_y2_raw, fe_y3_raw=fe_y3_raw,
    ))
    return df, models


# ========== (4) BUILD PREDICTION GRID ========================================
#### ---- prediction grid ---- ####
def build_prediction_grid(df: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    """
    Grid over:
      - n spanning [min(n), max(n)] (n_points points)
      - prcp_t spanning [min, max] (n_points points)
    Other covariates (gdd_t, edd_t, elev, slope, tpi, clay, sand, water_storage) fixed at medians.
    Precipitation is binned into 5 quantile groups for belief-weighted mixtures.
    """
    n_seq = np.linspace(df["n"].min(), df["n"].max(), n_points)
    p_seq = np.linspace(df["prcp_t"].min(), df["prcp_t"].max(), n_points)
    grid = pd.DataFrame(np.array(np.meshgrid(n_seq, p_seq)).T.reshape(-1, 2), columns=["n","prcp_t"])

    # Hold other controls at medians (conditional predictions)
    med = df[["gdd_t","edd_t","elev","slope","tpi","clay","sand","water_storage"]].median()
    for c in med.index:
        grid[c] = med[c]

    # Precip bins
    q = np.quantile(grid["prcp_t"], np.linspace(0,1,6))
    labels = ["Very Low","Low","Moderate","High","Very High"]
    grid["prcp_group"] = pd.cut(grid["prcp_t"], bins=q, labels=labels, include_lowest=True)

    # For FE predictions, we need a field_id; pick the first
    grid["field_id"] = df["field_id"].iloc[0]
    return grid


# ========== (5) PREDICT ALL MOMENTS ON GRID ==================================
#### ---- grid predictions ---- ####
def predict_all(models: Dict[str, object], grid: pd.DataFrame) -> pd.DataFrame:
    """
    Predicts:
      - Central μ1 via GAM, and central μ2, μ3 via GAMs on residual moments.
      - Central μ1, μ2, μ3 via Quadratic and via FE (using model design_info).
      - Raw μ1, μ2, μ3 via GAM, Quadratic, FE.
    """
    out = grid.copy()

    # central GAM μ1 then central μ2, μ3
    out["mu1_hat_ct_gam"] = predict_gam_mu1((models["gam_mu1_ct"], 0), out)
    out["mu2_hat_ct_gam"] = predict_raw_gam(models["gam_y2_ct"], out)
    out["mu3_hat_ct_gam"] = predict_raw_gam(models["gam_y3_ct"], out)

    # central Quadratic
    out["mu1_hat_ct_quad"] = _predict_with_model_design(models["quad_mu1_ct"], out)
    out["mu2_hat_ct_quad"] = _predict_with_model_design(models["quad_y2_ct"],  out)
    out["mu3_hat_ct_quad"] = _predict_with_model_design(models["quad_y3_ct"],  out)

    # central FE
    out["mu1_hat_ct_fe"] = _predict_with_model_design(models["fe_mu1_ct"], out)
    out["mu2_hat_ct_fe"] = _predict_with_model_design(models["fe_y2_ct"],  out)
    out["mu3_hat_ct_fe"] = _predict_with_model_design(models["fe_y3_ct"],  out)

    # raw GAM
    out["mu1_hat_raw_gam"] = predict_raw_gam(models["gam_y1_raw"], out)
    out["mu2_hat_raw_gam"] = predict_raw_gam(models["gam_y2_raw"], out)
    out["mu3_hat_raw_gam"] = predict_raw_gam(models["gam_y3_raw"], out)

    # raw Quadratic
    out["mu1_hat_raw_quad"] = _predict_with_model_design(models["quad_y1_raw"], out)
    out["mu2_hat_raw_quad"] = _predict_with_model_design(models["quad_y2_raw"], out)
    out["mu3_hat_raw_quad"] = _predict_with_model_design(models["quad_y3_raw"], out)

    # raw FE
    out["mu1_hat_raw_fe"] = _predict_with_model_design(models["fe_y1_raw"], out)
    out["mu2_hat_raw_fe"] = _predict_with_model_design(models["fe_y2_raw"], out)
    out["mu3_hat_raw_fe"] = _predict_with_model_design(models["fe_y3_raw"], out)

    return out


# ========== (6) MBME DENSITIES ==============================================
#### ---- MBME densities ---- ####
def _me_density_lambdas(mu1: float, mu2: float, mu3: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find λ = (λ1, λ2, λ3) such that:
      ∫ y f(y) dy = μ1, ∫ y^2 f(y) dy = μ2, ∫ y^3 f(y) dy = μ3
    with:
      f(y) = exp(λ1*y + λ2*y^2 + λ3*y^3) / Z(λ),  y ∈ [0,1]
      Z(λ) = ∫ exp(λ1*y + λ2*y^2 + λ3*y^3) dy
    """
    y1, y2_pow, y3_pow = y, y**2, y**3

    def system(lmb):
        l1, l2, l3 = lmb
        expo = l1*y1 + l2*y2_pow + l3*y3_pow
        expo -= expo.max()  # numerical stability
        num = np.exp(expo)
        Z = np.trapz(num, y)
        f = num / Z
        m1 = np.trapz(y1 * f, y)
        m2 = np.trapz(y2_pow * f, y)
        m3 = np.trapz(y3_pow * f, y)
        return np.array([m1 - mu1, m2 - mu2, m3 - mu3])

    sol = root(system, x0=np.array([0.0, 0.0, 0.0]), method="hybr")
    if not sol.success:
        raise RuntimeError(f"ME solver failed: {sol.message}")

    lmb = sol.x
    expo = lmb[0]*y1 + lmb[1]*y2_pow + lmb[2]*y3_pow
    expo -= expo.max()
    num = np.exp(expo)
    Z = np.trapz(num, y)
    f = num / Z
    return lmb, f

def mbme_density_for_row(mu1, mu2, mu3, cfg: CFG):
    """
    Return (y_grid, f) on [0,1] if moments are valid; else None.
    Validity:
      - 0 <= μ1 <= 1
      - μ2 > μ1^2 (positive variance)
      - We accept any μ3 (skew) and rely on solver robustness.
    """
    if not (0 <= mu1 <= 1):
        return None
    if mu2 <= mu1**2:
        return None
    y = np.linspace(0, 1, cfg.support_points)
    try:
        _, f = _me_density_lambdas(mu1, mu2, mu3, y)
        return y, f
    except Exception:
        return None

def estimate_mbme_for_grid(pred_grid: pd.DataFrame, family: str, cfg: CFG) -> pd.DataFrame:
    """
    family ∈ {'gam','quad','fe'} uses mu{1,2,3}_hat_raw_{family}.
    Returns long table: [n, prcp_group, model, y, density]
    """
    rows = []
    fam = family.lower()
    for _, r in pred_grid.iterrows():
        mu1 = r[f"mu1_hat_raw_{fam}"]
        mu2 = r[f"mu2_hat_raw_{fam}"]
        mu3 = r[f"mu3_hat_raw_{fam}"]
        y_f = mbme_density_for_row(mu1, mu2, mu3, cfg)
        if y_f is None:
            continue
        y, f = y_f
        rows.append(pd.DataFrame({
            "n": r["n"], "prcp_group": r["prcp_group"], "model": fam.upper(),
            "y": y, "density": f
        }))
    if not rows:
        return pd.DataFrame(columns=["n","prcp_group","model","y","density"])
    return pd.concat(rows, ignore_index=True)


# ========== (7) UTILITIES =====================================================
#### ---- utilities ---- ####
def profit(y: ArrayLike, n: float, cfg: CFG) -> np.ndarray:
    """
    π(y, n) = p_yield * y - p_n * n
    Economics: linear revenue in realized yield y, linear cost in chosen N.
    """
    return cfg.p_yield * np.asarray(y) - cfg.p_n * n

def cara_u(pi: ArrayLike, r: float) -> np.ndarray:
    """CARA: U(π) = -exp(-r * π)."""
    return -np.exp(-r * np.asarray(pi))

def crra_u(pi: ArrayLike, gamma: float) -> np.ndarray:
    """
    CRRA: U(π) = (π^(1-γ) - 1) / (1-γ), γ ≠ 1;  U(π) = log(π) for γ = 1.
    Domain: π > 0 (so we clip to a small positive value for safety).
    """
    pi = np.maximum(np.asarray(pi), 1e-9)
    if np.isclose(gamma, 1.0):
        return np.log(pi)
    return (pi**(1.0 - gamma) - 1.0) / (1.0 - gamma)

def expected_u_mbme(y: np.ndarray, f: np.ndarray, n: float, cfg: CFG, kind: str = "CARA") -> float:
    """
    E[U] = ∫ U(π(y,n)) f(y) dy, where f is MBME on [0,1].
    """
    pi = profit(y, n, cfg)
    if kind.upper() == "CARA":
        u = cara_u(pi, cfg.cara_r)
    else:
        u = crra_u(pi, cfg.crra_gamma)
    return np.trapz(u * f, y)

def expected_u_normal(mu1: float, var: float, n: float, cfg: CFG, kind: str = "CARA") -> float:
    """
    Normal approximation for y:
      y ~ N(μ1, var) (we truncate to [0,1] numerically for CRRA)
    CARA closed form:
      Let a = p_yield, b = -p_n * n, then π = a*y + b.
      E[ -exp(-r π) ] = -exp( -r(a μ1 + b) + 0.5 r^2 a^2 var ).
    CRRA: integrate numerically (truncate to [0,1]).
    """
    a = cfg.p_yield
    b = -cfg.p_n * n
    if kind.upper() == "CARA":
        return -np.exp(-cfg.cara_r * (a*mu1 + b) + 0.5*(cfg.cara_r**2)*(a**2)*var)
    # CRRA numerical integration under truncated normal
    from scipy.stats import norm
    y = np.linspace(0, 1, cfg.support_points)
    f = norm.pdf(y, loc=mu1, scale=np.sqrt(max(var, 1e-9)))
    f /= np.trapz(f, y)  # renormalize on [0,1]
    return expected_u_mbme(y, f, n, cfg, kind="CRRA")

def compute_utilities(pred_grid: pd.DataFrame, mbme_long: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """
    For family in {GAM, FE}:
      - Risk-neutral utility (expected profit): util_rn = p_yield*μ1 - p_n*n.
      - Risk-averse CARA/CRRA under Normal approx (use μ1, Var = μ2 - μ1^2).
      - Risk-averse CARA/CRRA under MBME (integral w.r.t. MBME density).
    Output columns per family:
      util_rn_{fam}, util_ra_norm_{fam}_CARA, util_ra_norm_{fam}_CRRA,
      util_ra_mbme_{fam}_CARA, util_ra_mbme_{fam}_CRRA
    """
    out = pred_grid.copy()

    for fam in ["gam", "fe"]:
        mu1 = out[f"mu1_hat_raw_{fam}"]  # raw moment means
        mu2 = out[f"mu2_hat_raw_{fam}"]
        var = (mu2 - mu1**2).clip(lower=1e-9)

        # Risk-neutral (linear utility) equals expected profit
        out[f"util_rn_{fam}"] = cfg.p_yield * mu1 - cfg.p_n * out["n"]

        # Normal approximation E[U]
        out[f"util_ra_norm_{fam}_CARA"] = [
            expected_u_normal(m1, v, n, cfg, "CARA")
            for m1, v, n in zip(mu1, var, out["n"])
        ]
        out[f"util_ra_norm_{fam}_CRRA"] = [
            expected_u_normal(m1, v, n, cfg, "CRRA")
            for m1, v, n in zip(mu1, var, out["n"])
        ]

        # MBME-based E[U]
        key = (mbme_long["model"] == fam.upper())
        by_key = mbme_long.loc[key].groupby(["n","prcp_group"])
        mbme_map = {k: v for k, v in by_key}

        mbme_cara, mbme_crra = [], []
        for _, row in out.iterrows():
            k = (row["n"], row["prcp_group"])
            if k not in mbme_map:
                mbme_cara.append(np.nan)
                mbme_crra.append(np.nan)
                continue
            block = mbme_map[k]
            y, f = block["y"].to_numpy(), block["density"].to_numpy()
            mbme_cara.append(expected_u_mbme(y, f, float(row["n"]), cfg, "CARA"))
            mbme_crra.append(expected_u_mbme(y, f, float(row["n"]), cfg, "CRRA"))
        out[f"util_ra_mbme_{fam}_CARA"] = mbme_cara
        out[f"util_ra_mbme_{fam}_CRRA"] = mbme_crra

    return out


# ========== (8) BELIEF-WEIGHTED CE ===========================================
#### ---- CE aggregation with beliefs ---- ####
def _normalize_beliefs(beliefs: Optional[Dict[str, float]], groups: Iterable[str]) -> Dict[str, float]:
    """
    Normalize a dict of beliefs π_g so that ∑_g π_g = 1 over the present groups.
    If beliefs is None, use uniform weights.
    """
    groups = [str(g) for g in groups]
    if beliefs is None:
        return {g: 1.0/len(groups) for g in groups}
    w = {str(k): float(v) for k, v in beliefs.items() if str(k) in groups}
    tot = sum(w.values())
    if tot <= 0:
        return {g: 1.0/len(groups) for g in groups}
    w = {k: v/tot for k, v in w.items()}
    for g in groups:
        w.setdefault(g, 0.0)
    return w

def aggregate_ce_with_beliefs(util_grid: pd.DataFrame, family: str, cfg: CFG,
                              beliefs: Optional[Dict[str, float]]=None) -> pd.DataFrame:
    """
    For each N, mix expected utilities across precipitation groups using π_g
    and convert E[U] (CARA) to a certainty equivalent:
      CE = -(1/r) * log(-E[U]).
    Also report risk-neutral CE (which equals expected profit).
    Returns columns: n, ce_rn, ce_cara_norm, ce_cara_mbme, eu_norm_mix, eu_mbme_mix
    """
    fam = family.lower()
    gcol = "prcp_group"
    df = util_grid.copy()
    df[gcol] = df[gcol].astype(str)
    ug = df[gcol].unique().tolist()
    w = _normalize_beliefs(beliefs, ug)

    rows = []
    for n, block in df.groupby("n", as_index=False):
        rn = 0.0
        eu_norm = 0.0
        eu_mbme = 0.0
        for g, b in w.items():
            sub = block.loc[block[gcol] == g]
            if sub.empty:
                continue
            rn      += b * sub[f"util_rn_{fam}"].iloc[0]
            eu_norm += b * sub[f"util_ra_norm_{fam}_CARA"].iloc[0]
            eu_mbme += b * sub[f"util_ra_mbme_{fam}_CARA"].iloc[0]

        # Convert CARA E[U] to CE: CE = -(1/r) log(-E[U]); note E[U] < 0 for CARA
        ce_norm = (-1.0/cfg.cara_r) * np.log(-eu_norm) if eu_norm < 0 else np.nan
        ce_mbme = (-1.0/cfg.cara_r) * np.log(-eu_mbme) if eu_mbme < 0 else np.nan

        rows.append(dict(
            n=float(n), ce_rn=rn, ce_cara_norm=ce_norm, ce_cara_mbme=ce_mbme,
            eu_norm_mix=eu_norm, eu_mbme_mix=eu_mbme
        ))
    return pd.DataFrame(rows).sort_values("n").reset_index(drop=True)


# ========== (9) RISK CLASSIFICATION ==========================================
#### ---- risk classification ---- ####
def classify_input_risk(grid_pred: pd.DataFrame, beliefs: Optional[Dict[str, float]]=None,
                        use_col: str = "mu1_hat_ct_gam") -> pd.DataFrame:
    """
    Classify N as risk-increasing vs risk-reducing via the sign of:
      Cov_g( g, ∂g/∂N ), weighted by beliefs π_g.
    Numerical details:
      - For each precip group, compute ∂g/∂N by gradient over the n-grid.
      - Then weighted covariance across groups at each N.
      - If Cov < 0 → risk-reducing; else risk-increasing.
    """
    gcol = "prcp_group"
    df = grid_pred[["n", gcol, use_col]].copy()
    df[gcol] = df[gcol].astype(str)

    # ∂g/∂N within each group using np.gradient
    d_list = []
    for g, grp in df.groupby(gcol):
        grp = grp.sort_values("n")
        g_vals = grp[use_col].to_numpy()
        n_vals = grp["n"].to_numpy()
        if len(n_vals) >= 3 and np.all(np.diff(n_vals) > 0):
            gn = np.gradient(g_vals, n_vals)
        else:
            # fallback one-sided diff
            gn = np.append(np.diff(g_vals)/np.diff(n_vals), np.nan)
        tmp = grp.copy()
        tmp["g_N"] = gn
        d_list.append(tmp)
    ddf = pd.concat(d_list, ignore_index=True).dropna(subset=["g_N"])

    # Belief weights
    ug = ddf[gcol].unique().tolist()
    w = _normalize_beliefs(beliefs, ug)

    # Weighted covariance across groups for each N
    out_rows = []
    for n, block in ddf.groupby("n"):
        w_here = np.array([w[g] for g in block[gcol]])
        w_here = w_here / w_here.sum() if w_here.sum() > 0 else np.ones(len(block))/len(block)
        g_vals  = block[use_col].to_numpy()
        gn_vals = block["g_N"].to_numpy()
        eg  = np.sum(w_here * g_vals)
        egn = np.sum(w_here * gn_vals)
        cov = np.sum(w_here * (g_vals - eg) * (gn_vals - egn))
        label = "risk-reducing (Cov<0)" if cov < 0 else "risk-increasing (Cov>0)"
        out_rows.append(dict(n=float(n), cov_g_gN=cov, classification=label))
    return pd.DataFrame(out_rows).sort_values("n").reset_index(drop=True)


# ========== (10) ARGMAX N =====================================================
#### ---- argmax N ---- ####
def argmax_n(ce_df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """
    For each metric column in `cols`, find the row that maximizes it and
    return (metric, n_star, value).
    """
    rows = []
    for c in cols:
        sub = ce_df.dropna(subset=[c]).sort_values(c, ascending=False)
        if sub.empty:
            rows.append(dict(metric=c, n_star=np.nan, value=np.nan))
        else:
            top = sub.iloc[0]
            rows.append(dict(metric=c, n_star=float(top["n"]), value=float(top[c])))
    return pd.DataFrame(rows)


# ========== (11) MAIN SCRIPT ==================================================
#### ---- main driver ---- ####
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(">>> (1) Loading data")
    df = load_data(DATA_PATH)

    print(">>> (2) Feature engineering")
    df = make_features(df)

    print(">>> (3) Training baseline models (GAM, Quadratic, FE)")
    df_fit, models = train_baselines(df)

    print(">>> (4) Building prediction grid")
    grid = build_prediction_grid(df_fit, n_points=50)

    print(">>> (5) Predicting all moments on the grid")
    grid_pred = predict_all(models, grid)

    print(">>> (6) MBME densities for GAM and FE")
    cfg = CFG()  # tweak parameters here if needed
    mbme_gam = estimate_mbme_for_grid(grid_pred, family="gam", cfg=cfg)
    mbme_fe  = estimate_mbme_for_grid(grid_pred, family="fe",  cfg=cfg)

    print(">>> (7) Computing utilities (RN, CARA/CRRA under Normal & MBME)")
    util_grid = compute_utilities(grid_pred, pd.concat([mbme_gam, mbme_fe], ignore_index=True), cfg)

    print(">>> (8) Belief-weighted CE aggregation")
    BELIEFS = {
        "Very Low": 0.2,
        "Low": 0.2,
        "Moderate": 0.2,
        "High": 0.2,
        "Very High": 0.2
    }
    ce_mix_gam = aggregate_ce_with_beliefs(util_grid, "gam", cfg, BELIEFS)
    ce_mix_fe  = aggregate_ce_with_beliefs(util_grid, "fe",  cfg, BELIEFS)

    print(">>> (9) Risk classification via Cov(g, ∂g/∂N)")
    risk_class = classify_input_risk(grid_pred, BELIEFS, use_col="mu1_hat_ct_gam")

    print(">>> (10) Optimal N (argmax CE)")
    opt_gam = argmax_n(ce_mix_gam, ["ce_rn","ce_cara_norm","ce_cara_mbme"])
    opt_fe  = argmax_n(ce_mix_fe,  ["ce_rn","ce_cara_norm","ce_cara_mbme"])

    print(">>> (11) Saving outputs to:", OUT_DIR)
    df_fit.to_csv(OUT_DIR / "dat_com_mutate.csv", index=False)
    grid_pred.to_csv(OUT_DIR / "moments_grid_raw.csv", index=False)
    mbme_gam.to_csv(OUT_DIR / "mbme_density_gam.csv", index=False)
    mbme_fe.to_csv(OUT_DIR / "mbme_density_fe.csv", index=False)
    util_grid.to_csv(OUT_DIR / "utility_grid_compare.csv", index=False)
    ce_mix_gam.to_csv(OUT_DIR / "ce_mixture_gam.csv", index=False)
    ce_mix_fe.to_csv(OUT_DIR / "ce_mixture_fe.csv", index=False)
    risk_class.to_csv(OUT_DIR / "risk_classification.csv", index=False)
    opt_gam.to_csv(OUT_DIR / "nstar_gam.csv", index=False)
    opt_fe.to_csv(OUT_DIR / "nstar_fe.csv", index=False)
    with open(OUT_DIR / "beliefs_used.json", "w") as f:
        json.dump(BELIEFS, f, indent=2)

    print(">>> Done.")
