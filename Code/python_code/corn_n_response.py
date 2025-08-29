# -*- coding: utf-8 -*-
"""
Core routines for corn N-response analysis aligned to the latest Rmd + belief-weighted CE and risk classification.
Pipeline:
  1) load_data -> make_features (normalize + raw moments)
  2) fit models: GAM, Quadratic OLS, FE (field_id) for μ1 and central moments
  3) fit raw-moment models for μ1, μ2, μ3
  4) build prediction grid and predict all moments
  5) construct MBME densities from raw moments on [0,1]
  6) compute utilities at (n, precip-bin) level
  7) aggregate certainty equivalents (CE) across precip bins with subjective beliefs π
  8) classify N as risk-increasing vs risk-reducing via Cov(g, g_N)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

# Modeling libs
from pygam import LinearGAM, s, te
import statsmodels.api as sm
import patsy

# Optimization / integration
from scipy.optimize import root
from numpy.typing import ArrayLike

# ----------------------------- Config ----------------------------- #

@dataclass
class CFG:
    p_yield: float = 5.0     # price per unit yield (for SCALED yield in [0,1])
    p_n: float = 0.8         # price per unit nitrogen
    cara_r: float = 0.02     # CARA risk aversion
    crra_gamma: float = 2.0  # CRRA coefficient (>0, !=1)
    support_points: int = 500  # grid for [0,1] support in ME/expectations

# ------------------------ IO + preprocessing ---------------------- #

def load_data(path: str) -> pd.DataFrame:
    """
    Expect a CSV (RDS exported to CSV) with columns:
      yield, n, prcp_t, gdd_t, edd_t, elev, slope, aspect, tpi,
      clay, sand, silt, water_storage, ffy_id
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Please export your RDS to CSV and pass the .csv path.")
    needed = ["yield","n","prcp_t","gdd_t","edd_t","elev","slope","aspect","tpi",
              "clay","sand","silt","water_storage","ffy_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=needed).copy()
    df.rename(columns={"ffy_id":"field_id"}, inplace=True)
    return df

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize yield to [0,1], trim outliers (as in Rmd), compute raw moments."""
    df = df.copy()

    # Outlier thresholds (based on original scale)
    y_mean, y_sd = df["yield"].mean(), df["yield"].std()
    n_mean, n_sd = df["n"].mean(), df["n"].std()

    df = df.loc[
        (df["yield"].between(y_mean - 2*y_sd, y_mean + 2*y_sd)) &
        (df["n"].between(n_mean - 1.5*n_sd, n_mean + 1.5*n_sd))
    ].copy()

    y_min = df["yield"].min()
    y_rng = (df["yield"] - y_min).max()
    if y_rng <= 0:
        raise ValueError("Yield has zero range after filtering.")
    df["yield_scaled"] = (df["yield"] - y_min) / y_rng

    # Raw moments of scaled yield
    df["y1"] = df["yield_scaled"]
    df["y2"] = df["yield_scaled"]**2
    df["y3"] = df["yield_scaled"]**3

    return df

# ------------------------- Model fitting -------------------------- #

def _design_quadratic(df: pd.DataFrame, yvar: str):
    """Quadratic + selected covariates + N×P and N×GDD interactions (as in Rmd)."""
    formula = (
        f"{yvar} ~ n + I(n**2) + prcp_t + gdd_t + edd_t + n:prcp_t + n:gdd_t "
        "+ elev + slope + tpi + clay + sand + water_storage"
    )
    # LHS present only to define the same RHS; Patsy includes an Intercept by default
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    return y.iloc[:, 0], X  # DO NOT add a constant here

def _design_fe(df: pd.DataFrame, yvar: str):
    """FE via dummy absorption with C(field_id)."""
    formula = (
        f"{yvar} ~ n + I(n**2) + n:prcp_t + elev + slope + tpi + clay + sand + water_storage "
        "+ C(field_id)"
    )
    y, X = patsy.dmatrices(formula, df, return_type="dataframe")
    return y.iloc[:, 0], X  # DO NOT add a constant here

def fit_gam_mu1(df: pd.DataFrame) -> LinearGAM:
    """μ1 (mean) GAM: s(n) + s(prcp_t) + te(n, prcp_t) + s(gdd_t) + s(edd_t)."""
    X_smooth = df[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    X_lin = df[["elev","slope","tpi","clay","sand","water_storage"]].to_numpy()
    X = np.hstack([X_smooth, X_lin])
    gam = LinearGAM( s(0) + s(1) + te(0,1) + s(2) + s(3) )
    gam = gam.fit(X, df["yield_scaled"].to_numpy())
    return (gam, X_lin.shape[1])

def predict_gam_mu1(gam_and_k, df_like: pd.DataFrame) -> np.ndarray:
    gam, k_lin = gam_and_k
    X_smooth = df_like[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    X_lin = df_like[["elev","slope","tpi","clay","sand","water_storage"]].to_numpy()
    X = np.hstack([X_smooth, X_lin])
    return gam.predict(X)

def fit_quad(df: pd.DataFrame, yvar: str):
    y, X = _design_quadratic(df, yvar)
    mod = sm.OLS(y, X).fit()
    # Save Patsy design_info for bullet-proof prediction later
    mod._design_info = getattr(X, "design_info", None)
    return mod

def fit_fe(df: pd.DataFrame, yvar: str):
    y, X = _design_fe(df, yvar)
    mod = sm.OLS(y, X).fit()
    mod._design_info = getattr(X, "design_info", None)
    return mod

def fit_raw_moment_gam(df: pd.DataFrame, yvar: str) -> LinearGAM:
    """GAM for raw moments: y1, y2, y3 using same smooth structure."""
    X = df[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    gam = LinearGAM( s(0) + s(1) + te(0,1) + s(2) + s(3) )
    gam = gam.fit(X, df[yvar].to_numpy())
    return gam

def predict_raw_gam(gam: LinearGAM, df_like: pd.DataFrame) -> np.ndarray:
    X = df_like[["n","prcp_t","gdd_t","edd_t"]].to_numpy()
    return gam.predict(X)

# --- Predict using the fitted model's Patsy design_info (bullet-proof)
def _predict_with_model_design(model, new_df: pd.DataFrame) -> np.ndarray:
    """
    Rebuild the exogenous matrix for `new_df` using the fitted model's
    Patsy design_info, guaranteeing identical columns/order (incl. FE dummies
    and intercept naming). This avoids needing the LHS target in new_df.
    """
    design_info = getattr(model, "_design_info", None)
    if design_info is None:
        # Fallback: try statsmodels' stored design_info (when available)
        design_info = getattr(getattr(model, "model", None), "data", None)
        design_info = getattr(design_info, "design_info", None)
    if design_info is None:
        raise RuntimeError("Model is missing Patsy design_info; refit using fit_quad/fit_fe in this module.")

    X_new = patsy.build_design_matrices([design_info], new_df, return_type="dataframe")[0]
    return model.predict(X_new)

# ------------------------ Prediction grid ------------------------- #

def build_prediction_grid(df: pd.DataFrame, n_points: int = 50) -> pd.DataFrame:
    n_seq = np.linspace(df["n"].min(), df["n"].max(), n_points)
    p_seq = np.linspace(df["prcp_t"].min(), df["prcp_t"].max(), n_points)
    grid = pd.DataFrame(np.array(np.meshgrid(n_seq, p_seq)).T.reshape(-1, 2), columns=["n","prcp_t"])

    # fix other covariates at median (Rmd style)
    med = df[["gdd_t","edd_t","elev","slope","tpi","clay","sand","water_storage"]].median()
    for c in med.index:
        grid[c] = med[c]
    # precip groups (quantiles over grid prcp_t)
    q = np.quantile(grid["prcp_t"], np.linspace(0,1,6))
    labels = ["Very Low","Low","Moderate","High","Very High"]
    grid["prcp_group"] = pd.cut(grid["prcp_t"], bins=q, labels=labels, include_lowest=True)
    # carry a field_id for FE prediction (use the first)
    grid["field_id"] = df["field_id"].iloc[0]
    return grid

# -------------------- MBME (Maximum Entropy) ---------------------- #

def _me_density_lambdas(mu1: float, mu2: float, mu3: float, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve for λ such that:
      ∑ f(y)*y = μ1, ∑ f(y)*y^2 = μ2, ∑ f(y)*y^3 = μ3, with f(y)=exp(λ1*y+λ2*y^2+λ3*y^3)/Z over [0,1]
    """
    y1, y2_pow, y3_pow = y, y**2, y**3

    def system(lmb):
        l1,l2,l3 = lmb
        expo = l1*y1 + l2*y2_pow + l3*y3_pow
        expo -= expo.max()
        num = np.exp(expo)
        Z = np.trapz(num, y)
        f = num / Z
        m1 = np.trapz(y1 * f, y)
        m2 = np.trapz(y2_pow * f, y)
        m3 = np.trapz(y3_pow * f, y)
        return np.array([m1 - mu1, m2 - mu2, m3 - mu3])

    sol = root(system, x0=np.array([0.0, 0.0, 0.0]), method='hybr')
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
    """Return (y_grid, density) on [0,1] or None if invalid moments."""
    if not (0 <= mu1 <= 1): return None
    if mu2 <= mu1**2: return None  # needs positive variance
    y = np.linspace(0, 1, cfg.support_points)
    try:
        _, f = _me_density_lambdas(mu1, mu2, mu3, y)
        return y, f
    except Exception:
        return None

# ----------------------- Utility functions ------------------------ #

def profit(y: ArrayLike, n: float, cfg: CFG) -> np.ndarray:
    return cfg.p_yield * np.asarray(y) - cfg.p_n * n

def cara_u(pi: ArrayLike, r: float) -> np.ndarray:
    return -np.exp(-r * np.asarray(pi))

def crra_u(pi: ArrayLike, gamma: float) -> np.ndarray:
    pi = np.maximum(np.asarray(pi), 1e-9)  # clip for domain
    if np.isclose(gamma, 1.0):
        return np.log(pi)
    return (pi**(1.0 - gamma) - 1.0) / (1.0 - gamma)

def expected_u_mbme(y: np.ndarray, f: np.ndarray, n: float, cfg: CFG, kind: str = "CARA") -> float:
    pi = profit(y, n, cfg)
    if kind.upper() == "CARA":
        u = cara_u(pi, cfg.cara_r)
    else:
        u = crra_u(pi, cfg.crra_gamma)
    return np.trapz(u * f, y)

def expected_u_normal(mu1: float, var: float, n: float, cfg: CFG, kind: str = "CARA") -> float:
    """
    CARA has closed form for normal if profit linear in y:
      E[-exp(-r(a*y + b))] = -exp(-r(a*mu + b) + 0.5*r^2*a^2*var)
    Here a = p_yield, b = -p_n*n.
    For CRRA, we integrate numerically on [0,1].
    """
    a = cfg.p_yield
    b = -cfg.p_n * n
    if kind.upper() == "CARA":
        return -np.exp(-cfg.cara_r * (a*mu1 + b) + 0.5*(cfg.cara_r**2)*(a**2)*var)
    # CRRA numerical on truncated normal:
    y = np.linspace(0, 1, cfg.support_points)
    from scipy.stats import norm
    f = norm.pdf(y, loc=mu1, scale=np.sqrt(max(var, 1e-9)))
    f /= np.trapz(f, y)
    return expected_u_mbme(y, f, n, cfg, kind="CRRA")

# --------------------------- Orchestration ------------------------ #

def train_baselines(df: pd.DataFrame):
    """
    Fit:
      - μ1 GAM (central)
      - central residual moments y2_ct, y3_ct from μ1_hat_ct
      - central GAM/Quad/FE for y2_ct & y3_ct
      - raw GAM/Quad/FE for y1,y2,y3
    """
    models: Dict[str, object] = {}

    # μ1 central GAM
    gam_mu1_ct, _ = fit_gam_mu1(df)
    df["mu1_hat_ct_gam"] = predict_gam_mu1((gam_mu1_ct, 0), df)
    df["y2_ct"] = (df["yield_scaled"] - df["mu1_hat_ct_gam"])**2
    df["y3_ct"] = (df["yield_scaled"] - df["mu1_hat_ct_gam"])**3

    # Central models (GAM for y2_ct, y3_ct)
    gam_y2_ct = fit_raw_moment_gam(df, "y2_ct")
    gam_y3_ct = fit_raw_moment_gam(df, "y3_ct")

    # Quadratic (central)
    quad_mu1_ct = fit_quad(df, "yield_scaled")
    df["mu1_hat_ct_quad"] = _predict_with_model_design(quad_mu1_ct, df)
    df["y2_ct_quad_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_quad"])**2
    df["y3_ct_quad_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_quad"])**3
    quad_y2_ct = fit_quad(df, "y2_ct_quad_resid")
    quad_y3_ct = fit_quad(df, "y3_ct_quad_resid")

    # FE (central)
    fe_mu1_ct = fit_fe(df, "yield_scaled")
    df["mu1_hat_ct_fe"] = _predict_with_model_design(fe_mu1_ct, df)
    df["y2_ct_fe_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_fe"])**2
    df["y3_ct_fe_resid"] = (df["yield_scaled"] - df["mu1_hat_ct_fe"])**3
    fe_y2_ct = fit_fe(df, "y2_ct_fe_resid")
    fe_y3_ct = fit_fe(df, "y3_ct_fe_resid")

    # Raw-moment models
    gam_y1_raw = fit_raw_moment_gam(df, "y1")
    gam_y2_raw = fit_raw_moment_gam(df, "y2")
    gam_y3_raw = fit_raw_moment_gam(df, "y3")

    quad_y1_raw = fit_quad(df, "y1")
    quad_y2_raw = fit_quad(df, "y2")
    quad_y3_raw = fit_quad(df, "y3")

    fe_y1_raw = fit_fe(df, "y1")
    fe_y2_raw = fit_fe(df, "y2")
    fe_y3_raw = fit_fe(df, "y3")

    models.update(dict(
        gam_mu1_ct=gam_mu1_ct,
        gam_y2_ct=gam_y2_ct, gam_y3_ct=gam_y3_ct,
        quad_mu1_ct=quad_mu1_ct, quad_y2_ct=quad_y2_ct, quad_y3_ct=quad_y3_ct,
        fe_mu1_ct=fe_mu1_ct, fe_y2_ct=fe_y2_ct, fe_y3_ct=fe_y3_ct,
        gam_y1_raw=gam_y1_raw, gam_y2_raw=gam_y2_raw, gam_y3_raw=gam_y3_raw,
        quad_y1_raw=quad_y1_raw, quad_y2_raw=quad_y2_raw, quad_y3_raw=quad_y3_raw,
        fe_y1_raw=fe_y1_raw, fe_y2_raw=fe_y2_raw, fe_y3_raw=fe_y3_raw,
    ))
    return df, models

def predict_all(models: Dict[str, object], grid: pd.DataFrame) -> pd.DataFrame:
    """Predict central and raw moments on the grid for all three model families."""
    out = grid.copy()

    # central GAM μ1 then central y2,y3
    out["mu1_hat_ct_gam"] = predict_gam_mu1((models["gam_mu1_ct"], 0), out)
    out["mu2_hat_ct_gam"] = predict_raw_gam(models["gam_y2_ct"], out)
    out["mu3_hat_ct_gam"] = predict_raw_gam(models["gam_y3_ct"], out)

    # central Quad (use saved design_info)
    out["mu1_hat_ct_quad"] = _predict_with_model_design(models["quad_mu1_ct"], out)
    out["mu2_hat_ct_quad"] = _predict_with_model_design(models["quad_y2_ct"],  out)
    out["mu3_hat_ct_quad"] = _predict_with_model_design(models["quad_y3_ct"],  out)

    # central FE (use saved design_info)
    out["mu1_hat_ct_fe"] = _predict_with_model_design(models["fe_mu1_ct"], out)
    out["mu2_hat_ct_fe"] = _predict_with_model_design(models["fe_y2_ct"],  out)
    out["mu3_hat_ct_fe"] = _predict_with_model_design(models["fe_y3_ct"],  out)

    # raw GAM
    out["mu1_hat_raw_gam"] = predict_raw_gam(models["gam_y1_raw"], out)
    out["mu2_hat_raw_gam"] = predict_raw_gam(models["gam_y2_raw"], out)
    out["mu3_hat_raw_gam"] = predict_raw_gam(models["gam_y3_raw"], out)

    # raw Quad
    out["mu1_hat_raw_quad"] = _predict_with_model_design(models["quad_y1_raw"], out)
    out["mu2_hat_raw_quad"] = _predict_with_model_design(models["quad_y2_raw"], out)
    out["mu3_hat_raw_quad"] = _predict_with_model_design(models["quad_y3_raw"], out)

    # raw FE
    out["mu1_hat_raw_fe"] = _predict_with_model_design(models["fe_y1_raw"], out)
    out["mu2_hat_raw_fe"] = _predict_with_model_design(models["fe_y2_raw"], out)
    out["mu3_hat_raw_fe"] = _predict_with_model_design(models["fe_y3_raw"], out)

    return out

def estimate_mbme_for_grid(pred_grid: pd.DataFrame, family: str, cfg: CFG) -> pd.DataFrame:
    """
    family ∈ {'gam','quad','fe'} uses mu{1,2,3}_hat_raw_{family}
    Returns long table: [n, prcp_group, model, y, density]
    """
    rows = []
    for _, r in pred_grid.iterrows():
        mu1 = r[f"mu1_hat_raw_{family}"]
        mu2 = r[f"mu2_hat_raw_{family}"]
        mu3 = r[f"mu3_hat_raw_{family}"]
        y_f = mbme_density_for_row(mu1, mu2, mu3, cfg)
        if y_f is None:
            continue
        y, f = y_f
        rows.append(pd.DataFrame({
            "n": r["n"], "prcp_group": r["prcp_group"], "model": family.upper(),
            "y": y, "density": f
        }))
    if not rows:
        return pd.DataFrame(columns=["n","prcp_group","model","y","density"])
    return pd.concat(rows, ignore_index=True)

def compute_utilities(pred_grid: pd.DataFrame, mbme_long: pd.DataFrame, cfg: CFG) -> pd.DataFrame:
    """
    Add columns at (n, prcp_group):
      util_rn_{family}, util_ra_norm_{family}_CARA, util_ra_norm_{family}_CRRA,
      util_ra_mbme_{family}_CARA, util_ra_mbme_{family}_CRRA
    """
    out = pred_grid.copy()

    for fam in ["gam","fe"]:
        # risk-neutral from raw μ1
        out[f"util_rn_{fam}"] = cfg.p_yield * out[f"mu1_hat_raw_{fam}"] - cfg.p_n * out["n"]

        var = out[f"mu2_hat_raw_{fam}"] - out[f"mu1_hat_raw_{fam}"]**2
        var = var.clip(lower=1e-9)

        out[f"util_ra_norm_{fam}_CARA"] = [
            expected_u_normal(mu1, v, n, cfg, "CARA")
            for mu1, v, n in zip(out[f"mu1_hat_raw_{fam}"], var, out["n"])
        ]
        out[f"util_ra_norm_{fam}_CRRA"] = [
            expected_u_normal(mu1, v, n, cfg, "CRRA")
            for mu1, v, n in zip(out[f"mu1_hat_raw_{fam}"], var, out["n"])
        ]

        # MBME-based
        key = (mbme_long["model"] == fam.upper())
        by_key = mbme_long.loc[key].groupby(["n","prcp_group"])
        mbme_map = {k: v for k, v in by_key}
        mbme_cara = []
        mbme_crra = []
        for _, row in out.iterrows():
            k = (row["n"], row["prcp_group"])
            if k not in mbme_map:
                mbme_cara.append(np.nan); mbme_crra.append(np.nan); continue
            block = mbme_map[k]
            y, f = block["y"].to_numpy(), block["density"].to_numpy()
            mbme_cara.append( expected_u_mbme(y, f, float(row["n"]), cfg, "CARA") )
            mbme_crra.append( expected_u_mbme(y, f, float(row["n"]), cfg, "CRRA") )
        out[f"util_ra_mbme_{fam}_CARA"] = mbme_cara
        out[f"util_ra_mbme_{fam}_CRRA"] = mbme_crra

    return out

# ---------------- Belief-weighted CE aggregation & risk class ----- #

def _normalize_beliefs(beliefs: Optional[Dict[str, float]], groups: Iterable[str]) -> Dict[str, float]:
    """Return a dict with weights for each group label (sum to 1)."""
    groups = [str(g) for g in groups]
    if beliefs is None:
        w = {g: 1.0/len(groups) for g in groups}
        return w
    # project to provided groups only, renorm
    w = {str(k): float(v) for k,v in beliefs.items() if str(k) in groups}
    tot = sum(w.values())
    if tot <= 0:
        w = {g: 1.0/len(groups) for g in groups}
    else:
        w = {k: v/tot for k,v in w.items()}
    # fill any missing groups with zero (keeps consistent keys)
    for g in groups:
        w.setdefault(g, 0.0)
    return w

def aggregate_ce_with_beliefs(util_grid: pd.DataFrame, family: str, cfg: CFG, beliefs: Optional[Dict[str, float]]=None) -> pd.DataFrame:
    """
    Mix across precip groups using subjective beliefs π to get per-N CE curves.
    Returns df with columns:
      n, ce_rn, ce_cara_norm, ce_cara_mbme, eu_norm_mix, eu_mbme_mix
    """
    fam = family.lower()
    gcol = "prcp_group"
    # ensure groups are strings
    ug = util_grid[gcol].astype(str).unique().tolist()
    w = _normalize_beliefs(beliefs, ug)

    # per (n, group) EUs needed
    df = util_grid.copy()
    df[gcol] = df[gcol].astype(str)

    # group by n to mix
    rows = []
    for n, block in df.groupby("n", as_index=False):
        # RN mixture (mean profit): weighted mean of util_rn
        rn = 0.0
        eu_norm = 0.0
        eu_mbme = 0.0
        for g, b in w.items():
            sub = block.loc[block[gcol]==g]
            if sub.empty:
                continue
            rn += b * sub[f"util_rn_{fam}"].iloc[0]
            eu_norm += b * sub[f"util_ra_norm_{fam}_CARA"].iloc[0]   # E[U] at (n,g)
            eu_mbme += b * sub[f"util_ra_mbme_{fam}_CARA"].iloc[0]   # E[U] at (n,g)

        # convert CARA E[U] mixtures to CE
        ce_norm = (-1.0/cfg.cara_r) * np.log(-eu_norm) if eu_norm < 0 else np.nan
        ce_mbme = (-1.0/cfg.cara_r) * np.log(-eu_mbme) if eu_mbme < 0 else np.nan

        rows.append(dict(n=float(n), ce_rn=rn, ce_cara_norm=ce_norm, ce_cara_mbme=ce_mbme,
                         eu_norm_mix=eu_norm, eu_mbme_mix=eu_mbme))
    return pd.DataFrame(rows).sort_values("n").reset_index(drop=True)

def classify_input_risk(grid_pred: pd.DataFrame, beliefs: Optional[Dict[str, float]]=None,
                        use_col: str = "mu1_hat_ct_gam") -> pd.DataFrame:
    """
    Classify N as risk-increasing vs risk-reducing by sign of Cov(g, g_N) across precip groups, per N.
    Returns: df with columns [n, cov_g_gN, class]
    """
    gcol = "prcp_group"
    df = grid_pred[["n", gcol, use_col]].copy()
    df[gcol] = df[gcol].astype(str)

    # numerical derivative of g w.r.t n within each group
    d_list = []
    for g, grp in df.groupby(gcol):
        grp = grp.sort_values("n")
        g_vals = grp[use_col].to_numpy()
        n_vals = grp["n"].to_numpy()
        # gradient with respect to n
        if len(n_vals) >= 3 and np.all(np.diff(n_vals) > 0):
            gn = np.gradient(g_vals, n_vals)
        else:
            # fallback simple finite diff
            gn = np.append(np.diff(g_vals)/np.diff(n_vals), np.nan)
        tmp = grp.copy()
        tmp["g_N"] = gn
        d_list.append(tmp)
    ddf = pd.concat(d_list, ignore_index=True).dropna(subset=["g_N"])

    # beliefs
    ug = ddf[gcol].unique().tolist()
    w = _normalize_beliefs(beliefs, ug)

    # compute weighted covariance across groups for each n
    out_rows = []
    for n, block in ddf.groupby("n"):
        w_here = np.array([w[g] for g in block[gcol]])
        w_here = w_here / w_here.sum() if w_here.sum() > 0 else np.ones(len(block))/len(block)
        g_vals = block[use_col].to_numpy()
        gn_vals = block["g_N"].to_numpy()
        eg = np.sum(w_here * g_vals)
        egn = np.sum(w_here * gn_vals)
        cov = np.sum(w_here * (g_vals - eg) * (gn_vals - egn))
        label = "risk-reducing (Cov<0)" if cov < 0 else "risk-increasing (Cov>0)"
        out_rows.append(dict(n=float(n), cov_g_gN=cov, classification=label))
    return pd.DataFrame(out_rows).sort_values("n").reset_index(drop=True)

def argmax_n(ce_df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Return the n* that maximizes each CE column."""
    rows = []
    for c in cols:
        sub = ce_df.dropna(subset=[c]).sort_values(c, ascending=False)
        if sub.empty:
            rows.append(dict(metric=c, n_star=np.nan, value=np.nan))
        else:
            top = sub.iloc[0]
            rows.append(dict(metric=c, n_star=float(top["n"]), value=float(top[c])))
    return pd.DataFrame(rows)
