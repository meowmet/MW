# overfit_checker.py
# Purpose:
#   Robust overfitting checker for your saved bundle (model5_lgbm_business_bundle.joblib).
#   - Evaluates on TRAIN vs VAL (same split you choose)
#   - Computes residual quantiles
#   - Computes "overfit index" (val_rmse_log / train_rmse_log)
#   - Optional: Refit KFold CV (true generalization check)
#   - Optional: Learning curve (train_size sweep) refit check

import os
import re
import json
import time
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import lightgbm as lgb


# ----------------------------
# Thread env (helps avoid oversubscription on Windows)
# ----------------------------
def set_thread_env(n: int):
    n = int(max(1, n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


# ----------------------------
# Logging helper
# ----------------------------
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | INFO | {msg}")


# ----------------------------
# Robust I/O + cleaning
# ----------------------------
def robust_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            if "Price" in df.columns and df.shape[1] >= 5:
                return df
        except Exception as e:
            last_err = e
    if last_err:
        log(f"[WARN] robust_read_csv exhausted; falling back. Last err={type(last_err).__name__}: {last_err}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def clean_price_to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    digits = re.sub(r"[^\d]", "", s)
    return float(digits) if digits else np.nan


def normalize_colnames(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", "_", c2)
        out.append(c2)
    return out


def find_loan_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if "available_for_loan" in cl or "available for loan" in cl or ("loan" in cl) or ("kredi" in cl):
            return c
    return None


def filter_loan_eligible(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool, Optional[str]]:
    loan_col = find_loan_col(df)
    if loan_col is None:
        return df, False, None

    non_null_rate = df[loan_col].notna().mean()
    if non_null_rate < 0.05:
        return df, False, loan_col

    s = df[loan_col]
    if pd.api.types.is_numeric_dtype(s):
        out = df[s == 1].copy()
        return out, True, loan_col

    s2 = s.astype(str).str.strip().str.lower()
    yes_vals = {"yes", "true", "1", "evet", "uygun", "eligible"}
    out = df[s2.isin(yes_vals)].copy()
    return out, True, loan_col


# ----------------------------
# Helpers needed by Preprocessor
# ----------------------------
def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def parse_rooms(value) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", s)
    if not m:
        try:
            v = int(float(s))
            return v
        except Exception:
            return None
    a = int(m.group(1))
    b = int(m.group(2))
    return a + b


def detect_binary_like_cols(df: pd.DataFrame, max_unique: int = 3) -> List[str]:
    yes_set = {"1", "yes", "true", "evet", "var", "yok_degil", "uygun"}
    no_set = {"0", "no", "false", "hayir", "hayır", "yok", "degil", "uygun_degil"}

    cols = []
    for c in df.columns:
        s = df[c]
        nun = s.nunique(dropna=True)
        if nun == 0 or nun > max_unique:
            continue

        if pd.api.types.is_numeric_dtype(s):
            uniq = set(pd.to_numeric(s.dropna(), errors="coerce").unique().tolist())
            if uniq.issubset({0, 1}):
                cols.append(c)
            continue

        uniq = set(s.dropna().astype(str).str.strip().str.lower().unique().tolist())
        if len(uniq) == 0 or len(uniq) > max_unique:
            continue
        if uniq.issubset(yes_set.union(no_set)):
            cols.append(c)

    return cols


def binary_to_01(series: pd.Series) -> np.ndarray:
    yes_set = {"1", "yes", "true", "evet", "var", "uygun"}
    if pd.api.types.is_numeric_dtype(series):
        v = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        return (v == 1).astype(np.int32)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(yes_set).astype(np.int32).to_numpy()


# ============================================================
# REAL PREPROCESSOR CLASS (COPIED FROM TRAINING SCRIPT)
# ============================================================

@dataclass
class PreprocessStats:
    all_missing: int
    high_missing: int
    constant: int
    total_dropped: int


class Preprocessor:
    """
    Real implementation so joblib can reconstruct the object with methods intact.
    """

    def __init__(self, drop_missing_over: float = 0.50, debug: bool = True):
        self.drop_missing_over = float(drop_missing_over)
        self.debug = bool(debug)

        self.drop_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.numeric_medians_: Dict[str, float] = {}
        self.feature_cols_: List[str] = []
        self.engineered_cols_: List[str] = []
        self.stats_: Optional[PreprocessStats] = None

        self._room_col: Optional[str] = None
        self._net_col: Optional[str] = None
        self._gross_col: Optional[str] = None
        self._age_col: Optional[str] = None
        self._amenity_cols: List[str] = []

    def _detect_common_cols(self, df: pd.DataFrame):
        # Room-like
        if self._room_col is None:
            for c in df.columns:
                if "room" in c.lower():
                    self._room_col = c
                    break

        # Net/Gross
        for c in df.columns:
            cl = c.lower()
            if self._net_col is None and ("net" in cl and ("m2" in cl or "m²" in cl or "area" in cl)):
                self._net_col = c
            if self._gross_col is None and ("gross" in cl and ("m2" in cl or "m²" in cl or "area" in cl)):
                self._gross_col = c

        # Age
        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["building_age", "age_of_building", "bina_yasi", "bina_yaşı", "yas", "yaş"]):
                self._age_col = c
                break

        # Amenities
        self._amenity_cols = detect_binary_like_cols(df, max_unique=3)

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = []

        # Date parts
        date_candidates = []
        for c in df.columns:
            cl = c.lower()
            if "adrtisement_date" in cl or "advertisement_date" in cl:
                date_candidates.append(c)
            if "pick_up_data_time" in cl or "pickup_data_time" in cl or "pick_up_date" in cl:
                date_candidates.append(c)

        for c in sorted(set(date_candidates)):
            dt = to_datetime_safe(df[c])
            df[c + "_year"] = dt.dt.year
            df[c + "_month"] = dt.dt.month
            df[c + "_day"] = dt.dt.day
            df[c + "_dow"] = dt.dt.dayofweek
            engineered += [c + "_year", c + "_month", c + "_day", c + "_dow"]

        if len(sorted(set(date_candidates))) >= 2:
            c1, c2 = sorted(set(date_candidates))[:2]
            d1 = to_datetime_safe(df[c1])
            d2 = to_datetime_safe(df[c2])
            df["listing_age_days"] = (d2 - d1).dt.days
            engineered.append("listing_age_days")

        if self._room_col and self._room_col in df.columns:
            df["total_rooms"] = df[self._room_col].apply(parse_rooms)
            engineered.append("total_rooms")

        if self._net_col and self._gross_col and (self._net_col in df.columns) and (self._gross_col in df.columns):
            net = pd.to_numeric(df[self._net_col], errors="coerce")
            gross = pd.to_numeric(df[self._gross_col], errors="coerce")
            df["net_gross_ratio"] = (net / gross).replace([np.inf, -np.inf], np.nan)
            engineered.append("net_gross_ratio")

        if self._amenity_cols:
            mat = np.zeros((len(df),), dtype=np.int32)
            for c in self._amenity_cols:
                mat += binary_to_01(df[c])
            df["amenity_count"] = mat
            engineered.append("amenity_count")

            groups = {
                "amenity_luxury_count": ["sauna", "jacuzzi", "pool", "spa", "hamam"],
                "amenity_security_count": ["security", "alarm", "camera", "guard", "site"],
            }
            low_cols = {c: c.lower() for c in self._amenity_cols}
            for newc, kws in groups.items():
                cols = [c for c, cl in low_cols.items() if any(k in cl for k in kws)]
                if cols:
                    m = np.zeros((len(df),), dtype=np.int32)
                    for c in cols:
                        m += binary_to_01(df[c])
                    df[newc] = m
                    engineered.append(newc)

        if self._age_col and self._age_col in df.columns:
            age = pd.to_numeric(df[self._age_col], errors="coerce")
            bins = [-np.inf, 0, 5, 20, 40, np.inf]
            labels = [0, 1, 2, 3, 4]
            df["building_age_bin"] = pd.cut(age, bins=bins, labels=labels).astype("float")
            engineered.append("building_age_bin")

        self.engineered_cols_ = sorted(set(self.engineered_cols_).union(engineered))
        return df

    def fit(self, X: pd.DataFrame):
        df = X.copy()
        self._detect_common_cols(df)
        df = self._add_features(df)

        # drop raw date cols if derived parts exist
        raw_date_cols = []
        for c in df.columns:
            if df[c].dtype == "object":
                if (c + "_year") in df.columns and (c + "_month") in df.columns:
                    raw_date_cols.append(c)
        df = df.drop(columns=raw_date_cols, errors="ignore")

        all_missing = [c for c in df.columns if df[c].isna().all()]
        miss_rate = df.isna().mean()
        high_missing = miss_rate[miss_rate > self.drop_missing_over].index.tolist()
        nun = df.nunique(dropna=False)
        const_cols = nun[nun <= 1].index.tolist()

        self.drop_cols_ = sorted(set(all_missing + high_missing + const_cols))
        df = df.drop(columns=self.drop_cols_, errors="ignore")

        self.stats_ = PreprocessStats(
            all_missing=len(all_missing),
            high_missing=len(high_missing),
            constant=len(const_cols),
            total_dropped=len(self.drop_cols_),
        )

        self.cat_cols_ = [c for c in df.columns if df[c].dtype == "object"]

        self.numeric_medians_ = {}
        for c in df.columns:
            if c in self.cat_cols_:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            self.numeric_medians_[c] = float(s.median()) if s.notna().any() else 0.0

        self.feature_cols_ = list(df.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df = self._add_features(df)

        raw_date_cols = []
        for c in df.columns:
            if df[c].dtype == "object":
                if (c + "_year") in df.columns and (c + "_month") in df.columns:
                    raw_date_cols.append(c)
        df = df.drop(columns=raw_date_cols, errors="ignore")

        df = df.drop(columns=self.drop_cols_, errors="ignore")

        for c in self.feature_cols_:
            if c not in df.columns:
                df[c] = np.nan
        df = df[self.feature_cols_].copy()

        for c in df.columns:
            if c in self.cat_cols_:
                df[c] = df[c].fillna("Unknown").astype(str)
                df[c] = df[c].astype("category")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.numeric_medians_.get(c, 0.0))

        return df


class KFoldTargetEncoder:
    def __init__(self, cols: List[str], n_splits: int = 5, seed: int = 42, smoothing: float = 20.0):
        self.cols = cols
        self.n_splits = n_splits
        self.seed = seed
        self.smoothing = float(smoothing)
        self.global_mean_ = None
        self.maps_: Dict[str, Dict] = {}

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        X = X.copy()
        y = np.asarray(y, dtype=float)
        self.global_mean_ = float(np.mean(y))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for c in self.cols:
            out = np.zeros((len(X),), dtype=float)
            for tr_idx, va_idx in kf.split(X):
                tr = X.iloc[tr_idx]
                ytr = y[tr_idx]

                tmp = pd.DataFrame({c: tr[c].values, "_y": ytr})
                grp = tmp.groupby(c)["_y"].agg(["count", "mean"])
                count = grp["count"]
                mean = grp["mean"]

                smooth = (count * mean + self.smoothing * self.global_mean_) / (count + self.smoothing)
                m = smooth.to_dict()

                out[va_idx] = X.iloc[va_idx][c].map(m).fillna(self.global_mean_).to_numpy()

            X[c + "__te"] = out

            tmp_full = pd.DataFrame({c: X[c].values, "_y": y})
            grp_full = tmp_full.groupby(c)["_y"].agg(["count", "mean"])
            count = grp_full["count"]
            mean = grp_full["mean"]
            smooth_full = (count * mean + self.smoothing * self.global_mean_) / (count + self.smoothing)
            self.maps_[c] = smooth_full.to_dict()

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.cols:
            m = self.maps_.get(c, {})
            X[c + "__te"] = X[c].map(m).fillna(self.global_mean_)
        return X


# ----------------------------
# Metrics
# ----------------------------
def rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def eval_reg(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)

    rmse_log = rmse(y_true_log, y_pred_log)
    r2_log = float(r2_score(y_true_log, y_pred_log))

    y_true_tl = np.expm1(y_true_log)
    y_pred_tl = np.expm1(y_pred_log)

    rmse_tl = rmse(y_true_tl, y_pred_tl)
    r2_tl = float(r2_score(y_true_tl, y_pred_tl))
    mae_tl = float(mean_absolute_error(y_true_tl, y_pred_tl))

    return {
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "rmse_tl": rmse_tl,
        "r2_tl": r2_tl,
        "mae_tl": mae_tl,
    }


def residual_quantiles(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    resid = np.asarray(y_pred_log, dtype=float) - np.asarray(y_true_log, dtype=float)
    qs = np.quantile(resid, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    return {
        "min": float(qs[0]),
        "p01": float(qs[1]),
        "p05": float(qs[2]),
        "median": float(qs[3]),
        "p95": float(qs[4]),
        "p99": float(qs[5]),
        "max": float(qs[6]),
        "mean": float(np.mean(resid)),
        "std": float(np.std(resid)),
    }


def interpret_overfit_index(idx: float) -> str:
    if idx <= 1.05:
        return "VERY LOW overfitting"
    if idx <= 1.15:
        return "LOW overfitting (acceptable)"
    if idx <= 1.30:
        return "MODERATE overfitting"
    return "HIGH overfitting risk"


# ----------------------------
# Utility: get params for refit
# ----------------------------
def extract_lgb_params_from_model(model: lgb.LGBMRegressor) -> Dict:
    # get_params() includes many fields; keep only the most important ones for consistent refit
    p = model.get_params(deep=False)
    keep = [
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "reg_alpha",
        "random_state",
        "n_jobs",
        "force_row_wise",
        "verbosity",
        "max_depth",  # Added this as you are now using it
    ]
    out = {k: p.get(k) for k in keep if k in p}
    # Ensure these exist
    out.setdefault("verbosity", -1)
    return out


def safe_cat_feature_list(Xt: pd.DataFrame) -> List[str]:
    return [c for c in Xt.columns if str(Xt[c].dtype) == "category"]


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default="data/hackathon_train_set.csv")
    ap.add_argument("--bundle", type=str, default="models/model5_lgbm_business_bundle.joblib")
    ap.add_argument("--meta", type=str, default="models/model5_lgbm_business_meta.json")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)

    # diagnostics
    ap.add_argument("--run_cv", action="store_true", default=False)
    ap.add_argument("--cv_splits", type=int, default=5)
    ap.add_argument("--learning_curve", action="store_true", default=False)
    ap.add_argument("--lc_points", type=int, default=6)
    ap.add_argument("--lc_min_frac", type=float, default=0.15)

    args = ap.parse_args()

    set_thread_env(args.threads)
    log(f"os.cpu_count() = {os.cpu_count()} | threads={args.threads}")

    # Load bundle + meta
    log(f"[LOAD] bundle: {args.bundle}")
    bundle = joblib.load(args.bundle)
    pre = bundle.get("preprocessor", None)
    te = bundle.get("target_encoder", None)
    model = bundle.get("model", None)
    best_iter = int(bundle.get("best_iter", 0) or 0)

    if pre is None or model is None:
        raise RuntimeError("Bundle missing 'preprocessor' or 'model'. Wrong bundle file?")

    meta = {}
    if args.meta and os.path.exists(args.meta):
        log(f"[LOAD] meta: {args.meta}")
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

    if best_iter <= 0:
        best_iter = int(meta.get("best_iter", 0) or getattr(model, "best_iteration_", 0) or model.get_params().get("n_estimators", 20000))

    log(f"[INFO] best_iter={best_iter}")
    log(f"[INFO] te_in_bundle={te is not None}")

    # Load data
    log(f"[LOAD] data: {args.data}")
    df = robust_read_csv(args.data)
    df.columns = normalize_colnames(df.columns)
    if "Price" not in df.columns:
        raise ValueError("Price column not found (after normalization).")

    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()

    # Loan filter (same as training style)
    before = len(df)
    df, did_filter, loan_col = filter_loan_eligible(df)
    after = len(df)
    if did_filter:
        log(f"[FILTER] Loan eligible via '{loan_col}': {before} -> {after}")
    else:
        log("[FILTER] Loan filter not applied (missing/too empty).")

    # Use clip quantiles from meta if present, else fallback
    clip_q = meta.get("clip_quantiles", [0.001, 0.995])
    clip_low_q = float(clip_q[0])
    clip_high_q = float(clip_q[1])

    y_tl_raw = df["Price_num"].astype(float).to_numpy()
    lo = float(np.quantile(y_tl_raw, clip_low_q))
    hi = float(np.quantile(y_tl_raw, clip_high_q))
    y_tl_clip = np.clip(y_tl_raw, lo, hi)
    y_log = np.log1p(y_tl_clip).astype(float)

    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()
    row_ids = df.index.to_numpy()

    # Split (reproducible)
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X, y_log, row_ids,
        test_size=args.test_size,
        random_state=args.seed
    )
    log(f"[SPLIT] Train={X_train.shape} | Val={X_val.shape}")

    # Transform via stored pipeline
    log("[PRE] Transforming train/val with stored preprocessor...")
    Xt_train = pre.transform(X_train)
    Xt_val = pre.transform(X_val)

    if te is not None:
        # IMPORTANT NOTE:
        # If TE was trained with leakage-safe OOF and you saved only final maps,
        # applying te.transform on train will be optimistic.
        log("[PRE] Applying stored target_encoder maps (train metrics may be optimistic).")
        Xt_train = te.transform(Xt_train)
        Xt_val = te.transform(Xt_val)

    log(f"[PRE] Xt_train={Xt_train.shape} | Xt_val={Xt_val.shape}")

    # Predict
    log("[EVAL] Predict TRAIN...")
    pred_train = np.asarray(model.predict(Xt_train, num_iteration=best_iter), dtype=float)
    m_train = eval_reg(y_train, pred_train)
    q_train = residual_quantiles(y_train, pred_train)

    log("[EVAL] Predict VAL...")
    pred_val = np.asarray(model.predict(Xt_val, num_iteration=best_iter), dtype=float)
    m_val = eval_reg(y_val, pred_val)
    q_val = residual_quantiles(y_val, pred_val)

    # Print summary
    log("===== TRAIN (in-sample) =====")
    log(f"RMSE_LOG: {m_train['rmse_log']:.6f} | R2_LOG: {m_train['r2_log']:.6f}")
    log(f"RMSE_TL : {m_train['rmse_tl']:,.2f} | R2_TL : {m_train['r2_tl']:.6f} | MAE_TL: {m_train['mae_tl']:,.2f}")
    log(f"[TRAIN] Residuals(log) min={q_train['min']:.4f}, p01={q_train['p01']:.4f}, p05={q_train['p05']:.4f}, "
        f"median={q_train['median']:.4f}, p95={q_train['p95']:.4f}, p99={q_train['p99']:.4f}, max={q_train['max']:.4f}")
    log(f"[TRAIN] Residual mean={q_train['mean']:.6f}, std={q_train['std']:.6f}")

    log("===== VAL (held-out) =====")
    log(f"RMSE_LOG: {m_val['rmse_log']:.6f} | R2_LOG: {m_val['r2_log']:.6f}")
    log(f"RMSE_TL : {m_val['rmse_tl']:,.2f} | R2_TL : {m_val['r2_tl']:.6f} | MAE_TL: {m_val['mae_tl']:,.2f}")
    log(f"[VAL] Residuals(log) min={q_val['min']:.4f}, p01={q_val['p01']:.4f}, p05={q_val['p05']:.4f}, "
        f"median={q_val['median']:.4f}, p95={q_val['p95']:.4f}, p99={q_val['p99']:.4f}, max={q_val['max']:.4f}")
    log(f"[VAL] Residual mean={q_val['mean']:.6f}, std={q_val['std']:.6f}")

    overfit_index = float(m_val["rmse_log"] / max(m_train["rmse_log"], 1e-12))
    log(f"[DIAG] Overfit index = val_rmse_log / train_rmse_log = {overfit_index:.3f}")
    log(f"[DIAG] {interpret_overfit_index(overfit_index)}")

    # --------------------------------------------
    # Optional: True KFold CV refit (no early stop)
    # --------------------------------------------
    if args.run_cv:
        log(f"[CV] Running {args.cv_splits}-fold CV (refitting from scratch each fold)...")

        # Use the stored preprocessor/TE to transform full X once
        Xt_full = pre.transform(X)
        if te is not None:
            Xt_full = te.transform(Xt_full)

        # Refit params from original model (stable)
        params = extract_lgb_params_from_model(model)
        params["n_jobs"] = args.threads
        params["random_state"] = args.seed

        # Keep estimators consistent with best_iter (recommended)
        # If best_iter is too small/large, you can override with CLI later.
        params["n_estimators"] = int(best_iter) if best_iter > 0 else int(params.get("n_estimators", 20000))

        kf = KFold(n_splits=args.cv_splits, shuffle=True, random_state=args.seed)
        cat_cols = safe_cat_feature_list(Xt_full)

        rmses = []
        r2s = []
        maes = []

        fold = 0
        for tr_idx, va_idx in kf.split(Xt_full):
            fold += 1
            Xtr = Xt_full.iloc[tr_idx]
            Xva = Xt_full.iloc[va_idx]
            ytr = y_log[tr_idx]
            yva = y_log[va_idx]

            mdl = lgb.LGBMRegressor(**params)

            t0 = time.time()
            if cat_cols:
                mdl.fit(Xtr, ytr, categorical_feature=cat_cols)
            else:
                mdl.fit(Xtr, ytr)
            dt = time.time() - t0

            pva = mdl.predict(Xva)
            m = eval_reg(yva, pva)

            rmses.append(m["rmse_log"])
            r2s.append(m["r2_log"])
            maes.append(m["mae_tl"])

            log(f"[CV] Fold {fold}/{args.cv_splits} | train_time={int(dt)}s | RMSE_LOG={m['rmse_log']:.6f} | R2_LOG={m['r2_log']:.6f}")

        log("===== CV SUMMARY =====")
        log(f"RMSE_LOG mean={float(np.mean(rmses)):.6f} std={float(np.std(rmses)):.6f}")
        log(f"R2_LOG   mean={float(np.mean(r2s)):.6f} std={float(np.std(r2s)):.6f}")
        log(f"MAE_TL   mean={float(np.mean(maes)):.2f} std={float(np.std(maes)):.2f}")
        log(f"[COMPARE] single-split VAL RMSE_LOG={m_val['rmse_log']:.6f}")

    # --------------------------------------------
    # Optional: Learning curve (refit on subsets)
    # --------------------------------------------
    if args.learning_curve:
        log("[LC] Running learning curve (refit on increasing train fractions)...")

        # Transform full once (fast)
        Xt_full = pre.transform(X)
        if te is not None:
            Xt_full = te.transform(Xt_full)

        params = extract_lgb_params_from_model(model)
        params["n_jobs"] = args.threads
        params["random_state"] = args.seed
        params["n_estimators"] = int(best_iter) if best_iter > 0 else int(params.get("n_estimators", 20000))

        cat_cols = safe_cat_feature_list(Xt_full)

        # Create one fixed val set for LC
        Xtr_all, Xva_fixed, ytr_all, yva_fixed = train_test_split(
            Xt_full, y_log, test_size=args.test_size, random_state=args.seed
        )

        n_total = len(Xtr_all)
        min_frac = float(np.clip(args.lc_min_frac, 0.05, 0.9))
        points = int(max(3, args.lc_points))
        fracs = np.linspace(min_frac, 1.0, points)

        rows = []
        for f in fracs:
            n = int(max(100, round(n_total * f)))
            Xtr = Xtr_all.iloc[:n]
            ytr = ytr_all[:n]

            mdl = lgb.LGBMRegressor(**params)
            t0 = time.time()
            if cat_cols:
                mdl.fit(Xtr, ytr, categorical_feature=cat_cols)
            else:
                mdl.fit(Xtr, ytr)
            dt = time.time() - t0

            p_tr = mdl.predict(Xtr)
            p_va = mdl.predict(Xva_fixed)

            mt = eval_reg(ytr, p_tr)
            mv = eval_reg(yva_fixed, p_va)

            rows.append({
                "train_frac": float(f),
                "train_n": int(n),
                "train_rmse_log": float(mt["rmse_log"]),
                "val_rmse_log": float(mv["rmse_log"]),
                "train_r2_log": float(mt["r2_log"]),
                "val_r2_log": float(mv["r2_log"]),
                "train_time_s": float(dt),
            })

            log(f"[LC] frac={f:.2f} n={n} | train_rmse_log={mt['rmse_log']:.4f} | val_rmse_log={mv['rmse_log']:.4f} | time={dt:.1f}s")

        lc_df = pd.DataFrame(rows)
        out_path = "learning_curve_overfit.csv"
        lc_df.to_csv(out_path, index=False, encoding="utf-8")
        log(f"[LC][SAVE] {out_path}")

    log("[DONE]")


if __name__ == "__main__":
    main()