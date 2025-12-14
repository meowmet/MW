# model6_balanced.py
# Optimized for low overfitting (target ~1.0) with better overall performance
# Key improvements: Stronger regularization, better feature engineering, and ensemble-like stability

import os
import re
import json
import time
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb


def set_thread_env(n: int):
    n = int(max(1, n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} | INFO | {msg}")


def robust_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            if df.shape[1] >= 3:
                return df
        except Exception as e:
            last_err = e
    if last_err:
        log(f"[WARN] robust_read_csv exhausted; fallback. Last err={type(last_err).__name__}: {last_err}")
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def normalize_colnames(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", "_", c2)
        out.append(c2)
    return out


def clean_price_to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    digits = re.sub(r"[^\d]", "", s)
    return float(digits) if digits else np.nan


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
    yes_vals = {"yes", "true", "1", "evet", "uygun", "eligible", "var"}
    out = df[s2.isin(yes_vals)].copy()
    return out, True, loan_col


def to_datetime_safe(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def parse_rooms(value) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", s)
    if not m:
        try:
            return int(float(s))
        except Exception:
            return None
    return int(m.group(1)) + int(m.group(2))


def detect_binary_like_cols(df: pd.DataFrame, max_unique: int = 3) -> List[str]:
    yes_set = {"1", "yes", "true", "evet", "var", "uygun", "eligible"}
    no_set = {"0", "no", "false", "hayir", "hayır", "yok", "degil", "değil", "uygun_degil", "uygun_değil"}

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
    yes_set = {"1", "yes", "true", "evet", "var", "uygun", "eligible"}
    if pd.api.types.is_numeric_dtype(series):
        v = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
        return (v == 1).astype(np.int32)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(yes_set).astype(np.int32).to_numpy()


@dataclass
class PreprocessStats:
    all_missing: int
    high_missing: int
    constant: int
    total_dropped: int


class Preprocessor:
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
        if self._room_col is None:
            for c in df.columns:
                if "room" in c.lower():
                    self._room_col = c
                    break

        for c in df.columns:
            cl = c.lower()
            if self._net_col is None and ("net" in cl and ("m2" in cl or "m²" in cl or "area" in cl)):
                self._net_col = c
            if self._gross_col is None and ("gross" in cl and ("m2" in cl or "m²" in cl or "area" in cl)):
                self._gross_col = c

        for c in df.columns:
            cl = c.lower()
            if any(k in cl for k in ["building_age", "age_of_building", "bina_yasi", "bina_yaşı", "yas", "yaş"]):
                self._age_col = c
                break

        self._amenity_cols = detect_binary_like_cols(df, max_unique=3)

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = []

        date_candidates = []
        for c in df.columns:
            cl = c.lower()
            if "adrtisement_date" in cl or "advertisement_date" in cl:
                date_candidates.append(c)
            if "pick_up_data_time" in cl or "pickup_data_time" in cl or "pick_up_date" in cl:
                date_candidates.append(c)

        for dcol in date_candidates:
            dt = to_datetime_safe(df[dcol])
            if dt.notna().sum() > 0:
                df[f"{dcol}_year"] = dt.dt.year.astype(float)
                df[f"{dcol}_month"] = dt.dt.month.astype(float)
                df[f"{dcol}_day"] = dt.dt.day.astype(float)
                df[f"{dcol}_dayofweek"] = dt.dt.dayofweek.astype(float)
                engineered.extend([f"{dcol}_year", f"{dcol}_month", f"{dcol}_day", f"{dcol}_dayofweek"])

        if len(date_candidates) >= 2:
            ad_col = date_candidates[0]
            pk_col = date_candidates[1]
            ad_dt = to_datetime_safe(df[ad_col])
            pk_dt = to_datetime_safe(df[pk_col])
            mask = ad_dt.notna() & pk_dt.notna()
            delta = (pk_dt - ad_dt).dt.total_seconds() / 86400.0
            df.loc[mask, "listing_age_days"] = delta[mask]
            engineered.append("listing_age_days")

        if self._room_col is not None:
            df["total_rooms"] = df[self._room_col].apply(parse_rooms).astype(float)
            engineered.append("total_rooms")

        if self._net_col is not None and self._gross_col is not None:
            net_val = pd.to_numeric(df[self._net_col], errors="coerce")
            gross_val = pd.to_numeric(df[self._gross_col], errors="coerce")
            ratio = net_val / np.clip(gross_val, 1e-6, np.inf)
            df["net_gross_ratio"] = ratio
            engineered.append("net_gross_ratio")
            
            # Add area interaction features
            df["net_x_gross"] = net_val * gross_val
            df["net_sqrt"] = np.sqrt(net_val.clip(0))
            df["gross_sqrt"] = np.sqrt(gross_val.clip(0))
            engineered.extend(["net_x_gross", "net_sqrt", "gross_sqrt"])

        if self._age_col is not None:
            age_vals = pd.to_numeric(df[self._age_col], errors="coerce")
            bins = [-np.inf, 5, 10, 20, 30, np.inf]
            labels = ["0-5", "6-10", "11-20", "21-30", "30+"]
            df["building_age_bin"] = pd.cut(age_vals, bins=bins, labels=labels)
            engineered.append("building_age_bin")
            
            # Age squared for non-linear effects
            df["building_age_squared"] = age_vals ** 2
            engineered.append("building_age_squared")

        if len(self._amenity_cols) > 0:
            am_bin = df[self._amenity_cols].apply(binary_to_01)
            df["amenity_count"] = am_bin.sum(axis=1).astype(float)
            engineered.append("amenity_count")
            
            # Amenity ratio feature
            df["amenity_density"] = am_bin.mean(axis=1).astype(float)
            engineered.append("amenity_density")

        # Add price per area features if available
        if self._net_col is not None:
            # This will be filled after target is known in fit phase
            df["has_net_area"] = pd.to_numeric(df[self._net_col], errors="coerce").notna().astype(float)
            engineered.append("has_net_area")

        self.engineered_cols_ = engineered
        return df

    def fit(self, X: pd.DataFrame):
        self._detect_common_cols(X)
        X = self._add_features(X.copy())

        all_missing_cols = []
        high_missing_cols = []
        constant_cols = []

        for c in X.columns:
            s = X[c]
            if s.isna().all():
                all_missing_cols.append(c)
                continue

            miss_frac = s.isna().mean()
            if miss_frac > self.drop_missing_over:
                high_missing_cols.append(c)
                continue

            nuniq = s.nunique(dropna=True)
            if nuniq <= 1:
                constant_cols.append(c)

        self.drop_cols_ = list(set(all_missing_cols + high_missing_cols + constant_cols))
        self.stats_ = PreprocessStats(
            all_missing=len(all_missing_cols),
            high_missing=len(high_missing_cols),
            constant=len(constant_cols),
            total_dropped=len(self.drop_cols_)
        )

        X = X.drop(columns=self.drop_cols_, errors="ignore")
        self.feature_cols_ = list(X.columns)

        cat_cols = []
        numeric_cols = []

        for c in X.columns:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s):
                numeric_cols.append(c)
            else:
                cat_cols.append(c)

        self.cat_cols_ = cat_cols

        self.numeric_medians_ = {}
        for c in numeric_cols:
            med = X[c].median()
            if pd.isna(med):
                med = 0.0
            self.numeric_medians_[c] = float(med)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._add_features(X.copy())
        X = X.drop(columns=self.drop_cols_, errors="ignore")

        for c in self.feature_cols_:
            if c not in X.columns:
                X[c] = np.nan

        X = X[self.feature_cols_]

        for c in X.columns:
            s = X[c]
            if pd.api.types.is_numeric_dtype(s):
                med = self.numeric_medians_.get(c, 0.0)
                X[c] = s.fillna(med)
            else:
                X[c] = s.fillna("missing").astype(str).astype("category")

        return X


class RobustTargetEncoder:
    def __init__(self, cols: List[str], n_splits: int = 5, seed: int = 42, smoothing: float = 50.0):
        self.cols = cols
        self.n_splits = int(n_splits)
        self.seed = int(seed)
        self.smoothing = float(smoothing)  # Increased smoothing for stability
        self.global_mean_ = None
        self.encodings_ = {}

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        X = X.copy()
        self.global_mean_ = float(np.mean(y))

        for col in self.cols:
            te_vals = np.full(len(X), self.global_mean_, dtype=float)
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

            for tr_idx, val_idx in kf.split(X):
                X_tr = X.iloc[tr_idx]
                y_tr = y[tr_idx]

                cat_col = X_tr[col].astype(str)
                gb = pd.DataFrame({"cat": cat_col, "target": y_tr}).groupby("cat")["target"]
                counts = gb.count()
                sums = gb.sum()
                # More conservative smoothing for small groups
                smoothing_factor = np.where(counts < 10, self.smoothing * 2, self.smoothing)
                means = (sums + smoothing_factor * self.global_mean_) / (counts + smoothing_factor)

                cat_val = X.iloc[val_idx][col].astype(str)
                te_vals[val_idx] = cat_val.map(means).fillna(self.global_mean_).to_numpy()

            X[f"{col}_te"] = te_vals

            cat_all = X[col].astype(str)
            gb_all = pd.DataFrame({"cat": cat_all, "target": y}).groupby("cat")["target"]
            counts_all = gb_all.count()
            sums_all = gb_all.sum()
            smoothing_all = np.where(counts_all < 10, self.smoothing * 2, self.smoothing)
            means_all = (sums_all + smoothing_all * self.global_mean_) / (counts_all + smoothing_all)
            self.encodings_[col] = means_all.to_dict()

        return X

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            cat_vals = X[col].astype(str)
            enc_map = self.encodings_[col]
            X[f"{col}_te"] = cat_vals.map(enc_map).fillna(self.global_mean_)
        return X


def eval_reg(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse_log = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_log = float(r2_score(y_true, y_pred))

    y_true_tl = np.expm1(y_true)
    y_pred_tl = np.expm1(y_pred)

    rmse_tl = float(np.sqrt(mean_squared_error(y_true_tl, y_pred_tl)))
    r2_tl = float(r2_score(y_true_tl, y_pred_tl))
    mae_tl = float(mean_absolute_error(y_true_tl, y_pred_tl))

    return {
        "rmse_log": rmse_log,
        "r2_log": r2_log,
        "rmse_tl": rmse_tl,
        "r2_tl": r2_tl,
        "mae_tl": mae_tl,
    }


def interpret_overfit_index(idx: float) -> str:
    if idx < 1.05:
        return "EXCELLENT (low overfitting)"
    elif idx < 1.15:
        return "GOOD (moderate overfitting)"
    elif idx < 1.25:
        return "MODERATE overfitting"
    else:
        return "HIGH overfitting"


class ETALogger:
    def __init__(self, total_estimators: int, log_every: int = 1000):
        self.total = total_estimators
        self.log_every = log_every
        self.start_time = time.time()

    def __call__(self, env):
        if env.iteration == 1 or (env.iteration % self.log_every == 0) or env.iteration == self.total:
            elapsed = time.time() - self.start_time
            if env.iteration > 0:
                eta = (elapsed / env.iteration) * (self.total - env.iteration)
            else:
                eta = 0

            val_metric = env.evaluation_result_list[0][2] if env.evaluation_result_list else 0.0
            log(f"[LGBM][{env.iteration}/{self.total}] | valid_0.rmse={val_metric:.6f} | elapsed={int(elapsed//60)}m {int(elapsed%60)}s | eta≈{int(eta//60)}m {int(eta%60)}s")


def make_eta_logger(total_estimators: int, log_every: int = 1000):
    return ETALogger(total_estimators, log_every)


def evaluate_bundle_on_saved_split(bundle_path: str, data_path: str, threads: int = 10):
    log(f"[SELF-CHECK] Loading bundle: {bundle_path}")
    bundle = joblib.load(bundle_path)

    pre = bundle["preprocessor"]
    te = bundle.get("target_encoder", None)
    model = bundle["model"]
    best_iter = bundle["best_iter"]
    split = bundle["split"]

    train_ids = split["train_ids"]
    val_ids = split["val_ids"]
    meta = bundle["meta"]

    lo = meta["clip_lo_value"]
    hi = meta["clip_hi_value"]

    log(f"[SELF-CHECK] Reloading data: {data_path}")
    df = robust_read_csv(data_path)
    df.columns = normalize_colnames(df.columns)

    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()

    df, did_filter, loan_col = filter_loan_eligible(df)
    if did_filter:
        log(f"[SELF-CHECK] Loan filter via '{loan_col}' -> rows={len(df)}")

    y_tl_raw = df["Price_num"].astype(float).to_numpy()
    y_tl_clip = np.clip(y_tl_raw, lo, hi)
    y_log = np.log1p(y_tl_clip).astype(float)

    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()

    train_mask = df.index.isin(train_ids)
    val_mask = df.index.isin(val_ids)

    X_train = X[train_mask].copy()
    X_val = X[val_mask].copy()
    y_train = y_log[train_mask]
    y_val = y_log[val_mask]

    Xt_train = pre.transform(X_train)
    Xt_val = pre.transform(X_val)

    if te is not None:
        Xt_train = te.transform(Xt_train)
        Xt_val = te.transform(Xt_val)

    pred_train = np.asarray(model.predict(Xt_train, num_iteration=best_iter), dtype=float)
    pred_val = np.asarray(model.predict(Xt_val, num_iteration=best_iter), dtype=float)

    m_train = eval_reg(y_train, pred_train)
    m_val = eval_reg(y_val, pred_val)
    overfit_idx = m_val["rmse_log"] / max(m_train["rmse_log"], 1e-12)

    log("===== SELF-CHECK RESULTS (reloaded bundle) =====")
    log(f"[TRAIN] RMSE_LOG={m_train['rmse_log']:.6f} | R2_LOG={m_train['r2_log']:.6f} | RMSE_TL={m_train['rmse_tl']:,.2f}")
    log(f"[VAL  ] RMSE_LOG={m_val['rmse_log']:.6f} | R2_LOG={m_val['r2_log']:.6f} | RMSE_TL={m_val['rmse_tl']:,.2f} | MAE_TL={m_val['mae_tl']:,.2f}")
    log(f"[DIAG ] Overfit index={overfit_idx:.3f} -> {interpret_overfit_index(overfit_idx)}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default="data/hackathon_train_set.csv")
    ap.add_argument("--outdir", type=str, default="models")
    ap.add_argument("--out_name", type=str, default="model6_balanced")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)

    # Preprocessing toggles
    ap.add_argument("--drop_missing_over", type=float, default=0.50)
    ap.add_argument("--clip_low_q", type=float, default=0.001)
    ap.add_argument("--clip_high_q", type=float, default=0.995)

    # --- BALANCED CONFIGURATION FOR LOW OVERFITTING ---
    # Conservative settings to achieve ~1.0 overfit index
    
    ap.add_argument("--n_estimators", type=int, default=15000)  # Reduced for faster convergence
    ap.add_argument("--lr", type=float, default=0.005)  # Lower learning rate for stability

    # TREE STRUCTURE: Shallow trees with conservative leaf counts
    ap.add_argument("--num_leaves", type=int, default=31)   # Balanced complexity
    ap.add_argument("--max_depth", type=int, default=4)     # Very shallow to prevent memorization
    ap.add_argument("--min_child_samples", type=int, default=300)  # Higher for more generalization

    # STOCHASTIC: Moderate randomness
    ap.add_argument("--subsample", type=float, default=0.70)  # More data per tree
    ap.add_argument("--subsample_freq", type=int, default=1)
    ap.add_argument("--colsample", type=float, default=0.50)  # Balanced feature sampling

    # REGULARIZATION: Strong but balanced
    ap.add_argument("--reg_lambda", type=float, default=100.0)  # Strong L2
    ap.add_argument("--reg_alpha", type=float, default=50.0)    # Moderate L1
    ap.add_argument("--min_gain_to_split", type=float, default=1.0)  # Higher threshold

    ap.add_argument("--extra_trees", action="store_true", default=True)  # Enable for stability
    ap.add_argument("--force_row_wise", action="store_true", default=True)

    ap.add_argument("--early_stopping_rounds", type=int, default=800)  # More patience
    ap.add_argument("--log_every", type=int, default=1000)

    # Target encoding with stronger smoothing
    ap.add_argument("--use_target_encoding", action="store_true", default=True)
    ap.add_argument("--te_min_unique", type=int, default=20)  # Lower threshold for more categories
    ap.add_argument("--te_splits", type=int, default=5)
    ap.add_argument("--te_smoothing", type=float, default=50.0)  # Higher smoothing

    args = ap.parse_args()

    set_thread_env(args.threads)
    os.makedirs(args.outdir, exist_ok=True)

    log(f"os.cpu_count()={os.cpu_count()} | threads={args.threads}")
    log(f"[LOAD] {args.data}")

    df = robust_read_csv(args.data)
    df.columns = normalize_colnames(df.columns)

    if "Price" not in df.columns:
        raise ValueError("Price column not found. Check file/sep/headers.")

    # target clean
    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()

    # loan filter
    before = len(df)
    df, did_filter, loan_col = filter_loan_eligible(df)
    after = len(df)
    if did_filter:
        log(f"[FILTER] Loan eligible via '{loan_col}': {before} -> {after}")
    else:
        log("[FILTER] Loan filter not applied.")

    # clip bounds (save actual lo/hi values for reproducibility)
    y_tl_raw = df["Price_num"].astype(float).to_numpy()
    lo = float(np.quantile(y_tl_raw, args.clip_low_q))
    hi = float(np.quantile(y_tl_raw, args.clip_high_q))
    y_tl_clip = np.clip(y_tl_raw, lo, hi)
    y_log = np.log1p(y_tl_clip).astype(float)

    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()
    row_ids = df.index.to_numpy()

    # split (and save ids!)
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X, y_log, row_ids,
        test_size=args.test_size,
        random_state=args.seed
    )
    log(f"[SPLIT] Train={X_train.shape} | Val={X_val.shape}")

    # preprocess
    pre = Preprocessor(drop_missing_over=args.drop_missing_over, debug=True)
    pre.fit(X_train)
    Xt_train = pre.transform(X_train)
    Xt_val = pre.transform(X_val)

    if pre.stats_:
        st = pre.stats_
        log(f"[DROP] all_missing={st.all_missing} high_missing={st.high_missing} constant={st.constant} -> total_dropped={st.total_dropped}")

    # optional TE
    te = None
    if args.use_target_encoding and len(pre.cat_cols_) > 0:
        high_card = []
        for c in pre.cat_cols_:
            nun = Xt_train[c].nunique(dropna=True)
            if nun >= args.te_min_unique:
                high_card.append(c)
        if high_card:
            log(f"[TE] Using Robust KFold TE for {len(high_card)} cols (sample): {high_card[:6]}")
            te = RobustTargetEncoder(high_card, n_splits=args.te_splits, seed=args.seed, smoothing=args.te_smoothing)
            Xt_train = te.fit_transform(Xt_train, y_train)
            Xt_val = te.transform(Xt_val)
        else:
            log("[TE] No high-card columns met threshold; skipping.")

    cat_cols = [c for c in Xt_train.columns if str(Xt_train[c].dtype) == "category"]
    log(f"[INFO] Xt_train={Xt_train.shape} | Xt_val={Xt_val.shape} | cat_cols={len(cat_cols)}")

    # model
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=args.n_estimators,
        learning_rate=args.lr,

        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        min_gain_to_split=args.min_gain_to_split,

        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample,

        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,

        extra_trees=args.extra_trees,

        n_jobs=args.threads,
        random_state=args.seed,
        force_row_wise=args.force_row_wise,
        verbosity=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
        make_eta_logger(total_estimators=args.n_estimators, log_every=args.log_every),
    ]

    log("[TRAIN] Fitting LightGBM (Balanced Configuration for Low Overfitting)...")
    t0 = time.time()
    model.fit(
        Xt_train, y_train,
        eval_set=[(Xt_val, y_val)],
        eval_metric="rmse",
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=callbacks,
    )
    train_sec = time.time() - t0

    best_iter = int(getattr(model, "best_iteration_", 0) or 0)
    if best_iter <= 0:
        best_iter = args.n_estimators

    log(f"[TRAIN] done in {int(train_sec//60)}m {int(train_sec%60)}s | best_iter={best_iter}")

    # eval train/val
    pred_train = np.asarray(model.predict(Xt_train, num_iteration=best_iter), dtype=float)
    pred_val = np.asarray(model.predict(Xt_val, num_iteration=best_iter), dtype=float)

    m_train = eval_reg(y_train, pred_train)
    m_val = eval_reg(y_val, pred_val)
    overfit_idx = m_val["rmse_log"] / max(m_train["rmse_log"], 1e-12)

    log("===== OVERFITTING REPORT (same split) =====")
    log(f"[TRAIN] RMSE_LOG={m_train['rmse_log']:.6f} | R2_LOG={m_train['r2_log']:.6f} | RMSE_TL={m_train['rmse_tl']:,.2f}")
    log(f"[VAL  ] RMSE_LOG={m_val['rmse_log']:.6f} | R2_LOG={m_val['r2_log']:.6f} | RMSE_TL={m_val['rmse_tl']:,.2f} | MAE_TL={m_val['mae_tl']:,.2f}")
    log(f"[DIAG ] Overfit index={overfit_idx:.3f} -> {interpret_overfit_index(overfit_idx)}")

    # save meta (for humans)
    meta = {
        "out_name": args.out_name,
        "seed": args.seed,
        "test_size": args.test_size,
        "threads": args.threads,
        "clip_low_q": args.clip_low_q,
        "clip_high_q": args.clip_high_q,
        "clip_lo_value": lo,
        "clip_hi_value": hi,
        "preprocess": {
            "drop_missing_over": args.drop_missing_over,
            "stats": asdict(pre.stats_) if pre.stats_ else None,
            "n_features": len(pre.feature_cols_),
            "n_cat_cols": len(pre.cat_cols_),
            "engineered_cols": pre.engineered_cols_,
        },
        "model_params": model.get_params(deep=False),
        "best_iter": best_iter,
        "metrics_train": m_train,
        "metrics_val": m_val,
        "overfit_index": overfit_idx,
    }

    meta_path = os.path.join(args.outdir, f"{args.out_name}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # save bundle
    bundle = {
        "preprocessor": pre,
        "target_encoder": te,
        "model": model,
        "best_iter": best_iter,
        "split": {
            "train_ids": id_train.tolist(),
            "val_ids": id_val.tolist(),
        },
        "meta": meta,
    }

    bundle_path = os.path.join(args.outdir, f"{args.out_name}_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    log(f"[SAVE] meta: {meta_path}")
    log(f"[SAVE] bundle: {bundle_path}")

    # self-check
    evaluate_bundle_on_saved_split(bundle_path=bundle_path, data_path=args.data, threads=args.threads)

    log("DONE")

if __name__ == "__main__":
    main()