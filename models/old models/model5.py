import os
import re
import json
import time
import argparse
from dataclasses import asdict, dataclass
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


def find_loan_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        cl = c.lower()
        if "available for loan" in cl or ("loan" in cl) or ("kredi" in cl):
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


def normalize_colnames(cols: List[str]) -> List[str]:
    out = []
    for c in cols:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", "_", c2)
        out.append(c2)
    return out


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


@dataclass
class PreprocessStats:
    all_missing: int
    high_missing: int
    constant: int
    total_dropped: int


class Preprocessor:
    """
    Same as model4, plus (still safe):
      - amenity grouped counts, building_age_bin, listing_age_days, total_rooms, net_gross_ratio
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


# ----------------------------
# Leakage-safe KFold target encoding (FIXED version from model4)
# ----------------------------
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
# LightGBM logging callback with ETA
# ----------------------------
def make_eta_logger(total_estimators: int, log_every: int = 200):
    start_time = time.time()

    def _callback(env):
        it = env.iteration + 1
        if it == 1 or (it % log_every == 0):
            elapsed = time.time() - start_time
            sec_per_iter = elapsed / max(it, 1)
            remaining = max(total_estimators - it, 0)
            eta_sec = remaining * sec_per_iter

            msg_metric = ""
            if env.evaluation_result_list:
                dname, ename, val, _ = env.evaluation_result_list[-1]
                msg_metric = f" | {dname}.{ename}={val:.6f}"

            log(f"[LGBM][{it}/{total_estimators}]{msg_metric} | elapsed={int(elapsed//60)}m {int(elapsed%60)}s | eta≈{int(eta_sec//60)}m {int(eta_sec%60)}s")

    _callback.order = 10
    return _callback


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


# ----------------------------
# Business Logic (learn thresholds + classify)
# ----------------------------
def learn_thresholds_from_train(
    y_train_log: np.ndarray,
    pred_train_log: np.ndarray,
    q_low: float = 0.20,
    q_high: float = 0.80,
    clamp_min: float = 0.03,
    clamp_max: float = 0.30,
) -> Dict[str, float]:
    """
    diff_ratio = (listing - fair) / fair
      negative => listing cheaper than fair (deal)
      positive => listing more expensive (expensive)

    We learn:
      t_buy  = abs(quantile(diff_ratio, q_low))   (lower tail magnitude)
      t_sell = quantile(diff_ratio, q_high)       (upper tail)
    """
    y_tl = np.expm1(np.asarray(y_train_log, dtype=float))
    fair_tl = np.expm1(np.asarray(pred_train_log, dtype=float))
    fair_tl = np.clip(fair_tl, 1.0, np.inf)

    diff = (y_tl - fair_tl) / fair_tl

    buy_tail = float(np.quantile(diff, q_low))   # likely negative
    sell_tail = float(np.quantile(diff, q_high)) # likely positive

    t_buy = float(abs(buy_tail))
    t_sell = float(sell_tail)

    t_buy = float(np.clip(t_buy, clamp_min, clamp_max))
    t_sell = float(np.clip(t_sell, clamp_min, clamp_max))

    return {
        "q_low": float(q_low),
        "q_high": float(q_high),
        "t_buy": t_buy,
        "t_sell": t_sell,
        "definition": "FIRSAT if listing <= fair*(1-t_buy); PAHALI if listing >= fair*(1+t_sell); else NORMAL",
    }


def classify_investment(listing_tl: float, fair_tl: float, t_buy: float, t_sell: float) -> str:
    if not np.isfinite(listing_tl) or not np.isfinite(fair_tl) or fair_tl <= 0:
        return "NORMAL"
    if listing_tl <= fair_tl * (1.0 - t_buy):
        return "FIRSAT"
    if listing_tl >= fair_tl * (1.0 + t_sell):
        return "PAHALI"
    return "NORMAL"


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default="data/hackathon_train_set.csv")
    ap.add_argument("--outdir", type=str, default="models")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--drop_missing_over", type=float, default=0.50)
    ap.add_argument("--clip_low_q", type=float, default=0.001)
    ap.add_argument("--clip_high_q", type=float, default=0.995)

    # LGBM params - ANTI-OVERFIT MODE
    ap.add_argument("--n_estimators", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.01)
    
    # Strict complexity constraints
    ap.add_argument("--num_leaves", type=int, default=20)          # Reduced from 100
    ap.add_argument("--max_depth", type=int, default=5)            # ADDED: Shallow trees
    ap.add_argument("--min_child_samples", type=int, default=500)  # Increased from 200
    
    # Stochastic regularization
    ap.add_argument("--subsample", type=float, default=0.6)        # Reduced from 0.8
    ap.add_argument("--colsample", type=float, default=0.4)        # Reduced from 0.8
    
    # L1/L2 Regularization
    ap.add_argument("--reg_lambda", type=float, default=50.0)      # Increased from 10.0
    ap.add_argument("--reg_alpha", type=float, default=20.0)       # Increased from 5.0
    
    ap.add_argument("--early_stopping_rounds", type=int, default=300)
    ap.add_argument("--force_row_wise", action="store_true", default=True)

    # Debug/logging
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--save_top_errors", action="store_true", default=True)

    # Optional target encoding
    ap.add_argument("--use_target_encoding", action="store_true", default=False)
    ap.add_argument("--te_min_unique", type=int, default=30)
    ap.add_argument("--te_splits", type=int, default=5)
    ap.add_argument("--te_smoothing", type=float, default=20.0)

    # Business logic learning
    ap.add_argument("--deal_q_low", type=float, default=0.20)
    ap.add_argument("--deal_q_high", type=float, default=0.80)
    ap.add_argument("--export_val_predictions", action="store_true", default=True)

    args = ap.parse_args()

    set_thread_env(args.threads)
    os.makedirs(args.outdir, exist_ok=True)

    log(f"os.cpu_count() = {os.cpu_count()} | threads={args.threads}")
    log(f"[LOAD] {args.data}")

    df = robust_read_csv(args.data)
    df.columns = normalize_colnames(df.columns)

    if "Price" not in df.columns:
        raise ValueError("Price column not found (after reading/normalizing). Check file/separator.")

    # Clean target
    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()

    # Filter loan eligible
    before = len(df)
    df, did_filter, loan_col = filter_loan_eligible(df)
    after = len(df)
    if did_filter:
        log(f"[FILTER] Loan eligible via '{loan_col}': {before} -> {after}")
    else:
        log("[FILTER] Loan filter not applied (missing/too empty).")

    # Clip + log target
    y_tl_raw = df["Price_num"].astype(float).to_numpy()
    lo = float(np.quantile(y_tl_raw, args.clip_low_q))
    hi = float(np.quantile(y_tl_raw, args.clip_high_q))
    y_tl_clip = np.clip(y_tl_raw, lo, hi)
    y_log = np.log1p(y_tl_clip).astype(float)

    # Features
    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()

    # Keep original row ids
    row_ids = df.index.to_numpy()

    X_train, X_val, y_train, y_val, id_train, id_val, y_train_tl, y_val_tl = train_test_split(
        X, y_log, row_ids, y_tl_clip,
        test_size=args.test_size,
        random_state=args.seed
    )

    # Preprocess
    pre = Preprocessor(drop_missing_over=args.drop_missing_over, debug=True)
    pre.fit(X_train)
    Xt = pre.transform(X_train)
    Xv = pre.transform(X_val)

    if pre.stats_:
        st = pre.stats_
        log(f"[DROP] all_missing={st.all_missing} high_missing={st.high_missing} constant={st.constant} -> total_dropped={st.total_dropped}")

    # Optional KFold target encoding on high-card categorical cols
    te = None
    if args.use_target_encoding and len(pre.cat_cols_) > 0:
        high_card = []
        for c in pre.cat_cols_:
            nun = Xt[c].nunique(dropna=True)
            if nun >= args.te_min_unique:
                high_card.append(c)
        if high_card:
            log(f"[TE] Using KFold target encoding for {len(high_card)} high-card cols: {high_card[:6]}{'...' if len(high_card)>6 else ''}")
            te = KFoldTargetEncoder(
                cols=high_card, n_splits=args.te_splits, seed=args.seed, smoothing=args.te_smoothing
            )
            Xt = te.fit_transform(Xt, y_train)
            Xv = te.transform(Xv)
        else:
            log("[TE] No categorical columns met high-card threshold; skipping TE.")

    cat_cols = [c for c in Xt.columns if str(Xt[c].dtype) == "category"]
    log(f"[INFO] Train: {Xt.shape} | Val: {Xv.shape}")
    log(f"[INFO] Cat cols: {len(cat_cols)} | Threads: {args.threads}")
    log(f"[INFO] Engineered cols: {len(pre.engineered_cols_)}")

    # LightGBM model
    lgbm = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,             # <--- UPDATED
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        n_jobs=args.threads,
        random_state=args.seed,
        force_row_wise=args.force_row_wise,
        verbosity=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
        make_eta_logger(total_estimators=args.n_estimators, log_every=args.log_every),
    ]

    log("[TRAIN] Fitting LightGBM (Anti-Overfit Mode)...")
    t0 = time.time()
    lgbm.fit(
        Xt, y_train,
        eval_set=[(Xv, y_val)],
        eval_metric="rmse",
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=callbacks,
    )
    train_sec = time.time() - t0

    best_iter = int(getattr(lgbm, "best_iteration_", 0) or 0)
    if best_iter <= 0:
        best_iter = args.n_estimators

    log(f"[TRAIN] done in {int(train_sec//60)}m {int(train_sec%60)}s | best_iter={best_iter}")

    # Predict train + val to check overfitting
    pred_train_log = np.asarray(lgbm.predict(Xt, num_iteration=best_iter), dtype=float)
    pred_val_log = np.asarray(lgbm.predict(Xv, num_iteration=best_iter), dtype=float)

    m_train = eval_reg(y_train, pred_train_log)
    m_val = eval_reg(y_val, pred_val_log)

    log("===== OVERFITTING CHECK (Train vs Val) =====")
    log(f"[TRAIN] RMSE_LOG: {m_train['rmse_log']:.6f} | R2_LOG: {m_train['r2_log']:.6f} | RMSE_TL: {m_train['rmse_tl']:,.2f}")
    log(f"[VAL  ] RMSE_LOG: {m_val['rmse_log']:.6f} | R2_LOG: {m_val['r2_log']:.6f} | RMSE_TL: {m_val['rmse_tl']:,.2f}")

    # Calculate overfitting index
    overfit_idx = m_val['rmse_log'] / m_train['rmse_log'] if m_train['rmse_log'] > 0 else 0
    log(f"[METRIC] Overfitting Index (Val/Train): {overfit_idx:.4f}")

    log("===== FINAL EVAL (VAL) =====")
    log(f"RMSE_LOG: {m_val['rmse_log']:.6f} | R2_LOG: {m_val['r2_log']:.6f}")
    log(f"RMSE_TL : {m_val['rmse_tl']:,.2f} | R2_TL : {m_val['r2_tl']:.6f} | MAE_TL: {m_val['mae_tl']:,.2f}")

    # Residual diagnostics (log space)
    resid = pred_val_log - np.asarray(y_val, dtype=float)
    qs = np.quantile(resid, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    log(
        "Residuals (log space) quantiles: "
        f"min={qs[0]:.4f}, p01={qs[1]:.4f}, p05={qs[2]:.4f}, median={qs[3]:.4f}, "
        f"p95={qs[4]:.4f}, p99={qs[5]:.4f}, max={qs[6]:.4f}"
    )
    log(f"Residual mean={float(np.mean(resid)):.6f}, std={float(np.std(resid)):.6f}")

    # Top errors (positional indexing fix)
    if args.save_top_errors:
        y_val_tl_true = np.expm1(np.asarray(y_val, dtype=float))
        y_val_tl_pred = np.expm1(pred_val_log)
        abs_err_tl = np.abs(y_val_tl_true - y_val_tl_pred)
        topk_idx = np.argsort(-abs_err_tl)[: args.topk]

        top_df = pd.DataFrame({
            "row_id": id_val[topk_idx],
            "y_true_tl": y_val_tl_true[topk_idx],
            "y_pred_tl": y_val_tl_pred[topk_idx],
            "abs_err_tl": abs_err_tl[topk_idx],
            "y_true_log": np.asarray(y_val, dtype=float)[topk_idx],
            "y_pred_log": pred_val_log[topk_idx],
            "resid_log": resid[topk_idx],
        })
        top_path = os.path.join(args.outdir, "model5_top_errors.csv")
        top_df.to_csv(top_path, index=False, encoding="utf-8")
        log(f"[SAVE] top errors: {top_path}")

    # Feature importance
    booster = lgbm.booster_
    imp_gain = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")
    feat_names = booster.feature_name()

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "gain": imp_gain,
        "split": imp_split,
    }).sort_values("gain", ascending=False)

    imp_path = os.path.join(args.outdir, "model5_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")
    log(f"[SAVE] feature importance: {imp_path}")

    # ----------------------------
    # Business thresholds (learned on TRAIN predictions)
    # ----------------------------
    thresholds = learn_thresholds_from_train(
        y_train_log=np.asarray(y_train, dtype=float),
        pred_train_log=pred_train_log,
        q_low=args.deal_q_low,
        q_high=args.deal_q_high,
    )
    t_buy = thresholds["t_buy"]
    t_sell = thresholds["t_sell"]
    log(f"[BUSINESS] Learned thresholds: t_buy={t_buy:.3f} (deal) | t_sell={t_sell:.3f} (expensive)")

    thr_path = os.path.join(args.outdir, "model5_business_thresholds.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] thresholds: {thr_path}")

    # Export validation predictions with labels
    if args.export_val_predictions:
        fair_val_tl = np.expm1(pred_val_log)
        listing_val_tl = np.expm1(np.asarray(y_val, dtype=float))

        discount_pct = (fair_val_tl - listing_val_tl) / np.clip(fair_val_tl, 1.0, np.inf)
        labels = [
            classify_investment(listing_tl=float(listing_val_tl[i]), fair_tl=float(fair_val_tl[i]), t_buy=t_buy, t_sell=t_sell)
            for i in range(len(fair_val_tl))
        ]

        val_pred_df = pd.DataFrame({
            "row_id": id_val,
            "listing_price_tl": listing_val_tl,
            "fair_value_tl": fair_val_tl,
            "discount_pct": discount_pct,   # + => cheaper than fair
            "label": labels,
            "pred_log": pred_val_log,
        })

        out_pred_path = os.path.join(args.outdir, "model5_val_predictions.csv")
        val_pred_df.to_csv(out_pred_path, index=False, encoding="utf-8")
        log(f"[SAVE] val predictions: {out_pred_path}")

    # Save bundle (production use)
    bundle = {
        "preprocessor": pre,
        "target_encoder": te,   # can be None
        "model": lgbm,
        "best_iter": best_iter,
        "business_thresholds": thresholds,
    }
    bundle_path = os.path.join(args.outdir, "model5_lgbm_business_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    log(f"[SAVE] {bundle_path}")

    # Save meta
    meta = {
        "seed": args.seed,
        "threads": args.threads,
        "loan_filter_enabled": bool(did_filter),
        "loan_filter_column": loan_col if did_filter else None,
        "drop_missing_over": args.drop_missing_over,
        "clip_quantiles": [args.clip_low_q, args.clip_high_q],
        "dropped_columns": pre.drop_cols_,
        "categorical_columns": pre.cat_cols_,
        "engineered_columns": pre.engineered_cols_,
        "feature_columns": pre.feature_cols_,
        "preprocess_stats": asdict(pre.stats_) if pre.stats_ else None,
        "best_iter": best_iter,
        "metrics_train": m_train,
        "metrics_val": m_val,
        "business_thresholds": thresholds,
        "params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.lr,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample,
            "reg_lambda": args.reg_lambda,
            "reg_alpha": args.reg_alpha,
            "early_stopping_rounds": args.early_stopping_rounds,
            "force_row_wise": bool(args.force_row_wise),
            "use_target_encoding": bool(args.use_target_encoding),
            "te_min_unique": args.te_min_unique,
            "te_splits": args.te_splits,
            "te_smoothing": args.te_smoothing,
            "deal_q_low": args.deal_q_low,
            "deal_q_high": args.deal_q_high,
        },
    }

    meta_path = os.path.join(args.outdir, "model5_lgbm_business_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] {meta_path}")

    log("DONE")


if __name__ == "__main__":
    main()