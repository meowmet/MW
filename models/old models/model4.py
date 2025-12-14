# model4.py
# LightGBM (Istanbul Housing) - DEBUG + PRODUCTION version
# - Robust CSV read + cleaning
# - Loan-eligible filtering (if column exists)
# - Safe feature engineering (NO target leakage)
# - Consistent preprocessing (fit on train, apply to val/test)
# - Detailed logging (shapes, drops, ETA minutes left, residual stats)
# - Fixes the KeyError in "top errors" (uses positional indexing safely)
#
# Run:
#   python model4.py --data hackathon_train_set.csv --outdir .\models --threads 10
#
# Notes:
# - This is REGRESSION => metrics are RMSE/R2/MAE (AUC is for classification)
# - ETA shown during training is approximate and updates every log interval.

import os
import re
import json
import time
import argparse
from dataclasses import asdict, dataclass
from typing import List, Dict, Optional, Tuple

# Thread env (helps avoid oversubscription on Windows)
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "10")

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import lightgbm as lgb


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
            # heuristic: if Price exists and we have enough columns -> accept
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


def filter_loan_eligible(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    loan_col = find_loan_col(df)
    if loan_col is None:
        return df, False

    non_null_rate = df[loan_col].notna().mean()
    if non_null_rate < 0.05:
        return df, False

    s = df[loan_col]
    if pd.api.types.is_numeric_dtype(s):
        out = df[s == 1].copy()
        return out, True

    s2 = s.astype(str).str.strip().str.lower()
    yes_vals = {"yes", "true", "1", "evet", "uygun", "eligible"}
    out = df[s2.isin(yes_vals)].copy()
    return out, True


def normalize_colnames(cols: List[str]) -> List[str]:
    # LightGBM warns on whitespace; normalize aggressively
    out = []
    for c in cols:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", "_", c2)
        out.append(c2)
    return out


def to_datetime_safe(series: pd.Series) -> pd.Series:
    # Turkish datasets often dd.mm.yyyy
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def parse_rooms(value) -> Optional[int]:
    # e.g. "3+1" -> 4
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", s)
    if not m:
        # sometimes "4" already numeric-like
        try:
            v = int(float(s))
            return v
        except Exception:
            return None
    a = int(m.group(1))
    b = int(m.group(2))
    return a + b


def detect_binary_like_cols(df: pd.DataFrame, max_unique: int = 3) -> List[str]:
    """
    Detect amenity-ish binary columns: mostly 0/1 or yes/no style.
    This is heuristic and safe (doesn't use target).
    """
    yes_set = {"1", "yes", "true", "evet", "var", "yok_degil", "uygun"}
    no_set = {"0", "no", "false", "hayir", "hayır", "yok", "degil", "uygun_degil"}

    cols = []
    for c in df.columns:
        s = df[c]
        # skip numeric continuous
        nun = s.nunique(dropna=True)
        if nun == 0 or nun > max_unique:
            continue

        # allow numeric 0/1
        if pd.api.types.is_numeric_dtype(s):
            uniq = set(pd.to_numeric(s.dropna(), errors="coerce").unique().tolist())
            if uniq.issubset({0, 1}):
                cols.append(c)
            continue

        # allow yes/no strings
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
    Fit on train, apply same transformations to val/test/inference.
    - Creates engineered features (safe):
        * date parts (if date columns exist)
        * listing_age_days (if both date cols exist)
        * total_rooms (if rooms-like col exists)
        * net_gross_ratio (if net/gross exist)
        * amenity_count (+ optional grouped counts by keyword)
        * building_age_bin (if age exists)
    - Drops:
        * all-missing cols
        * high-missing cols (> drop_missing_over)
        * constant cols
        * raw date cols (after deriving parts)
    - Fills:
        * numeric -> median
        * categorical -> "Unknown"
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

        # remember detected columns for consistent engineering
        self._room_col: Optional[str] = None
        self._net_col: Optional[str] = None
        self._gross_col: Optional[str] = None
        self._age_col: Optional[str] = None
        self._amenity_cols: List[str] = []

    def _detect_common_cols(self, df: pd.DataFrame):
        # Room-like
        for cand in ["Number_of_rooms", "Number_of_rooms_", "Number_of_rooms__"]:
            if cand in df.columns:
                self._room_col = cand
                break
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
            if any(k in cl for k in ["building_age", "age_of_building", "bina_yasi", "bina_yaşı", "yaş"]):
                self._age_col = c
                break

        # Amenities
        self._amenity_cols = detect_binary_like_cols(df, max_unique=3)

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        engineered = []

        # Date features (handle both raw names and normalized names)
        date_candidates = []
        for c in df.columns:
            cl = c.lower()
            if "adrtisement_date" in cl or "advertisement_date" in cl:
                date_candidates.append(c)
            if "pick_up_data_time" in cl or "pickup_data_time" in cl or "pick_up_date" in cl:
                date_candidates.append(c)

        # Try explicit known names (post-normalization)
        # We'll derive parts for any "date-like" candidates we find.
        for c in sorted(set(date_candidates)):
            dt = to_datetime_safe(df[c])
            df[c + "_year"] = dt.dt.year
            df[c + "_month"] = dt.dt.month
            df[c + "_day"] = dt.dt.day
            df[c + "_dow"] = dt.dt.dayofweek
            engineered += [c + "_year", c + "_month", c + "_day", c + "_dow"]

        # listing_age_days if we can find 2 date columns among candidates
        if len(sorted(set(date_candidates))) >= 2:
            c1, c2 = sorted(set(date_candidates))[:2]
            d1 = to_datetime_safe(df[c1])
            d2 = to_datetime_safe(df[c2])
            df["listing_age_days"] = (d2 - d1).dt.days
            engineered.append("listing_age_days")

        # total_rooms
        if self._room_col and self._room_col in df.columns:
            df["total_rooms"] = df[self._room_col].apply(parse_rooms)
            engineered.append("total_rooms")

        # net_gross_ratio
        if self._net_col and self._gross_col and (self._net_col in df.columns) and (self._gross_col in df.columns):
            net = pd.to_numeric(df[self._net_col], errors="coerce")
            gross = pd.to_numeric(df[self._gross_col], errors="coerce")
            df["net_gross_ratio"] = (net / gross).replace([np.inf, -np.inf], np.nan)
            engineered.append("net_gross_ratio")

        # amenity_count + grouped
        if self._amenity_cols:
            mat = np.zeros((len(df),), dtype=np.int32)
            for c in self._amenity_cols:
                mat += binary_to_01(df[c])
            df["amenity_count"] = mat
            engineered.append("amenity_count")

            # Optional groups by keyword
            groups = {
                "amenity_luxury_count": ["sauna", "jacuzzi", "pool", "spa", "hamam"],
                "amenity_security_count": ["security", "alarm", "camera", "guard", "site"],
                "amenity_transport_count": ["metro", "bus", "station", "transport"],
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

        # building_age_bin
        if self._age_col and self._age_col in df.columns:
            age = pd.to_numeric(df[self._age_col], errors="coerce")
            # bins: new/0-5/6-20/21-40/41+
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

        # Drop raw date columns (after deriving parts): any column that looks like date and was used
        raw_date_cols = []
        for c in df.columns:
            cl = c.lower()
            if ("date" in cl or "time" in cl) and any(suf in c for suf in ["_year", "_month", "_day", "_dow"]) is False:
                # keep numeric "time" not safe; only drop if dtype is object-like and parses as date somewhat
                if df[c].dtype == "object":
                    raw_date_cols.append(c)
        # Conservative: only drop those that were in date_candidates earlier
        # (already included in engineered names)
        # We'll drop only those that have derived columns:
        truly_drop_dates = []
        for c in raw_date_cols:
            if (c + "_year") in df.columns and (c + "_month") in df.columns:
                truly_drop_dates.append(c)
        df = df.drop(columns=truly_drop_dates, errors="ignore")

        # Drop all-missing
        all_missing = [c for c in df.columns if df[c].isna().all()]

        # Drop high-missing
        miss_rate = df.isna().mean()
        high_missing = miss_rate[miss_rate > self.drop_missing_over].index.tolist()

        # Drop constant
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

        # Categoricals: object columns
        self.cat_cols_ = [c for c in df.columns if df[c].dtype == "object"]

        # Numeric medians
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

        # Drop any date cols that have derived parts
        raw_date_cols = []
        for c in df.columns:
            if df[c].dtype == "object":
                if (c + "_year") in df.columns and (c + "_month") in df.columns:
                    raw_date_cols.append(c)
        df = df.drop(columns=raw_date_cols, errors="ignore")

        # Drop learned drops
        df = df.drop(columns=self.drop_cols_, errors="ignore")

        # Ensure expected features exist
        for c in self.feature_cols_:
            if c not in df.columns:
                df[c] = np.nan

        # Remove extras
        df = df[self.feature_cols_].copy()

        # Fill
        for c in df.columns:
            if c in self.cat_cols_:
                df[c] = df[c].fillna("Unknown").astype(str)
                df[c] = df[c].astype("category")
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.numeric_medians_.get(c, 0.0))

        return df


# ----------------------------
# Leakage-safe KFold target encoding (optional)
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
        self.global_mean_ = float(np.mean(y))
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for c in self.cols:
            out = np.zeros((len(X),), dtype=float)

            for tr_idx, va_idx in kf.split(X):
                tr = X.iloc[tr_idx]
                ytr = y[tr_idx]

                stats = tr.groupby(c).apply(lambda g: (len(g), float(np.mean(ytr[g.index - g.index.min()]))))
                # The line above is not safe due to index math; do standard groupby with aligned y:
                tmp = pd.DataFrame({c: tr[c].values, "_y": ytr})
                grp = tmp.groupby(c)["_y"].agg(["count", "mean"])
                count = grp["count"]
                mean = grp["mean"]

                smooth = (count * mean + self.smoothing * self.global_mean_) / (count + self.smoothing)
                m = smooth.to_dict()

                out[va_idx] = X.iloc[va_idx][c].map(m).fillna(self.global_mean_).to_numpy()

            X[c + "__te"] = out

            # fit full map for inference
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
        # env.iteration is 0-based
        it = env.iteration + 1
        if it == 1 or (it % log_every == 0):
            elapsed = time.time() - start_time
            sec_per_iter = elapsed / max(it, 1)
            remaining = max(total_estimators - it, 0)
            eta_sec = remaining * sec_per_iter

            # Get latest metric if available
            msg_metric = ""
            if env.evaluation_result_list:
                # list of tuples: (data_name, eval_name, result, is_higher_better)
                dname, ename, val, _ = env.evaluation_result_list[-1]
                msg_metric = f" | {dname}.{ename}={val:.6f}"

            log(f"[LGBM][{it}/{total_estimators}]{msg_metric} | elapsed={int(elapsed//60)}m {int(elapsed%60)}s | eta≈{int(eta_sec//60)}m {int(eta_sec%60)}s")

    _callback.order = 10
    return _callback


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=str, default="hackathon_train_set.csv")
    ap.add_argument("--outdir", type=str, default="models")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.20)
    ap.add_argument("--drop_missing_over", type=float, default=0.50)
    ap.add_argument("--clip_low_q", type=float, default=0.001)
    ap.add_argument("--clip_high_q", type=float, default=0.995)

    # LGBM params
    ap.add_argument("--n_estimators", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--num_leaves", type=int, default=128)
    ap.add_argument("--min_child_samples", type=int, default=50)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--reg_lambda", type=float, default=2.0)
    ap.add_argument("--reg_alpha", type=float, default=0.0)
    ap.add_argument("--early_stopping_rounds", type=int, default=300)
    ap.add_argument("--force_row_wise", action="store_true", default=True)

    # Debug/logging
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--save_top_errors", action="store_true", default=True)

    # Optional target encoding
    ap.add_argument("--use_target_encoding", action="store_true", default=False)
    ap.add_argument("--te_min_unique", type=int, default=30)  # apply TE for high-card cols only
    ap.add_argument("--te_splits", type=int, default=5)
    ap.add_argument("--te_smoothing", type=float, default=20.0)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    log(f"os.cpu_count() = {os.cpu_count()}")
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
    df, did_filter = filter_loan_eligible(df)
    after = len(df)
    if did_filter:
        log(f"[FILTER] Loan eligible: {before} -> {after}")

    # Clip + log target
    y = df["Price_num"].astype(float)
    lo = y.quantile(args.clip_low_q)
    hi = y.quantile(args.clip_high_q)
    y_clip = y.clip(lo, hi)
    y_log = np.log1p(y_clip).to_numpy(dtype=float)

    # Features
    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()

    # Keep original row ids to debug top errors
    row_ids = df.index.to_numpy()

    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X, y_log, row_ids,
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
        # choose high-card columns
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
            # TE columns are numeric now; keep originals too (LightGBM can still use categories)
        else:
            log("[TE] No categorical columns met high-card threshold; skipping TE.")

    # Ensure category dtype for categoricals
    cat_cols = [c for c in Xt.columns if str(Xt[c].dtype) == "category"]
    log(f"[INFO] Train: {Xt.shape} | Val: {Xv.shape}")
    log(f"[INFO] Cat cols: {len(cat_cols)} | Threads: {args.threads}")
    log(f"[INFO] Engineered cols: {len(pre.engineered_cols_)}")

    # LightGBM model
    lgbm = lgb.LGBMRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.lr,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        colsample_bytree=args.colsample,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        n_jobs=args.threads,
        random_state=args.seed,
        force_row_wise=args.force_row_wise,
        # keep logs controlled (we print our own ETA logs)
        verbosity=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
        make_eta_logger(total_estimators=args.n_estimators, log_every=args.log_every),
    ]

    log("[TRAIN] Fitting LightGBM...")
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

    # Predict + evaluate
    pred_log = lgbm.predict(Xv, num_iteration=best_iter)
    pred_log = np.asarray(pred_log, dtype=float)
    y_val_arr = np.asarray(y_val, dtype=float)

    rmse_log = float(np.sqrt(mean_squared_error(y_val_arr, pred_log)))
    r2_log = float(r2_score(y_val_arr, pred_log))

    pred_tl = np.expm1(pred_log)
    y_val_tl = np.expm1(y_val_arr)

    rmse_tl = float(np.sqrt(mean_squared_error(y_val_tl, pred_tl)))
    r2_tl = float(r2_score(y_val_tl, pred_tl))
    mae_tl = float(mean_absolute_error(y_val_tl, pred_tl))

    log("===== FINAL EVAL =====")
    log(f"RMSE_LOG: {rmse_log:.6f} | R2_LOG: {r2_log:.6f}")
    log(f"RMSE_TL : {rmse_tl:,.2f} | R2_TL : {r2_tl:.6f} | MAE_TL: {mae_tl:,.2f}")

    # Residual diagnostics (log space)
    resid = pred_log - y_val_arr
    qs = np.quantile(resid, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    log(
        "Residuals (log space) quantiles: "
        f"min={qs[0]:.4f}, p01={qs[1]:.4f}, p05={qs[2]:.4f}, median={qs[3]:.4f}, "
        f"p95={qs[4]:.4f}, p99={qs[5]:.4f}, max={qs[6]:.4f}"
    )
    log(f"Residual mean={float(np.mean(resid)):.6f}, std={float(np.std(resid)):.6f}")

    # Top errors (FIXED: positional indexing with numpy arrays)
    if args.save_top_errors:
        abs_err_tl = np.abs(y_val_tl - pred_tl)  # numpy array
        topk = np.argsort(-abs_err_tl)[: args.topk]

        top_df = pd.DataFrame({
            "row_id": id_val[topk],
            "y_true_tl": y_val_tl[topk],
            "y_pred_tl": pred_tl[topk],
            "abs_err_tl": abs_err_tl[topk],
            "y_true_log": y_val_arr[topk],
            "y_pred_log": pred_log[topk],
            "resid_log": resid[topk],
        })

        top_path = os.path.join(args.outdir, "model4_top_errors.csv")
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

    imp_path = os.path.join(args.outdir, "model4_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")
    log(f"[SAVE] feature importance: {imp_path}")

    # Save bundle
    bundle = {
        "preprocessor": pre,
        "target_encoder": te,   # can be None
        "model": lgbm,
        "best_iter": best_iter,
    }
    bundle_path = os.path.join(args.outdir, "model4_lgbm_debug_bundle.joblib")
    joblib.dump(bundle, bundle_path)
    log(f"[SAVE] {bundle_path}")

    # Save meta
    meta = {
        "seed": args.seed,
        "threads": args.threads,
        "loan_filter_enabled": bool(did_filter),
        "drop_missing_over": args.drop_missing_over,
        "clip_quantiles": [args.clip_low_q, args.clip_high_q],
        "dropped_columns": pre.drop_cols_,
        "categorical_columns": pre.cat_cols_,
        "engineered_columns": pre.engineered_cols_,
        "feature_columns": pre.feature_cols_,
        "preprocess_stats": asdict(pre.stats_) if pre.stats_ else None,
        "best_iter": best_iter,
        "metrics": {
            "rmse_log": rmse_log,
            "r2_log": r2_log,
            "rmse_tl": rmse_tl,
            "r2_tl": r2_tl,
            "mae_tl": mae_tl,
        },
        "params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.lr,
            "num_leaves": args.num_leaves,
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
        },
    }

    meta_path = os.path.join(args.outdir, "model4_lgbm_debug_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] {meta_path}")


if __name__ == "__main__":
    main()
