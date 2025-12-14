# model1.py
# CatBoost (House Price) - full version: preprocessing + parallel training + save bundle
# Usage (PowerShell):
#   python model1.py --data .\hackathon_train_set.csv --outdir .\models --threads 10

import os
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "10")

import re
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from catboost import CatBoostRegressor, Pool


def rmse(y_true, y_pred):
    # sklearn >= 1.4 introduced root_mean_squared_error; sklearn 1.6+ removed squared= in MSE
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def robust_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if "Price" in df.columns and len(df.columns) > 3:
                return df
        except Exception as e:
            last_err = e
    if last_err:
        print("[WARN] robust_read_csv fallbacks exhausted, using default read_csv.")
    return pd.read_csv(path, encoding="utf-8-sig")


def clean_price_to_float(x):
    if pd.isna(x):
        return np.nan
    digits = re.sub(r"[^\d]", "", str(x))
    return float(digits) if digits else np.nan


def find_loan_col(df: pd.DataFrame):
    for c in df.columns:
        cl = c.lower()
        if "available for loan" in cl or "loan" in cl or "kredi" in cl:
            return c
    return None


def filter_loan_eligible(df: pd.DataFrame) -> pd.DataFrame:
    loan_col = find_loan_col(df)
    if loan_col is None:
        return df

    non_null_rate = df[loan_col].notna().mean()
    if non_null_rate < 0.05:
        return df

    s = df[loan_col]
    if pd.api.types.is_numeric_dtype(s):
        return df[s == 1].copy()

    s2 = s.astype(str).str.strip().str.lower()
    yes_vals = {"yes", "true", "1", "evet", "uygun", "eligible"}
    return df[s2.isin(yes_vals)].copy()


def add_date_parts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
    df[col + "_year"] = dt.dt.year
    df[col + "_month"] = dt.dt.month
    df[col + "_day"] = dt.dt.day
    df[col + "_dow"] = dt.dt.dayofweek
    return df


class Preprocessor:
    """
    - Converts empty strings / NULL-like tokens to NaN
    - Adds date parts (if exists)
    - Drops: all-missing, >threshold missing, constants
    - Fills: numeric median, cat "Unknown"
    """

    def __init__(self, drop_missing_over=0.50):
        self.drop_missing_over = float(drop_missing_over)
        self.drop_cols_ = []
        self.cat_cols_ = []
        self.numeric_medians_ = {}
        self.feature_cols_ = []
        self.debug_drop_counts_ = {}

    @staticmethod
    def _standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
        # treat blanks and common tokens as missing
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df = df.replace(
            {"None": np.nan, "NONE": np.nan, "null": np.nan, "NULL": np.nan, "NaN": np.nan, "nan": np.nan}
        )
        return df

    def fit(self, X: pd.DataFrame):
        df = self._standardize_missing(X.copy())

        # date-derived
        for dcol in ["Adrtisement Date", "Pick Up Data Time"]:
            if dcol in df.columns:
                df = add_date_parts(df, dcol)

        if ("Adrtisement Date" in df.columns) and ("Pick Up Data Time" in df.columns):
            ad = pd.to_datetime(df["Adrtisement Date"], errors="coerce", dayfirst=True)
            pu = pd.to_datetime(df["Pick Up Data Time"], errors="coerce", dayfirst=True)
            df["listing_age_days"] = (pu - ad).dt.days

        df = df.drop(columns=["Adrtisement Date", "Pick Up Data Time"], errors="ignore")

        all_missing = [c for c in df.columns if df[c].isna().all()]
        miss_rate = df.isna().mean()
        high_missing = miss_rate[miss_rate > self.drop_missing_over].index.tolist()
        nun = df.nunique(dropna=False)
        const_cols = nun[nun <= 1].index.tolist()

        self.debug_drop_counts_ = {
            "all_missing": len(all_missing),
            "high_missing": len(high_missing),
            "constant": len(const_cols),
        }

        self.drop_cols_ = sorted(set(all_missing + high_missing + const_cols))
        df = df.drop(columns=self.drop_cols_, errors="ignore")

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
        df = self._standardize_missing(X.copy())

        for dcol in ["Adrtisement Date", "Pick Up Data Time"]:
            if dcol in df.columns:
                df = add_date_parts(df, dcol)

        if ("Adrtisement Date" in df.columns) and ("Pick Up Data Time" in df.columns):
            ad = pd.to_datetime(df["Adrtisement Date"], errors="coerce", dayfirst=True)
            pu = pd.to_datetime(df["Pick Up Data Time"], errors="coerce", dayfirst=True)
            df["listing_age_days"] = (pu - ad).dt.days

        df = df.drop(columns=["Adrtisement Date", "Pick Up Data Time"], errors="ignore")
        df = df.drop(columns=self.drop_cols_, errors="ignore")

        for c in self.feature_cols_:
            if c not in df.columns:
                df[c] = np.nan

        df = df[self.feature_cols_].copy()

        for c in df.columns:
            if c in self.cat_cols_:
                df[c] = df[c].fillna("Unknown").astype(str)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.numeric_medians_.get(c, 0.0))

        return df


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

    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.04)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--l2", type=float, default=8.0)
    ap.add_argument("--min_leaf", type=int, default=30)
    ap.add_argument("--rsm", type=float, default=0.8)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--od_wait", type=int, default=300)
    ap.add_argument("--verbose", type=int, default=200)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("os.cpu_count() =", os.cpu_count())
    print(f"[LOAD] Reading: {args.data}")

    df = robust_read_csv(args.data)
    if "Price" not in df.columns:
        raise ValueError("Price column not found. Check separator/encoding.")

    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()

    before = len(df)
    df = filter_loan_eligible(df)
    after = len(df)
    if after != before:
        print(f"[FILTER] Loan eligible: {before} -> {after}")

    y = df["Price_num"].astype(float)
    lo = y.quantile(args.clip_low_q)
    hi = y.quantile(args.clip_high_q)
    y_clip = y.clip(lo, hi)
    y_log = np.log1p(y_clip)

    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=args.test_size, random_state=args.seed
    )

    pre = Preprocessor(drop_missing_over=args.drop_missing_over).fit(X_train)
    Xt = pre.transform(X_train)
    Xv = pre.transform(X_val)

    cat_cols = pre.cat_cols_
    cat_idx = [Xt.columns.get_loc(c) for c in cat_cols if c in Xt.columns]

    print(f"[INFO] Train shape: {Xt.shape} | Val shape: {Xv.shape}")
    print(f"[INFO] Cat cols: {len(cat_cols)} | Threads: {args.threads}")
    print(f"[DROP] all_missing={pre.debug_drop_counts_.get('all_missing',0)} "
          f"high_missing={pre.debug_drop_counts_.get('high_missing',0)} "
          f"constant={pre.debug_drop_counts_.get('constant',0)} "
          f"-> total_dropped={len(pre.drop_cols_)}")

    train_pool = Pool(Xt, y_train, cat_features=cat_idx)
    val_pool = Pool(Xv, y_val, cat_features=cat_idx)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=args.iterations,
        learning_rate=args.lr,
        depth=args.depth,
        l2_leaf_reg=args.l2,
        min_data_in_leaf=args.min_leaf,
        random_strength=2.0,
        rsm=args.rsm,
        bootstrap_type="Bernoulli",
        subsample=args.subsample,
        od_type="Iter",
        od_wait=args.od_wait,
        random_seed=args.seed,
        thread_count=args.threads,
        task_type="CPU",
        verbose=args.verbose,
        allow_writing_files=False
    )

    print("[TRAIN] Fitting CatBoost...")
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    pred_log = model.predict(val_pool)

    rmse_log = rmse(y_val, pred_log)
    r2_log = r2_score(y_val, pred_log)

    pred_tl = np.expm1(pred_log)
    y_val_tl = np.expm1(y_val)

    rmse_tl = rmse(y_val_tl, pred_tl)
    r2_tl = r2_score(y_val_tl, pred_tl)
    mae_tl = mean_absolute_error(y_val_tl, pred_tl)

    print("\n===== EVAL (IMPORTANT) =====")
    print(f"RMSE_LOG: {rmse_log:.6f} | R2_LOG: {r2_log:.6f}")
    print(f"RMSE_TL : {rmse_tl:,.2f} | R2_TL : {r2_tl:.6f} | MAE_TL: {mae_tl:,.2f}")
    print(f"Best iteration: {model.get_best_iteration()}")

    cbm_path = os.path.join(args.outdir, "catboost_model1.cbm")
    model.save_model(cbm_path)

    bundle_path = os.path.join(args.outdir, "model1_bundle.joblib")
    joblib.dump({"preprocessor": pre, "model": model}, bundle_path)

    meta = {
        "threads": args.threads,
        "seed": args.seed,
        "drop_missing_over": args.drop_missing_over,
        "clip_quantiles": [args.clip_low_q, args.clip_high_q],
        "feature_columns": pre.feature_cols_,
        "categorical_columns": pre.cat_cols_,
        "dropped_columns": pre.drop_cols_,
        "metrics": {
            "rmse_log": float(rmse_log),
            "r2_log": float(r2_log),
            "rmse_tl": float(rmse_tl),
            "r2_tl": float(r2_tl),
            "mae_tl": float(mae_tl),
        },
        "catboost_params": {
            "iterations": args.iterations,
            "learning_rate": args.lr,
            "depth": args.depth,
            "l2_leaf_reg": args.l2,
            "min_data_in_leaf": args.min_leaf,
            "rsm": args.rsm,
            "subsample": args.subsample,
            "od_wait": args.od_wait
        }
    }
    meta_path = os.path.join(args.outdir, "model1_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {cbm_path}")
    print(f"[SAVE] {bundle_path}")
    print(f"[SAVE] {meta_path}")


if __name__ == "__main__":
    main()
