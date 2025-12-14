# model2_lgbm.py
# Usage:
#   python model2_lgbm.py --data hackathon_train_set.csv --outdir models --threads 10

import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import lightgbm as lgb


# -----------------------------
# IO + cleaning
# -----------------------------
def robust_read_csv(path: str) -> pd.DataFrame:
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if "Price" in df.columns:
                return df
        except Exception:
            pass
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
    if df[loan_col].notna().mean() < 0.05:
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


def clip_target(y: pd.Series, low_q=0.001, high_q=0.995) -> pd.Series:
    lo = y.quantile(low_q)
    hi = y.quantile(high_q)
    return y.clip(lo, hi)


# -----------------------------
# Metrics (sklearn-compatible)
# -----------------------------
def rmse(y_true, y_pred) -> float:
    """
    Compatible with old + new scikit-learn versions.
    - sklearn >= 1.4: root_mean_squared_error exists
    - sklearn 1.6+: mean_squared_error no longer supports squared=...
    """
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# Preprocessor (same spirit as model1)
# -----------------------------
class Preprocessor:
    def __init__(self, drop_missing_over=0.50):
        self.drop_missing_over = float(drop_missing_over)
        self.drop_cols_ = []
        self.cat_cols_ = []
        self.numeric_medians_ = {}
        self.feature_cols_ = []

    def fit(self, X: pd.DataFrame):
        df = X.copy()

        # date features if present
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
        df = X.copy()

        for dcol in ["Adrtisement Date", "Pick Up Data Time"]:
            if dcol in df.columns:
                df = add_date_parts(df, dcol)

        if ("Adrtisement Date" in df.columns) and ("Pick Up Data Time" in df.columns):
            ad = pd.to_datetime(df["Adrtisement Date"], errors="coerce", dayfirst=True)
            pu = pd.to_datetime(df["Pick Up Data Time"], errors="coerce", dayfirst=True)
            df["listing_age_days"] = (pu - ad).dt.days

        df = df.drop(columns=["Adrtisement Date", "Pick Up Data Time"], errors="ignore")
        df = df.drop(columns=self.drop_cols_, errors="ignore")

        # add missing columns that existed in train
        for c in self.feature_cols_:
            if c not in df.columns:
                df[c] = np.nan

        # drop extra columns and reorder
        df = df[self.feature_cols_].copy()

        # fill
        for c in df.columns:
            if c in self.cat_cols_:
                df[c] = df[c].fillna("Unknown").astype(str)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.numeric_medians_.get(c, 0.0))

        return df


def make_lgbm_ready(df: pd.DataFrame, cat_cols):
    # LightGBM can use pandas "category" dtype for categorical features
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


# -----------------------------
# Train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="hackathon_train_set.csv")
    ap.add_argument("--outdir", type=str, default="models")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--drop_missing_over", type=float, default=0.50)
    ap.add_argument("--clip_low_q", type=float, default=0.001)
    ap.add_argument("--clip_high_q", type=float, default=0.995)

    # LGBM params
    ap.add_argument("--n_estimators", type=int, default=20000)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--num_leaves", type=int, default=128)
    ap.add_argument("--min_child_samples", type=int, default=50)
    ap.add_argument("--subsample", type=float, default=0.8)         # bagging_fraction
    ap.add_argument("--colsample", type=float, default=0.8)         # feature_fraction
    ap.add_argument("--reg_lambda", type=float, default=2.0)        # lambda_l2
    ap.add_argument("--reg_alpha", type=float, default=0.0)         # lambda_l1
    ap.add_argument("--early_stopping", type=int, default=300)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    cpu = os.cpu_count() or args.threads
    args.threads = min(args.threads, cpu)

    print("os.cpu_count() =", os.cpu_count())
    print(f"[LOAD] {args.data}")
    df = robust_read_csv(args.data)

    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()

    before = len(df)
    df = filter_loan_eligible(df)
    after = len(df)
    if after != before:
        print(f"[FILTER] Loan eligible: {before} -> {after}")

    y = df["Price_num"].astype(float)
    y_clip = clip_target(y, args.clip_low_q, args.clip_high_q)
    y_log = np.log1p(y_clip)

    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=args.test_size, random_state=args.seed
    )

    pre = Preprocessor(drop_missing_over=args.drop_missing_over).fit(X_train)
    Xt = pre.transform(X_train)
    Xv = pre.transform(X_val)

    # prepare categoricals for LGBM
    Xt = make_lgbm_ready(Xt, pre.cat_cols_)
    Xv = make_lgbm_ready(Xv, pre.cat_cols_)

    print(f"[INFO] Train: {Xt.shape} | Val: {Xv.shape}")
    print(f"[INFO] Cat cols: {len(pre.cat_cols_)} | Threads: {args.threads}")
    print(f"[INFO] Dropped cols: {len(pre.drop_cols_)}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=1,
        colsample_bytree=args.colsample,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        random_state=args.seed,
        n_jobs=args.threads
    )

    print("[TRAIN] Fitting LightGBM...")
    model.fit(
        Xt, y_train,
        eval_set=[(Xv, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=True),
            lgb.log_evaluation(200)
        ],
        categorical_feature=pre.cat_cols_ if len(pre.cat_cols_) > 0 else "auto"
    )

    # evaluate in LOG + TL
    pred_log = model.predict(Xv, num_iteration=model.best_iteration_)
    rmse_log = rmse(y_val, pred_log)
    r2_log = r2_score(y_val, pred_log)

    pred_tl = np.expm1(pred_log)
    y_val_tl = np.expm1(y_val)
    rmse_tl = rmse(y_val_tl, pred_tl)
    r2_tl = r2_score(y_val_tl, pred_tl)
    mae_tl = mean_absolute_error(y_val_tl, pred_tl)

    print("\n===== MODEL2 (LGBM) EVAL =====")
    print(f"Best iter: {model.best_iteration_}")
    print(f"RMSE_LOG: {rmse_log:.4f} | R2_LOG: {r2_log:.4f}")
    print(f"RMSE_TL : {rmse_tl:,.2f} | R2_TL : {r2_tl:.4f} | MAE_TL: {mae_tl:,.2f}")

    # save bundle
    bundle_path = os.path.join(args.outdir, "model2_lgbm_bundle.joblib")
    joblib.dump({"preprocessor": pre, "model": model}, bundle_path)

    meta = {
        "model": "LightGBMRegressor",
        "threads": int(args.threads),
        "seed": int(args.seed),
        "drop_missing_over": float(args.drop_missing_over),
        "clip_quantiles": [float(args.clip_low_q), float(args.clip_high_q)],
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
        "params": {
            "n_estimators": int(args.n_estimators),
            "learning_rate": float(args.learning_rate),
            "num_leaves": int(args.num_leaves),
            "min_child_samples": int(args.min_child_samples),
            "subsample": float(args.subsample),
            "colsample_bytree": float(args.colsample),
            "reg_lambda": float(args.reg_lambda),
            "reg_alpha": float(args.reg_alpha),
            "early_stopping_rounds": int(args.early_stopping)
        }
    }
    meta_path = os.path.join(args.outdir, "model2_lgbm_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {bundle_path}")
    print(f"[SAVE] {meta_path}")


if __name__ == "__main__":
    main()
