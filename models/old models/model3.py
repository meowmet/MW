# model3.py (v2)
# Ensemble: LightGBM + XGBoost(core train) + optimal weight blend (log-space)
# Usage:
#   python model3.py --data .\hackathon_train_set.csv --outdir .\models --threads 10
# Optional:
#   python model3.py --no_loan_filter

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

import lightgbm as lgb
import xgboost as xgb


# -----------------------------
# Metrics (sklearn version safe)
# -----------------------------
def rmse(y_true, y_pred):
    try:
        from sklearn.metrics import root_mean_squared_error
        return float(root_mean_squared_error(y_true, y_pred))
    except Exception:
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# -----------------------------
# IO + cleaning
# -----------------------------
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


def sanitize_colname(c: str) -> str:
    c = str(c).replace("\u00a0", " ")
    c = re.sub(r"\s+", "_", c.strip())
    return c


# -----------------------------
# Preprocessor
# -----------------------------
class Preprocessor:
    """
    - Standardizes missing tokens -> NaN
    - Adds date parts if present
    - Drops: all-missing, >threshold missing, constants
    - Fills: numeric median, cat "Unknown"
    - One-hot encodes categoricals for XGBoost
    - Keeps stable column set for inference
    """

    def __init__(self, drop_missing_over=0.50):
        self.drop_missing_over = float(drop_missing_over)
        self.drop_cols_ = []
        self.cat_cols_raw_ = []
        self.numeric_medians_ = {}
        self.feature_cols_raw_ = []
        self.rename_map_ = {}
        self.debug_drop_counts_ = {}

        self.ohe_columns_ = None  # final columns after get_dummies on train

    @staticmethod
    def _standardize_missing(df: pd.DataFrame) -> pd.DataFrame:
        df = df.replace("\u00a0", " ", regex=False)
        df = df.replace(r"^[\s\u00a0]*$", np.nan, regex=True)
        df = df.replace({"None": np.nan, "NONE": np.nan, "null": np.nan, "NULL": np.nan, "NaN": np.nan, "nan": np.nan})
        return df

    def _base_transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

        for c in self.feature_cols_raw_:
            if c not in df.columns:
                df[c] = np.nan
        df = df[self.feature_cols_raw_].copy()

        for c in df.columns:
            if c in self.cat_cols_raw_:
                df[c] = df[c].fillna("Unknown").astype(str)
            else:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(self.numeric_medians_.get(c, 0.0))

        df = df.rename(columns=self.rename_map_)
        return df

    def fit(self, X: pd.DataFrame):
        df = self._standardize_missing(X.copy())

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

        self.cat_cols_raw_ = [c for c in df.columns if df[c].dtype == "object"]

        self.numeric_medians_ = {}
        for c in df.columns:
            if c in self.cat_cols_raw_:
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            self.numeric_medians_[c] = float(s.median()) if s.notna().any() else 0.0

        self.feature_cols_raw_ = list(df.columns)
        self.rename_map_ = {c: sanitize_colname(c) for c in self.feature_cols_raw_}

        base = self._base_transform(X)
        cat_cols_s = [self.rename_map_[c] for c in self.cat_cols_raw_ if c in self.rename_map_]
        ohe = pd.get_dummies(base, columns=[c for c in cat_cols_s if c in base.columns], drop_first=False)
        self.ohe_columns_ = list(ohe.columns)

        return self

    def transform_base(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._base_transform(X)

    def transform_ohe(self, X: pd.DataFrame) -> pd.DataFrame:
        base = self._base_transform(X)
        cat_cols_s = [self.rename_map_[c] for c in self.cat_cols_raw_ if c in self.rename_map_]
        ohe = pd.get_dummies(base, columns=[c for c in cat_cols_s if c in base.columns], drop_first=False)

        for c in self.ohe_columns_:
            if c not in ohe.columns:
                ohe[c] = 0
        ohe = ohe[self.ohe_columns_].copy()
        return ohe


def make_lgbm_ready(df: pd.DataFrame, cat_cols):
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def clip_target(y: pd.Series, low_q=0.001, high_q=0.995) -> pd.Series:
    lo = y.quantile(low_q)
    hi = y.quantile(high_q)
    return y.clip(lo, hi)


def optimal_weight(y, pred_a, pred_b):
    d = (pred_a - pred_b)
    denom = float(np.mean(d * d))
    if denom <= 1e-12:
        return 0.5
    w = float(np.mean((y - pred_b) * d) / denom)
    return float(np.clip(w, 0.0, 1.0))


def eval_block(name, y_val, pred_log):
    r_log = rmse(y_val, pred_log)
    r2l = r2_score(y_val, pred_log)

    pred_tl = np.expm1(pred_log)
    y_tl = np.expm1(y_val)
    r_tl = rmse(y_tl, pred_tl)
    r2t = r2_score(y_tl, pred_tl)
    mae = mean_absolute_error(y_tl, pred_tl)

    print(f"\n===== {name} =====")
    print(f"RMSE_LOG: {r_log:.6f} | R2_LOG: {r2l:.6f}")
    print(f"RMSE_TL : {r_tl:,.2f} | R2_TL : {r2t:.6f} | MAE_TL: {mae:,.2f}")
    return {"rmse_log": r_log, "r2_log": r2l, "rmse_tl": r_tl, "r2_tl": r2t, "mae_tl": mae}


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
    ap.add_argument("--no_loan_filter", action="store_true")

    # LightGBM
    ap.add_argument("--lgb_n_estimators", type=int, default=20000)
    ap.add_argument("--lgb_lr", type=float, default=0.03)
    ap.add_argument("--lgb_num_leaves", type=int, default=128)
    ap.add_argument("--lgb_min_child", type=int, default=50)
    ap.add_argument("--lgb_subsample", type=float, default=0.8)
    ap.add_argument("--lgb_colsample", type=float, default=0.8)
    ap.add_argument("--lgb_reg_lambda", type=float, default=2.0)
    ap.add_argument("--lgb_reg_alpha", type=float, default=0.0)
    ap.add_argument("--lgb_early_stopping", type=int, default=300)

    # XGBoost core
    ap.add_argument("--xgb_n_estimators", type=int, default=20000)
    ap.add_argument("--xgb_lr", type=float, default=0.03)
    ap.add_argument("--xgb_max_depth", type=int, default=8)
    ap.add_argument("--xgb_min_child_weight", type=float, default=5.0)
    ap.add_argument("--xgb_subsample", type=float, default=0.8)
    ap.add_argument("--xgb_colsample", type=float, default=0.8)
    ap.add_argument("--xgb_reg_lambda", type=float, default=2.0)
    ap.add_argument("--xgb_reg_alpha", type=float, default=0.0)
    ap.add_argument("--xgb_gamma", type=float, default=0.0)
    ap.add_argument("--xgb_early_stopping", type=int, default=300)
    ap.add_argument("--verbose", type=int, default=200)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("os.cpu_count() =", os.cpu_count())
    print(f"[LOAD] Reading: {args.data}")

    df = robust_read_csv(args.data)
    if "Price" not in df.columns:
        raise ValueError("Price column not found.")

    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()

    if not args.no_loan_filter:
        before = len(df)
        df = filter_loan_eligible(df)
        after = len(df)
        if after != before:
            print(f"[FILTER] Loan eligible: {before} -> {after}")
    else:
        print("[FILTER] Loan eligible: DISABLED")

    y = df["Price_num"].astype(float)
    y_clip = clip_target(y, args.clip_low_q, args.clip_high_q)
    y_log = np.log1p(y_clip)

    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_log, test_size=args.test_size, random_state=args.seed
    )

    pre = Preprocessor(drop_missing_over=args.drop_missing_over).fit(X_train)

    # Base (for LGBM)
    Xt_base = pre.transform_base(X_train)
    Xv_base = pre.transform_base(X_val)

    # OHE (for XGB)
    Xt_ohe = pre.transform_ohe(X_train)
    Xv_ohe = pre.transform_ohe(X_val)

    cat_cols_s = [sanitize_colname(c) for c in pre.cat_cols_raw_]
    cat_cols_s = [c for c in cat_cols_s if c in Xt_base.columns]

    print(f"[INFO] Train(base): {Xt_base.shape} | Val(base): {Xv_base.shape}")
    print(f"[INFO] Train(ohe) : {Xt_ohe.shape} | Val(ohe) : {Xv_ohe.shape}")
    print(f"[INFO] Cat cols(base): {len(cat_cols_s)} | Threads: {args.threads}")
    print(f"[DROP] all_missing={pre.debug_drop_counts_['all_missing']} "
          f"high_missing={pre.debug_drop_counts_['high_missing']} "
          f"constant={pre.debug_drop_counts_['constant']} "
          f"-> total_dropped={len(pre.drop_cols_)}")

    # -----------------------------
    # LightGBM
    # -----------------------------
    Xt_lgb = make_lgbm_ready(Xt_base, cat_cols_s)
    Xv_lgb = make_lgbm_ready(Xv_base, cat_cols_s)

    lgbm = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=args.lgb_n_estimators,
        learning_rate=args.lgb_lr,
        num_leaves=args.lgb_num_leaves,
        min_child_samples=args.lgb_min_child,
        subsample=args.lgb_subsample,
        subsample_freq=1,
        colsample_bytree=args.lgb_colsample,
        reg_lambda=args.lgb_reg_lambda,
        reg_alpha=args.lgb_reg_alpha,
        random_state=args.seed,
        n_jobs=args.threads,
        force_row_wise=True,
    )

    print("\n[TRAIN] LightGBM...")
    lgbm.fit(
        Xt_lgb, y_train,
        eval_set=[(Xv_lgb, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(args.lgb_early_stopping, verbose=True),
            lgb.log_evaluation(args.verbose)
        ],
        categorical_feature=cat_cols_s if len(cat_cols_s) > 0 else "auto"
    )
    pred_lgb_log = lgbm.predict(Xv_lgb, num_iteration=getattr(lgbm, "best_iteration_", None))

    # -----------------------------
    # XGBoost (core API with early stopping)  ✅ fixes your error
    # -----------------------------
    print("\n[TRAIN] XGBoost (core)...")

    dtrain = xgb.DMatrix(Xt_ohe, label=y_train)
    dval = xgb.DMatrix(Xv_ohe, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": args.xgb_lr,
        "max_depth": args.xgb_max_depth,
        "min_child_weight": args.xgb_min_child_weight,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "lambda": args.xgb_reg_lambda,
        "alpha": args.xgb_reg_alpha,
        "gamma": args.xgb_gamma,
        "tree_method": "hist",
        "seed": args.seed,
        "nthread": args.threads,
        "verbosity": 1,
    }

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=args.xgb_n_estimators,
        evals=[(dval, "valid")],
        early_stopping_rounds=args.xgb_early_stopping,
        verbose_eval=args.verbose,
        evals_result=evals_result,
    )

    # predict with best iteration
    best_iter = getattr(booster, "best_iteration", None)
    if best_iter is None:
        best_iter = args.xgb_n_estimators - 1
    n_best = int(best_iter) + 1

    try:
        pred_xgb_log = booster.predict(dval, iteration_range=(0, n_best))
    except TypeError:
        # older xgboost fallback
        ntl = getattr(booster, "best_ntree_limit", 0)
        pred_xgb_log = booster.predict(dval, ntree_limit=ntl if ntl else n_best)

    # -----------------------------
    # Evaluate + Blend
    # -----------------------------
    m_lgb = eval_block("LightGBM", y_val, pred_lgb_log)
    m_xgb = eval_block("XGBoost(core)", y_val, pred_xgb_log)

    w = optimal_weight(y_val.values, pred_lgb_log, pred_xgb_log)  # weight for LGBM
    pred_blend_log = w * pred_lgb_log + (1.0 - w) * pred_xgb_log
    m_bl = eval_block(f"BLEND (w_lgb={w:.3f}, w_xgb={1-w:.3f})", y_val, pred_blend_log)

    # -----------------------------
    # Save
    # -----------------------------
    bundle_path = os.path.join(args.outdir, "model3_lgbm_xgb_blend_bundle.joblib")
    joblib.dump(
        {
            "preprocessor": pre,
            "lightgbm": lgbm,
            "xgb_booster": booster,  # core booster
            "blend_weight_lgbm": w,
            "ohe_columns": pre.ohe_columns_,
        },
        bundle_path
    )

    meta = {
        "seed": args.seed,
        "threads": args.threads,
        "loan_filter_enabled": (not args.no_loan_filter),
        "drop_missing_over": args.drop_missing_over,
        "clip_quantiles": [args.clip_low_q, args.clip_high_q],
        "dropped_columns": pre.drop_cols_,
        "metrics": {
            "lightgbm": {k: float(v) for k, v in m_lgb.items()},
            "xgboost": {k: float(v) for k, v in m_xgb.items()},
            "blend": {k: float(v) for k, v in m_bl.items()},
            "blend_weight_lgbm": float(w),
            "blend_weight_xgb": float(1.0 - w),
        },
        "best_iter": {
            "lightgbm": int(getattr(lgbm, "best_iteration_", 0) or 0),
            "xgboost": int(best_iter),
        },
        "params": {
            "lightgbm": {
                "n_estimators": args.lgb_n_estimators,
                "learning_rate": args.lgb_lr,
                "num_leaves": args.lgb_num_leaves,
                "min_child_samples": args.lgb_min_child,
                "subsample": args.lgb_subsample,
                "colsample_bytree": args.lgb_colsample,
                "reg_lambda": args.lgb_reg_lambda,
                "reg_alpha": args.lgb_reg_alpha,
                "early_stopping_rounds": args.lgb_early_stopping,
            },
            "xgboost": {
                "num_boost_round": args.xgb_n_estimators,
                "eta": args.xgb_lr,
                "max_depth": args.xgb_max_depth,
                "min_child_weight": args.xgb_min_child_weight,
                "subsample": args.xgb_subsample,
                "colsample_bytree": args.xgb_colsample,
                "lambda": args.xgb_reg_lambda,
                "alpha": args.xgb_reg_alpha,
                "gamma": args.xgb_gamma,
                "early_stopping_rounds": args.xgb_early_stopping,
                "tree_method": "hist",
            },
        },
    }

    meta_path = os.path.join(args.outdir, "model3_lgbm_xgb_blend_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[SAVE] {bundle_path}")
    print(f"[SAVE] {meta_path}")


if __name__ == "__main__":
    main()
