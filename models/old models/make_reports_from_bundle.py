import model6 as m6  # must be in same folder as this script
Preprocessor = m6.Preprocessor
PreprocessStats = m6.PreprocessStats
RobustTargetEncoder = m6.RobustTargetEncoder
ETALogger = m6.ETALogger

import os, re, json, argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def align_features_to_booster(X: pd.DataFrame, booster) -> pd.DataFrame:
    """
    Ensures X columns exactly match training feature names (adds missing=0, drops extra, orders).
    """
    expected = list(booster.feature_name())
    X = X.copy()

    # add missing
    missing = [c for c in expected if c not in X.columns]
    for c in missing:
        X[c] = 0

    # drop extra
    extra = [c for c in X.columns if c not in expected]
    if extra:
        X = X.drop(columns=extra)

    # reorder
    return X[expected]

def normalize_colnames(cols):
    out = []
    for c in cols:
        c2 = str(c).strip()
        c2 = re.sub(r"\s+", "_", c2)
        out.append(c2)
    return out

def robust_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for sep in [";", ",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", low_memory=False)
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
    raise last_err or ValueError("Could not read CSV")

def parse_price_to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = re.sub(r"[^\d]", "", s)
    if s == "":
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan

def find_loan_col(df: pd.DataFrame):
    candidates = [
        "Available_for_Loan", "AvailableForLoan", "Loan", "Krediye_Uygun",
        "Kredi", "KrediyeUygun"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        cl = c.lower()
        if "loan" in cl or "kredi" in cl:
            return c
    return None

def filter_loan_eligible(df: pd.DataFrame):
    loan_col = find_loan_col(df)
    if loan_col is None:
        return df, False, None
    if df[loan_col].notna().mean() < 0.05:
        return df, False, loan_col

    s = df[loan_col]
    if pd.api.types.is_numeric_dtype(s):
        return df[s == 1].copy(), True, loan_col

    s2 = s.astype(str).str.strip().str.lower()
    yes_vals = {"yes", "true", "1", "evet", "uygun", "eligible"}
    return df[s2.isin(yes_vals)].copy(), True, loan_col

def align_lgbm_pandas_categories(Xp: pd.DataFrame, model) -> pd.DataFrame:
    """
    Fixes LightGBM error:
      ValueError: train and valid dataset categorical_feature do not match.
    by aligning pandas category dtypes with training categories stored in booster.pandas_categorical
    """
    booster = getattr(model, "_Booster", None) or getattr(model, "booster_", None)
    cats_saved = getattr(booster, "pandas_categorical", None)
    if cats_saved is None:
        return Xp

    Xp = Xp.copy()
    cat_cols = [c for c in Xp.columns if str(Xp[c].dtype) == "category"]

    # If mismatch, still stabilize categories as strings
    if len(cat_cols) != len(cats_saved):
        for c in cat_cols:
            Xp[c] = Xp[c].astype(str).fillna("Unknown").astype("category")
        return Xp

    for c, cats in zip(cat_cols, cats_saved):
        cats = list(cats)
        dtype = pd.CategoricalDtype(categories=cats, ordered=False)

        s = Xp[c].astype(str)
        if "Unknown" in dtype.categories:
            s = s.where(s.isin(dtype.categories), other="Unknown")
        else:
            s = s.where(s.isin(dtype.categories), other=np.nan)

        Xp[c] = s.astype(dtype)

    return Xp

def get_best_iter(bundle, meta):
    if isinstance(bundle, dict) and "best_iter" in bundle:
        try:
            bi = int(bundle["best_iter"])
            return bi if bi > 0 else None
        except Exception:
            pass
    if isinstance(meta, dict) and "best_iter" in meta:
        try:
            bi = int(meta["best_iter"])
            return bi if bi > 0 else None
        except Exception:
            pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="Path to joblib bundle")
    ap.add_argument("--meta", default=None, help="Optional meta.json (for clip quantiles/best_iter)")
    ap.add_argument("--data", default=None, help="Optional training CSV to compute top_errors")
    ap.add_argument("--outdir", default="models", help="Output folder")
    ap.add_argument("--prefix", default="report", help="Output prefix (files named <prefix>_*.csv)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_size", type=float, default=0.20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    bundle = joblib.load(args.bundle)
    meta = {}
    if args.meta and os.path.exists(args.meta):
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

    # Bundle keys (robust)
    if isinstance(bundle, dict):
        model = bundle.get("model") or bundle.get("lgbm") or bundle.get("regressor")
        pre = bundle.get("preprocessor") or bundle.get("pre")
        te = bundle.get("te")  # may be None
    else:
        raise ValueError("Bundle must be a dict-like joblib object (expected keys: model, preprocessor, ...).")

    if model is None:
        raise ValueError("Could not find model in bundle (expected key 'model').")
    if pre is None:
        raise ValueError("Could not find preprocessor in bundle (expected key 'preprocessor').")

    # ---- 1) FEATURE IMPORTANCE ----
    booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
    if booster is None:
        raise ValueError("Model does not expose a LightGBM booster_/_Booster. Is this an LGBM model?")

    imp_gain = booster.feature_importance(importance_type="gain")
    imp_split = booster.feature_importance(importance_type="split")
    feat_names = booster.feature_name()

    imp_df = pd.DataFrame({"feature": feat_names, "gain": imp_gain, "split": imp_split}) \
              .sort_values("gain", ascending=False)

    imp_path = os.path.join(args.outdir, f"{args.prefix}_feature_importance.csv")
    imp_df.to_csv(imp_path, index=False, encoding="utf-8")
    print(f"[OK] Wrote: {imp_path}")

    # ---- 2) TOP ERRORS (optional, needs data) ----
    if args.data:
        df = robust_read_csv(args.data)
        
        # Normalize columns (fixes "Number of rooms" -> "Number_of_rooms")
        df.columns = normalize_colnames(df.columns)

        if "Price" not in df.columns:
            raise ValueError(f"Price column not found in --data CSV (after normalization). Columns: {df.columns.tolist()}")

        df["Price_num"] = df["Price"].map(parse_price_to_float)
        df = df.dropna(subset=["Price_num"]).copy()

        df2, used, loan_col = filter_loan_eligible(df)
        df = df2

        # clip quantiles
        clip_q = meta.get("clip_quantiles", [0.001, 0.995])
        q_lo, q_hi = float(clip_q[0]), float(clip_q[1])
        y = df["Price_num"].astype(float)
        lo, hi = y.quantile(q_lo), y.quantile(q_hi)

        y_log = np.log1p(y.clip(lo, hi)).to_numpy(dtype=float)
        X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()

        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y_log, test_size=args.val_size, random_state=args.seed
        )

        # transform
        Xt_va = pre.transform(X_va)
        if te is not None:
            Xt_va = te.transform(Xt_va)

        # align
        Xt_va = align_lgbm_pandas_categories(Xt_va, model)
        booster = getattr(model, "booster_", None) or getattr(model, "_Booster", None)
        Xt_va = align_features_to_booster(Xt_va, booster)
        
        best_iter = get_best_iter(bundle, meta)
        pred_log = model.predict(Xt_va, num_iteration=best_iter) if best_iter else model.predict(Xt_va)

        # back to TL
        y_va_tl = np.expm1(y_va)
        pred_tl = np.expm1(pred_log)
        abs_err = np.abs(pred_tl - y_va_tl)

        # metrics - FIXED: replaced mean_squared_error(..., squared=False) with np.sqrt(mean_squared_error(...))
        mse_log = mean_squared_error(y_va, pred_log)
        rmse_log = np.sqrt(mse_log)
        
        r2_log = r2_score(y_va, pred_log)
        
        mse_tl = mean_squared_error(y_va_tl, pred_tl)
        rmse_tl = np.sqrt(mse_tl)
        
        r2_tl = r2_score(y_va_tl, pred_tl)
        mae_tl = mean_absolute_error(y_va_tl, pred_tl)
        
        print(f"[VAL] RMSE_LOG={rmse_log:.6f} R2_LOG={r2_log:.6f} | RMSE_TL={rmse_tl:,.2f} R2_TL={r2_tl:.6f} MAE_TL={mae_tl:,.2f}")

        top_k = min(200, len(X_va))
        top_idx = np.argsort(-abs_err)[:top_k]

        id_cols = [c for c in ["district", "neighborhood", "location", "Town", "Ilce", "Mahalle"] if c in X_va.columns]

        out = X_va.iloc[top_idx].copy()
        out["y_true_tl"] = y_va_tl[top_idx]
        out["y_pred_tl"] = pred_tl[top_idx]
        out["abs_error_tl"] = abs_err[top_idx]
        out = out[id_cols + [c for c in out.columns if c not in id_cols]]

        top_path = os.path.join(args.outdir, f"{args.prefix}_top_errors.csv")
        out.to_csv(top_path, index=False, encoding="utf-8")
        print(f"[OK] Wrote: {top_path}")

if __name__ == "__main__":
    main()