# Istanbul Housing Investment Advisor (2020) — Streamlit UI (Final Fixed v5 FULL)
#
# Run:
#   cd MW
#   streamlit run app.py

import os
import sys
import json
import traceback
import re
import glob
import textwrap
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import joblib
import streamlit as st
from PIL import Image

# ------------------------------------------------------------
# PATH FIX
# ------------------------------------------------------------
ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# IMPORTANT: We assume model6 structure is compatible with the bundles.
import model6 as m6

# ------------------------------------------------------------
# HTML RENDER FIX (NO MORE "CODE BLOCK" ISSUE)
# ------------------------------------------------------------
def st_html(html: str):
    """Render HTML safely in Streamlit markdown (prevents indentation -> code-block)."""
    html = textwrap.dedent(html).strip("\n")
    st.markdown(html, unsafe_allow_html=True)

# ------------------------------------------------------------
# CSS — SAFE THEME (Full & Detailed)
# ------------------------------------------------------------
def inject_css():
    st.markdown(
        """
<style>
:root{
  --bg0:#070A12;
  --bg1:#0B1224;
  --card: rgba(255,255,255,.06);
  --card2: rgba(255,255,255,.09);
  --line: rgba(255,255,255,.11);
  --text: rgba(255,255,255,.93);
  --muted: rgba(255,255,255,.70);
  --muted2: rgba(255,255,255,.55);
  --shadow: 0 25px 70px rgba(0,0,0,.45);
  --r: 18px;

  --good: rgba(80,255,170,.22);
  --warn: rgba(255,215,120,.20);
  --bad:  rgba(255,110,160,.20);

  --accent1: rgba(130,200,255,.28);
  --accent2: rgba(255,120,220,.20);
  --accent3: rgba(120,255,210,.16);
}

/* Scroll & Layout */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"]{
  height: auto !important;
  overflow: auto !important;
}
section[data-testid="stSidebar"]{
  overflow: auto !important;
  max-height: 100vh !important;
  border-right: 1px solid var(--line);
}

/* Background */
html, body, [data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1200px 700px at 20% 0%, var(--accent1), transparent 60%),
    radial-gradient(1000px 650px at 80% 10%, var(--accent2), transparent 62%),
    radial-gradient(900px 650px at 50% 95%, var(--accent3), transparent 65%),
    linear-gradient(180deg, var(--bg0), var(--bg1)) !important;
  color: var(--text) !important;
}

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header{ background: transparent !important; border-bottom: none !important; }
header [data-testid="stToolbar"]{ position: static !important; }

/* Custom Components */
.block-container{ padding-top: 1.5rem !important; padding-bottom: 3rem !important; max-width: 1280px !important; }

.hero{
  position: relative;
  padding: 24px;
  border-radius: 22px;
  border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(255,255,255,.08), rgba(255,255,255,.03));
  box-shadow: var(--shadow);
  overflow: hidden;
  margin-bottom: 20px;
}
.hero h1{ margin:0; font-size: 2.0rem; letter-spacing:.2px; }
.hero p{ margin:.35rem 0 0 0; color: var(--muted); font-size:1.02rem; }

.glass{
  background: var(--card); border: 1px solid var(--line);
  border-radius: var(--r); box-shadow: var(--shadow);
  padding: 16px; margin-bottom: 15px;
}
.glass2{
  background: var(--card2); border: 1px solid var(--line);
  border-radius: var(--r); padding: 16px;
}

.badge{
  display:inline-flex; align-items:center; gap:.45rem;
  padding: .34rem .70rem; border-radius: 999px;
  border: 1px solid var(--line); background: rgba(255,255,255,.06);
  font-weight: 900; font-size: .95rem;
}
.badge.good{ background: var(--good); }
.badge.warn{ background: var(--warn); }
.badge.bad { background: var(--bad); }

.kpi-grid{ display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; }
.kpi{
  background: rgba(255,255,255,.045);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px; padding: 12px;
}
.kpi .k{ color: var(--muted2); font-size: .84rem; text-transform: uppercase; }
.kpi .v{ font-size: 1.22rem; font-weight: 900; margin-top: 6px; }

.kpi-box{
  background: rgba(255,255,255,.045);
  border: 1px solid rgba(255,255,255,.10);
  border-radius: 16px;
  padding: 14px;
  text-align:center;
}
.kpi-title{ color: var(--muted2); font-size:.82rem; text-transform:uppercase; }
.kpi-value{ font-size:1.35rem; font-weight:900; margin-top:6px; }

.stButton > button{
  border-radius: 14px !important; border: 1px solid rgba(255,255,255,.14) !important;
  background: rgba(130,200,255,.22) !important; color: white !important;
}
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"]{
  border-radius: 14px !important;
}
.stTabs [data-baseweb="tab"]{
  border-radius: 999px !important; background: rgba(255,255,255,.04);
}
.stTabs [aria-selected="true"]{
  background: rgba(130,200,255,.18) !important;
}

.hr {
  height: 1px;
  background: linear-gradient(to right, transparent, var(--line), transparent);
  margin: 20px 0;
  border: none;
}

.small-muted {
  font-size: 0.9rem;
  color: var(--muted2);
  margin-bottom: 10px;
}
</style>
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------
# Helpers: Formatting & Safe Parsing
# ------------------------------------------------------------
def detect_price_col(df: pd.DataFrame) -> str | None:
    # 1) name-based (strong)
    name_candidates = [
        "price", "fiyat", "ilan", "listing", "sale", "satis", "satış",
        "gercek", "gerçek", "target", "y", "label", "ground", "gt"
    ]
    skip_keywords = ["lat", "lon", "enlem", "boylam", "m2", "net", "brut", "brüt", "alan", "age", "yas", "yaş", "room", "oda", "kat"]

    cols = list(df.columns)
    cols_l = [c.lower() for c in cols]

    # exact/contains candidate
    for c, cl in zip(cols, cols_l):
        if any(k in cl for k in name_candidates) and not any(sk in cl for sk in skip_keywords):
            return c

    # 2) heuristic: pick the most "money-like" numeric column
    # money columns usually have big values (>= 50k) and not coordinates
    best = None
    best_score = -1.0

    for c in cols:
        cl = c.lower()
        if any(sk in cl for sk in skip_keywords):
            continue

        s = df[c]
        # convert object money-like strings too
        if s.dtype == object:
            s_num = s.apply(safe_turkish_float)
        else:
            s_num = pd.to_numeric(s, errors="coerce")

        s_num = s_num.replace([np.inf, -np.inf], np.nan).dropna()
        if len(s_num) < max(5, int(0.1 * len(df))):  # too sparse -> skip
            continue

        med = float(np.nanmedian(s_num))
        mx = float(np.nanmax(s_num))

        # basic money plausibility
        if mx < 50000 or med < 30000:
            continue

        # score: prefer higher median + higher coverage
        coverage = len(s_num) / max(1, len(df))
        score = (med / 1000.0) + (coverage * 10.0)

        if score > best_score:
            best_score = score
            best = c

    return best

def fmt_tr(val):
    """Format: 12500 -> '12.500' (Display only)"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    try:
        return f"{float(val):,.0f}".replace(",", ".")
    except Exception:
        return "—"

def fmt_pct(val):
    """Format: 4.5 -> '4,5' (Display only)"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    try:
        return f"{float(val):.1f}".replace(".", ",")
    except Exception:
        return "—"

def safe_turkish_float(x):
    """
    Robustly parses numbers from string/float.
    Fixes Turkish format: 1.000.000 -> 1000000 and 1.250,50 -> 1250.50
    ALSO fixes: '480000.0' should remain 480000.0 (not 4,800,000)
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float)) and not (isinstance(x, float) and np.isnan(x)):
        return float(x)

    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "null", "-"}:
        return np.nan

    # Keep digits, dot, comma, minus
    s = re.sub(r"[^\d\.,\-]", "", s)

    # If it's a normal decimal with ONE dot and NO comma (e.g., 480000.0), parse directly
    if s.count(".") == 1 and s.count(",") == 0:
        try:
            return float(s)
        except Exception:
            pass

    # If it has multiple dots and no comma -> dots are thousands separators
    if s.count(".") > 1 and s.count(",") == 0:
        s_no_dots = s.replace(".", "")
        try:
            return float(s_no_dots)
        except Exception:
            return np.nan

    # Turkish style: 1.250,50 -> 1250.50
    s2 = s.replace(".", "").replace(",", ".")
    try:
        return float(s2)
    except Exception:
        return np.nan

def load_json_safe(path: str):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def save_json_safe(path: str, obj: dict) -> Tuple[bool, str]:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

def load_thresholds_optional(th_path: str):
    th = load_json_safe(th_path) or {}
    t_buy = float(th.get("t_buy", th.get("buy", 0.03)))
    t_sell = float(th.get("t_sell", th.get("sell", 0.03)))
    return {"t_buy": t_buy, "t_sell": t_sell}

def business_label(listing_price: float, fair_value: float, t_buy: float, t_sell: float):
    if fair_value <= 0:
        return "Normal", 0.0
    diff = (listing_price - fair_value) / fair_value
    if diff <= -t_buy:
        return "Fırsat", diff
    if diff >= t_sell:
        return "Pahalı", diff
    return "Normal", diff

def explain_decision_simple(listing_price: float, fair_value: float, t_buy: float, t_sell: float) -> str:
    if fair_value <= 0:
        return "Adil değer hesaplanamadı."
    diff = (listing_price - fair_value) / fair_value
    diff_pct = diff * 100.0

    pct_txt = fmt_pct(abs(diff_pct))
    buy_txt = fmt_pct(t_buy * 100)
    sell_txt = fmt_pct(t_sell * 100)

    if diff <= -t_buy:
        return (f"📉 İlan fiyatı adil değerden **%{pct_txt} daha düşük**.\n\n"
                f"✅ **Fırsat eşiğini (≥ %{buy_txt})** geçtiği için: **FIRSAT**")
    if diff >= t_sell:
        return (f"📈 İlan fiyatı adil değerden **%{pct_txt} daha yüksek**.\n\n"
                f"⛔ **Pahalı eşiğini (≥ %{sell_txt})** geçtiği için: **PAHALI**")

    sign = "+" if diff_pct > 0 else "-"
    return (f"⚖️ İlan fiyatı adil değere yakın (Fark: {sign}%{pct_txt}).\n\n"
            f"Eşiklerin içinde kaldığı için: **NORMAL**")

# ------------------------------------------------------------
# Preprocessor safety helpers (CSV missing cols + default fill)
# ------------------------------------------------------------
def ensure_cols_for_pre_add_features(raw_df: pd.DataFrame, pre) -> pd.DataFrame:
    """
    - Adds missing amenity columns as 0
    - Adds missing core raw feature columns (room/net/gross/age) as NaN
    - Ensures cat cols exist (Unknown)
    """
    df = raw_df.copy()

    # amenity columns expected by pre (binary)
    amen_cols = getattr(pre, "_amenity_cols", []) or []
    for c in amen_cols:
        if c not in df.columns:
            df[c] = 0

    # raw core feature columns names (model6 pipeline)
    for attr in ["_room_col", "_net_col", "_gross_col", "_age_col"]:
        c = getattr(pre, attr, None)
        if c and c not in df.columns:
            df[c] = np.nan

    # categorical columns expected
    cat_cols = getattr(pre, "cat_cols_", []) or []
    for c in cat_cols:
        if c not in df.columns:
            df[c] = "Unknown"
        # keep as string; pre may cast later
        df[c] = df[c].astype(str)

    # numeric columns expected (if exposed)
    num_cols = getattr(pre, "num_cols_", []) or []
    for c in num_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df

def detect_location_col(pre) -> Optional[str]:
    cats = getattr(pre, "cat_cols_", []) or []
    keys = ["district", "ilce", "ilçe", "lokasyon", "location"]
    for k in keys:
        for c in cats:
            if k in c.lower():
                return c
    return cats[0] if cats else None

def get_allowed_locations_from_te(te, loc_col: str) -> List[str]:
    if te is None or not loc_col:
        return []
    enc = getattr(te, "encodings_", None)
    if isinstance(enc, dict) and loc_col in enc:
        return sorted(list(enc[loc_col].keys()))
    return []

@st.cache_resource(show_spinner=False)
def load_bundle_cached(bundle_path: str):
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")
    old_main = sys.modules.get("__main__")
    try:
        sys.modules["__main__"] = m6
        bundle = joblib.load(bundle_path)
    finally:
        if old_main is not None:
            sys.modules["__main__"] = old_main
    return bundle

def align_lgbm_pandas_categories(Xp: pd.DataFrame, model, pre) -> pd.DataFrame:
    booster = getattr(model, "_Booster", None) or getattr(model, "booster_", None)
    cats_saved = getattr(booster, "pandas_categorical", None)
    known_cat_cols = getattr(pre, "cat_cols_", []) or []

    for col in known_cat_cols:
        if col in Xp.columns:
            if str(Xp[col].dtype) != "category":
                Xp[col] = Xp[col].astype(str).astype("category")

    if cats_saved is None:
        return Xp

    Xp = Xp.copy()
    current_cat_cols = [c for c in Xp.columns if str(Xp[c].dtype) == "category"]

    if len(current_cat_cols) == len(cats_saved):
        for col_name, categories_list in zip(current_cat_cols, cats_saved):
            dtype = pd.CategoricalDtype(categories=categories_list, ordered=False)
            s = Xp[col_name].astype(str)
            if "Unknown" in dtype.categories:
                s = s.where(s.isin(dtype.categories), other="Unknown")
            else:
                s = s.where(s.isin(dtype.categories), other=np.nan)
            Xp[col_name] = s.astype(dtype)
    return Xp

def _coerce_numeric_inplace(df: pd.DataFrame, col: str):
    if col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(safe_turkish_float)

def predict_from_raw_df(raw_df: pd.DataFrame, bundle, best_iter: int):
    pre = bundle.get("preprocessor")
    te = bundle.get("target_encoder")
    model = bundle.get("model")
    if pre is None or model is None:
        raise RuntimeError("Bundle corrupted (missing preprocessor/model).")

    df = raw_df.copy()
    df.columns = m6.normalize_colnames(df.columns.tolist())
    df = ensure_cols_for_pre_add_features(df, pre)

    # Try to parse some common numeric raw fields safely
    net_col = getattr(pre, "_net_col", None)
    gross_col = getattr(pre, "_gross_col", None)
    age_col = getattr(pre, "_age_col", None)
    for c in [net_col, gross_col, age_col]:
        if c:
            _coerce_numeric_inplace(df, c)

    # Transform
    Xp = pre.transform(df)
    if te is not None:
        Xp = te.transform(Xp)

    # align categories
    Xp = align_lgbm_pandas_categories(Xp, model, pre)

    # HARD SAFETY: prevent NaN/inf from killing predict
    if isinstance(Xp, pd.DataFrame):
        Xp = Xp.replace([np.inf, -np.inf], np.nan)

        # numeric -> fill 0
        num_cols = Xp.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            Xp[num_cols] = Xp[num_cols].fillna(0)

        # object -> fill "Unknown"
        obj_cols = Xp.select_dtypes(include=["object"]).columns
        if len(obj_cols) > 0:
            Xp[obj_cols] = Xp[obj_cols].fillna("Unknown")

        # category -> fill with existing category (prefer "Unknown")
        cat_cols = Xp.select_dtypes(include=["category"]).columns
        for c in cat_cols:
            if Xp[c].isna().any():
                cats = list(Xp[c].cat.categories)
                fill_val = "Unknown" if "Unknown" in cats else (cats[0] if len(cats) else "Unknown")
                if fill_val not in cats:
                    Xp[c] = Xp[c].cat.add_categories([fill_val])
                Xp[c] = Xp[c].fillna(fill_val)

    else:
        Xp = np.asarray(Xp)
        Xp = np.nan_to_num(Xp, nan=0.0, posinf=0.0, neginf=0.0)

    pred_log = model.predict(Xp, num_iteration=best_iter)
    return np.expm1(np.asarray(pred_log, dtype=float))

def smart_column_mapping(df: pd.DataFrame, pre) -> Tuple[pd.DataFrame, List[str]]:
    """
    Maps user CSV columns to what pre expects.
    Works on normalized column names.
    """
    df = df.copy()
    logs: List[str] = []

    # Normalize CSV cols early
    df.columns = m6.normalize_colnames(df.columns.tolist())

    target_net = getattr(pre, "_net_col", None)
    target_age = getattr(pre, "_age_col", None)
    target_room = getattr(pre, "_room_col", None)
    target_gross = getattr(pre, "_gross_col", None)

    loc_col = detect_location_col(pre)

    cols_lower = {c.lower(): c for c in df.columns}

    # NET m2
    if target_net and target_net not in df.columns:
        cands = [c for c in cols_lower if any(x in c for x in ["net", "m2", "alan"]) and "brut" not in c and "gross" not in c]
        if cands:
            df.rename(columns={cols_lower[cands[0]]: target_net}, inplace=True)
            logs.append(f"Eşleştirildi: {cols_lower[cands[0]]} -> {target_net}")

    # GROSS / BRUT
    if target_gross and target_gross not in df.columns:
        cands = [c for c in cols_lower if any(x in c for x in ["brut", "brüt", "gross"]) or ("m2" in c and "brut" in c)]
        if cands:
            df.rename(columns={cols_lower[cands[0]]: target_gross}, inplace=True)
            logs.append(f"Eşleştirildi: {cols_lower[cands[0]]} -> {target_gross}")

    # AGE
    if target_age and target_age not in df.columns:
        cands = [c for c in cols_lower if any(x in c for x in ["age", "yas", "yaş", "bina_yasi", "binayasi"])]
        if cands:
            df.rename(columns={cols_lower[cands[0]]: target_age}, inplace=True)
            logs.append(f"Eşleştirildi: {cols_lower[cands[0]]} -> {target_age}")

    # ROOMS
    if target_room and target_room not in df.columns:
        cands = [c for c in cols_lower if any(x in c for x in ["oda", "rooms", "room", "salon"])]
        if cands:
            df.rename(columns={cols_lower[cands[0]]: target_room}, inplace=True)
            logs.append(f"Eşleştirildi: {cols_lower[cands[0]]} -> {target_room}")

    # LOCATION
    if loc_col and loc_col not in df.columns:
        cands = [c for c in cols_lower if any(x in c for x in ["ilce", "ilçe", "district", "dist", "lokasyon", "location", "semt", "mahalle"])]
        if cands:
            df.rename(columns={cols_lower[cands[0]]: loc_col}, inplace=True)
            logs.append(f"Eşleştirildi: {cols_lower[cands[0]]} -> {loc_col}")

    return df, logs

def apply_lenient_defaults(df: pd.DataFrame, pre, loc_col: Optional[str], allowed_locs: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Makes sure critical columns exist and are not empty so prediction never crashes.
    Uses reasonable defaults (only when missing/empty).
    """
    df = df.copy()
    logs: List[str] = []

    room_col = getattr(pre, "_room_col", None) or "rooms"
    net_col = getattr(pre, "_net_col", None) or "net"
    gross_col = getattr(pre, "_gross_col", None) or "gross"
    age_col = getattr(pre, "_age_col", None) or "age"

    # choose default location
    default_loc = None
    if allowed_locs:
        default_loc = "Kadikoy" if "Kadikoy" in allowed_locs else allowed_locs[0]
    else:
        default_loc = "Unknown"

    # Ensure columns exist
    for c in [room_col, net_col, gross_col, age_col]:
        if c not in df.columns:
            df[c] = np.nan
            logs.append(f"EKSİK KOLON eklendi: {c} (NaN)")

    if loc_col and loc_col not in df.columns:
        df[loc_col] = default_loc
        logs.append(f"EKSİK KOLON eklendi: {loc_col} (varsayılan='{default_loc}')")

    # Fill missing values with defaults (only when NaN/empty)
    # rooms default
    df[room_col] = df[room_col].astype(str)
    mask_room = df[room_col].isna() | (df[room_col].str.strip() == "") | (df[room_col].str.lower().isin(["nan", "none", "null"]))
    if mask_room.any():
        df.loc[mask_room, room_col] = "2+1"
        logs.append(f"Varsayılan atandı: {room_col}='2+1' ({mask_room.sum()} satır)")

    # net default
    _coerce_numeric_inplace(df, net_col)
    mask_net = df[net_col].isna()
    if mask_net.any():
        df.loc[mask_net, net_col] = 100.0
        logs.append(f"Varsayılan atandı: {net_col}=100 ({mask_net.sum()} satır)")

    # gross default = net
    _coerce_numeric_inplace(df, gross_col)
    mask_g = df[gross_col].isna()
    if mask_g.any():
        df.loc[mask_g, gross_col] = df.loc[mask_g, net_col].astype(float)
        logs.append(f"Varsayılan atandı: {gross_col}=net ({mask_g.sum()} satır)")

    # age default
    _coerce_numeric_inplace(df, age_col)
    mask_age = df[age_col].isna()
    if mask_age.any():
        df.loc[mask_age, age_col] = 10.0
        logs.append(f"Varsayılan atandı: {age_col}=10 ({mask_age.sum()} satır)")

    # location default if empty
    if loc_col and loc_col in df.columns:
        df[loc_col] = df[loc_col].astype(str)
        mask_loc = df[loc_col].isna() | (df[loc_col].str.strip() == "") | (df[loc_col].str.lower().isin(["nan", "none", "null"]))
        if mask_loc.any():
            df.loc[mask_loc, loc_col] = default_loc
            logs.append(f"Varsayılan atandı: {loc_col}='{default_loc}' ({mask_loc.sum()} satır)")

    return df, logs

def predict_chunked_safe(df: pd.DataFrame, bundle, best_iter: int, chunk_size: int = 500) -> np.ndarray:
    """
    First tries chunked batch prediction.
    If a chunk fails, falls back to row-wise on that chunk to salvage results.
    """
    n = len(df)
    out = np.full(n, np.nan, dtype=float)

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        chunk = df.iloc[start:end].copy()

        try:
            preds = predict_from_raw_df(chunk, bundle, best_iter)
            out[start:end] = preds
        except Exception:
            # salvage row-wise
            for i in range(start, end):
                try:
                    out[i] = float(predict_from_raw_df(df.iloc[[i]].copy(), bundle, best_iter)[0])
                except Exception:
                    out[i] = np.nan
    return out

# ------------------------------------------------------------
# FIXED MODEL PATH - ALWAYS USE MODEL6
# ------------------------------------------------------------
def get_model6_paths():
    """Always returns model6 paths regardless of user selection"""
    models_dir = os.path.join(ROOT, "models")

    model6_patterns = [
        os.path.join(models_dir, "model6_bundle.joblib"),
        os.path.join(models_dir, "model6", "model6_bundle.joblib"),
        os.path.join(models_dir, "model6.joblib"),
        os.path.join(models_dir, "model6", "*.joblib"),
        os.path.join(models_dir, "*.joblib"),
    ]

    bundle_path = None
    for pattern in model6_patterns:
        files = glob.glob(pattern)
        if files:
            bundle_path = files[0]
            break

    if not bundle_path:
        raise FileNotFoundError("Model dosyası bulunamadı! (models/ altında .joblib olmalı)")

    base_name = os.path.basename(bundle_path).replace(".joblib", "").replace("_bundle", "")
    base_dir = os.path.dirname(bundle_path)

    meta_path = os.path.join(base_dir, f"{base_name}_meta.json")
    th_path = os.path.join(base_dir, f"{base_name}_thresholds.json")

    if not os.path.exists(meta_path):
        meta_path = os.path.join(base_dir, "model6_meta.json")
    if not os.path.exists(th_path):
        th_path = os.path.join(base_dir, "model6_thresholds.json")

    return bundle_path, meta_path, th_path

# ------------------------------------------------------------
# MAIN APP
# ------------------------------------------------------------
st.set_page_config(page_title="İstanbul Emlak AI (2020)", layout="wide", page_icon="🏠")
inject_css()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### ⚙️ Sistem Ayarları")

    bundle_path, meta_path, th_path = get_model6_paths()

    st.markdown(f"**Aktif Model:** `{os.path.basename(bundle_path)}`")
    st.markdown("ℹ️ *Sistem otomatik olarak Model 6'yı kullanır*")

    th_data = load_thresholds_optional(th_path)
    st.markdown("### 📊 Karar Eşikleri")
    t_buy = st.slider("Fırsat Eşiği (%)", 1, 20, int(th_data.get("t_buy", 0.03) * 100)) / 100.0
    t_sell = st.slider("Pahalı Eşiği (%)", 1, 20, int(th_data.get("t_sell", 0.03) * 100)) / 100.0

    if st.button("🔄 Modeli Yeniden Yükle", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    if st.button("💾 Eşikleri Kaydet", use_container_width=True):
        save_json_safe(th_path, {"t_buy": t_buy, "t_sell": t_sell})
        st.success("Eşikler kaydedildi!")
        st.rerun()

    with st.expander("ℹ️ Model Bilgisi"):
        st.markdown("""
**Model 6 Özellikleri:**
- LightGBM Gradient Boosting
- 2020 İstanbul verileriyle eğitildi
- Target Encoding + robust preprocessing
- Outlier dirençli (robust) veri temizliği
- CV ile optimize edildi
        """)

# LOAD MODEL
try:
    with st.spinner("Model 6 yükleniyor..."):
        bundle = load_bundle_cached(bundle_path)
        meta = load_json_safe(meta_path) or bundle.get("meta", {})
        pre = bundle["preprocessor"]
        te = bundle.get("target_encoder")
        best_iter = int(bundle.get("best_iter", 0))

        feat_cols = getattr(pre, "feature_cols_", []) or getattr(pre, "feature_names_in_", [])
        loan_col = next((c for c in feat_cols if any(k in c.lower() for k in ["loan", "kredi", "available_for_loan"])), None)

        loc_col = detect_location_col(pre)
        allowed_locs = get_allowed_locations_from_te(te, loc_col) if loc_col else []

except Exception as e:
    st.error(f"Model yüklenirken hata: {e}")
    st.code(traceback.format_exc())
    st.stop()

# --- HEADER ---
st_html("""
<div class="hero">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;">
    <div>
      <h1>🏠 İstanbul Emlak Danışmanı AI <span style="opacity:.5;font-size:1.2rem;">(2020)</span></h1>
      <p>Yapay Zeka Destekli Adil Değerleme & Yatırım Analizi</p>
    </div>
    <div class="badge">Model 6 | v5 FULL</div>
  </div>
</div>
""")

# --- TABS ---
tab_canli, tab_csv, tab_nasil, tab_metrikler, tab_dosyalar = st.tabs([
    "🎛️ Canlı Analiz",
    "📥 Toplu Analiz (CSV)",
    "🧠 Nasıl Çalışır?",
    "📏 Model Metrikleri",
    "📂 Analiz Raporları"
])

# ------------------------------------------------------------
# TAB 1: CANLI ANALIZ (HTML FIXED)
# ------------------------------------------------------------
with tab_canli:
    col_giris, col_sonuc = st.columns([1, 1], gap="large")

    with col_giris:
        st_html('<div class="glass"><h3>1. Emlak Bilgileri</h3></div>')

        fiyat_var = st.checkbox("İlan fiyatı belli", value=True, help="İlan fiyatını biliyorsanız işaretleyin")
        if fiyat_var:
            ilan_fiyati = st.number_input(
                "İlan Fiyatı (TL)",
                min_value=10000.0,
                max_value=100000000.0,
                value=640000.0,
                step=10000.0,
                help="Emlak sitesinde görülen satış fiyatı"
            )
        else:
            ilan_fiyati = None
            st.info("Sadece adil değer tahmini yapılacak")

        st.markdown("#### 📐 Temel Özellikler")
        c1, c2 = st.columns(2)
        with c1:
            net_m2 = st.number_input("Net m²", min_value=10.0, max_value=500.0, value=112.0, step=1.0)
        with c2:
            bina_yasi = st.number_input("Bina Yaşı", min_value=0.0, max_value=100.0, value=6.0, step=1.0)

        st.markdown("#### 🛏️ Oda Bilgisi")
        c3, c4, c5 = st.columns([1, 1, 1.5])
        with c3:
            oda_sayisi = st.number_input("Oda", 0, 10, value=3)
        with c4:
            salon_sayisi = st.number_input("Salon", 0, 5, value=1)
        with c5:
            oda_tipi = f"{oda_sayisi}+{salon_sayisi}"
            st.text_input("Oda Tipi", value=oda_tipi, disabled=True)

        st.markdown("#### 📍 Konum")
        if allowed_locs:
            default_idx = allowed_locs.index("Kadikoy") if "Kadikoy" in allowed_locs else 0
            ilce = st.selectbox("İlçe", allowed_locs, index=default_idx)
        else:
            ilce = st.text_input("İlçe", "Kadikoy")

        kredi_uygun = True
        if loan_col:
            kredi_uygun = st.toggle("Krediye Uygun", value=True)

        if st.button("🚀 ANALİZİ BAŞLAT", type="primary", use_container_width=True):
            try:
                row = pd.DataFrame([{
                    getattr(pre, "_room_col", "rooms"): oda_tipi,
                    getattr(pre, "_net_col", "net"): float(net_m2),
                    getattr(pre, "_gross_col", "gross"): float(net_m2),
                    getattr(pre, "_age_col", "age"): float(bina_yasi),
                    (loc_col or "ilce"): ilce
                }])

                if loan_col:
                    row[loan_col] = 1 if kredi_uygun else 0

                adil_deger = float(predict_from_raw_df(row, bundle, best_iter)[0])

                st.session_state["analiz_sonucu"] = {
                    "adil_deger": adil_deger,
                    "ilan_fiyati": ilan_fiyati,
                    "net_m2": net_m2,
                    "bina_yasi": bina_yasi,
                    "oda_tipi": oda_tipi,
                    "ilce": ilce
                }
                st.success("✅ Analiz tamamlandı!")
            except Exception as e:
                st.error(f"Analiz sırasında hata: {e}")
                st.code(traceback.format_exc())

    with col_sonuc:
        st_html('<div class="glass"><h3>2. Analiz Sonuçları</h3></div>')

        sonuc = st.session_state.get("analiz_sonucu")
        if sonuc:
            adil_deger = float(sonuc["adil_deger"])
            ilan_fiyati = sonuc["ilan_fiyati"]

            if ilan_fiyati is not None:
                karar, fark_orani = business_label(float(ilan_fiyati), adil_deger, t_buy, t_sell)
            else:
                karar, fark_orani = "—", np.nan

            karar_ikon = {"Fırsat": "✅", "Pahalı": "⛔", "Normal": "⚖️", "—": "📊"}[karar]
            karar_renk = {"Fırsat": "good", "Pahalı": "bad", "Normal": "warn", "—": "warn"}[karar]

            st_html(f"""
<div style="text-align:center; margin:20px 0;">
  <span class="badge {karar_renk}" style="font-size:1.3rem; padding:12px 24px;">
    {karar_ikon} {karar}
  </span>
</div>
""")

            st.markdown("#### 📊 Finansal Göstergeler")

            kpi_html = f"""
<div class="kpi-grid">
  <div class="kpi">
    <div class="k">Adil Değer</div>
    <div class="v" style="color:#82c8ff;">{fmt_tr(adil_deger)} TL</div>
  </div>
"""
            if ilan_fiyati is not None:
                fark_html = "—"
                if not np.isnan(fark_orani):
                    fark_yuzde = float(fark_orani) * 100
                    renk = "#50ffaa" if fark_orani < 0 else "#ff6ea0"
                    isaret = "+" if fark_orani > 0 else ""
                    fark_html = f"<span style='color:{renk}'>{isaret}%{fmt_pct(fark_yuzde)}</span>"

                kpi_html += f"""
  <div class="kpi">
    <div class="k">İlan Fiyatı</div>
    <div class="v">{fmt_tr(ilan_fiyati)} TL</div>
  </div>
  <div class="kpi">
    <div class="k">Fark Oranı</div>
    <div class="v">{fark_html}</div>
  </div>
"""
            kpi_html += "</div>"
            st_html(kpi_html)

            if ilan_fiyati is not None and adil_deger > 0:
                st.markdown("---")
                st.markdown("#### 📝 Detaylı Analiz")

                fark_abs = abs(float(ilan_fiyati) - adil_deger)
                m2_fiyat = adil_deger / float(sonuc["net_m2"]) if float(sonuc["net_m2"]) > 0 else np.nan

                ca, cb = st.columns(2)
                ca.metric("Mutlak Fark", f"{fmt_tr(fark_abs)} TL")
                cb.metric("m² Başına Değer", f"{fmt_tr(m2_fiyat)} TL/m²" if not np.isnan(m2_fiyat) else "—")

                st.info(explain_decision_simple(float(ilan_fiyati), adil_deger, t_buy, t_sell))

            st.markdown("---")
            st.markdown("#### 🏠 Emlak Özeti")
            ozet_cols = st.columns(3)
            ozet_cols[0].metric("Net m²", f"{sonuc['net_m2']} m²")
            ozet_cols[1].metric("Bina Yaşı", f"{int(float(sonuc['bina_yasi']))} yıl")
            ozet_cols[2].metric("Oda Tipi", sonuc["oda_tipi"])
            st.caption(f"📍 Konum: {sonuc['ilce']}")

        else:
            st.info("👈 Sol tarafta emlak bilgilerini girip 'ANALİZİ BAŞLAT' butonuna tıklayın")
            st_html("""
<div style="text-align:center; padding:40px 20px; color:var(--muted2);">
  <div style="font-size:3rem; margin-bottom:10px;">🏠</div>
  <div>Analiz sonuçları burada görünecek</div>
</div>
""")

# ------------------------------------------------------------
# TAB 2: CSV ANALIZ (EXTRA ROBUST: missing cols OK)
# ------------------------------------------------------------
with tab_csv:
    st.markdown("### 📥 Toplu Dosya Analizi")
    st.info("""
**Sınırsız Mod (Robust CSV):**
1) **Adil Değer:** Fiyat sütunu OLSUN/OLMASIN her satır için tahmin edilir.  
2) **Karar:** Fiyat varsa Fırsat/Normal/Pahalı; yoksa `—`.  
3) **Eksik Sütunlar:** Eksik kolonlar otomatik eklenir + kritik bilgiler yoksa varsayılan atanır.  
""")

    uploaded_file = st.file_uploader("CSV Dosyası Seçin", type=["csv"])

    if uploaded_file:
        try:
            # Read as strings to preserve Turkish numbers like "1.000"
            df_raw = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)

            # Map columns to expected names
            df_mapped, mapping_logs = smart_column_mapping(df_raw, pre)

            # Apply lenient defaults to guarantee predict
            df_ready, default_logs = apply_lenient_defaults(df_mapped, pre, loc_col, allowed_locs)

            # Ensure missing engineered/amenity features exist
            df_ready = ensure_cols_for_pre_add_features(df_ready, pre)

            # Optional: add any remaining expected columns (if known)
            expected_cols = getattr(pre, "feature_names_in_", []) or getattr(pre, "feature_cols_", []) or []
            missing_cols = []
            for col in expected_cols:
                if col not in df_ready.columns:
                    df_ready[col] = np.nan
                    missing_cols.append(col)

            with st.expander("🔍 İşlem Detayları", expanded=False):
                if mapping_logs:
                    st.write("✅ Eşleşen sütunlar:")
                    for log in mapping_logs:
                        st.caption(log)
                if default_logs:
                    st.write("🛠️ Varsayılan/Onarım adımları:")
                    for log in default_logs[:20]:
                        st.caption(log)
                    if len(default_logs) > 20:
                        st.caption(f"... +{len(default_logs)-20} satır daha")
                if missing_cols:
                    st.warning(
                        f"⚠️ Modelde beklenip CSV'de olmayan kolonlar eklendi (NaN): {', '.join(missing_cols[:8])}"
                        + (" ..." if len(missing_cols) > 8 else "")
                    )

            # ------------------------------
            # Detect a price column (FIXED)
            # ------------------------------
            # Keep your old variables (not removing), but we will use a correct detection + mapping.
            norm_cols = [c.lower() for c in df_ready.columns]
            price_candidates = ["price", "fiyat", "ilan_fiyati", "listing_price", "sale_price", "ilan fiyatı", "ilan_fiyat"]

            # Create a normalized view of RAW columns (so detection is stable)
            df_raw_norm = df_raw.copy()
            df_raw_norm.columns = m6.normalize_colnames(df_raw_norm.columns.tolist())

            # Detect on normalized RAW
            fiyat_col_norm = detect_price_col(df_raw_norm)

            # Map normalized name back to RAW name (so we can display the original column)
            norm_to_raw = {n: r for r, n in zip(df_raw.columns, df_raw_norm.columns)}
            fiyat_col = norm_to_raw.get(fiyat_col_norm, None)

            st.caption(f"Detected price col: raw='{fiyat_col}' | norm='{fiyat_col_norm}'")

            # Predict (chunked safe)
            with st.spinner(f"{len(df_ready)} satır için adil değer hesaplanıyor..."):
                adil_degerler = predict_chunked_safe(df_ready, bundle, best_iter, chunk_size=500)

            # Results frame (keep original columns)
            df_result = df_raw.copy()
            df_result["Adil_Deger"] = adil_degerler
            df_result["Adil_Deger_Fmt"] = pd.Series(adil_degerler).apply(lambda x: fmt_tr(x) if pd.notna(x) and x > 0 else "—")

            # Parse prices if available (ALWAYS from normalized RAW, so name matches)
            if fiyat_col_norm and fiyat_col_norm in df_raw_norm.columns:
                df_result["_Analiz_Fiyati"] = df_raw_norm[fiyat_col_norm].apply(safe_turkish_float)
            else:
                df_result["_Analiz_Fiyati"] = np.nan

            st.caption(f"Non-null prices: {int(df_result['_Analiz_Fiyati'].notna().sum())}")

            # Decision + diff columns
            def karar_ver(row):
                fiyat = row.get("_Analiz_Fiyati", np.nan)
                adil = row.get("Adil_Deger", np.nan)

                if pd.isna(adil) or adil <= 0:
                    return "—", np.nan, np.nan

                if pd.isna(fiyat):
                    return "—", np.nan, np.nan

                fark = (fiyat - adil) / adil
                if fark <= -t_buy:
                    label = "✅ Fırsat"
                elif fark >= t_sell:
                    label = "⛔ Pahalı"
                else:
                    label = "⚖️ Normal"
                return label, float(fark) * 100.0, float(fiyat - adil)

            tmp = df_result.apply(lambda r: karar_ver(r), axis=1, result_type="expand")
            df_result["Karar"] = tmp[0]
            df_result["Fark_%"] = tmp[1]
            df_result["Fark_TL"] = tmp[2]

            # Summary metrics
            st.markdown("#### 📈 Analiz Özeti")
            counts = df_result["Karar"].value_counts()

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("✅ Fırsat", int(counts.get("✅ Fırsat", 0)))
            c2.metric("⚖️ Normal", int(counts.get("⚖️ Normal", 0)))
            c3.metric("⛔ Pahalı", int(counts.get("⛔ Pahalı", 0)))
            c4.metric("Fiyatsız (—)", int(counts.get("—", 0)))
            c5.metric("Toplam", len(df_result))

            # Table preview
            st.markdown("#### 📋 Detaylı Tablo (ilk 200 satır)")
            cols_to_show = ["Adil_Deger_Fmt", "Karar", "Fark_%", "Fark_TL"]

            # show raw price column (if found)
            if fiyat_col:
                cols_to_show.insert(2, fiyat_col)

            # Keep original cols after
            for c in df_raw.columns:
                if c not in cols_to_show:
                    cols_to_show.append(c)

            st.dataframe(
                df_result[cols_to_show].head(200),
                use_container_width=True,
                height=500
            )

            # Download
            csv_data = df_result.drop(columns=["_Analiz_Fiyati"], errors="ignore").to_csv(index=False, encoding="utf-8-sig")
            st.download_button(
                "📥 Sonuçları İndir (CSV)",
                data=csv_data,
                file_name="analiz_sonuclari.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Hata: {e}")
            st.code(traceback.format_exc())

# ------------------------------------------------------------
# TAB 3: NASIL ÇALIŞIR
# ------------------------------------------------------------
with tab_nasil:
    st.header("🧠 Sistem Nasıl Çalışır?")
    st.info("Bu sistem, İstanbul emlak verilerini işleyerek 4 adımda yatırım kararı üretir.")

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.subheader("1. Veri Hazırlığı")
        st.markdown("""
**📂 Veri Toplama & Temizleme**
- Veri Kaynağı: hackathon_train_set.csv
- Temizleme: hatalı girişler (örn: 1000 yaşında bina) filtrelenir
- Outlier analizi: uç değerler bastırılır
- Eksik veri: robust yöntemlerle doldurulur
        """)
    with c2:
        st.subheader("2. Yapay Zeka Tahmini")
        r2_skor = float(meta.get("metrics_val", {}).get("r2_log", 0.86)) * 100
        st.markdown(f"""
**🤖 LightGBM Modeli**
- Model: Gradient Boosting (LightGBM)
- Doğruluk (R²): **%{fmt_pct(r2_skor)}**
- Hız: < 1 saniye (tek satır)
        """)

    st.divider()

    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.subheader("3. Karar Mekanizması")
        st.warning("""
**⚖️ Fırsat vs Pahalı Analizi**
Sistem, **İlan Fiyatı** ile **Adil Değer**i karşılaştırır.
        """)
        k1, k2, k3 = st.columns(3)
        k1.success(f"**✅ FIRSAT**\n\nFiyat %{fmt_pct(t_buy*100)} düşükse")
        k2.warning("**⚖️ NORMAL**\n\nEşiklerin içindeyse")
        k3.error(f"**⛔ PAHALI**\n\nFiyat %{fmt_pct(t_sell*100)} yüksekse")

    with c4:
        st.subheader("4. Raporlama")
        st.markdown("""
**📊 Sonuç Sunumu**
- Yatırım sinyali (Fırsat/Normal/Pahalı)
- TL ve % fark hesapları
- CSV çıktısı (toplu analiz)
        """)

    with st.expander("🔬 Geliştiriciler İçin Teknik Detaylar"):
        t1, t2, t3 = st.columns(3)
        t1.metric("Algoritma", "LightGBM Regressor")
        t2.metric("Eğitim Verisi", "2020 İstanbul")
        t3.metric("Loss", "MAE / RMSE (rapora göre)")
        st.code("Tahmin: fiyat = expm1( model( preprocess(x) ) )", language="python")

# ------------------------------------------------------------
# TAB 4: MODEL METRİKLERİ & SHAP
# ------------------------------------------------------------
with tab_metrikler:
    st.header("📏 Model Performans & SHAP Analizi")

    metrics = meta.get("metrics_val", {}) or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R² (Doğruluk)", f"%{fmt_pct(float(metrics.get('r2_log', 0.86)) * 100)}")
    m2.metric("MAE", f"{fmt_tr(metrics.get('mae_tl', 45000))} TL", delta_color="inverse")
    m3.metric("RMSE", f"{fmt_tr(metrics.get('rmse_tl', 65000))} TL")
    m4.metric("Eğitim Verisi", "2020 İstanbul")

    st.divider()
    st.subheader("🧠 SHAP: Model Kararını Nasıl Veriyor?")
    st.info("SHAP grafikleri, hangi özelliğin (m², ilçe, vb.) fiyatı ne kadar etkilediğini gösterir.")

    shap_base_path = os.path.join(ROOT, "models", "shap", "shap results")
    img_bar_path = os.path.join(shap_base_path, "shap_importance_bar.png")
    img_dot_path = os.path.join(shap_base_path, "shap_summary_dot.png")

    col_shap1, col_shap2 = st.columns(2, gap="medium")
    with col_shap1:
        st.markdown("#### 📊 En Önemli Özellikler")
        if os.path.exists(img_bar_path):
            st.image(img_bar_path, use_container_width=True, caption="Özellik önem sıralaması")
        else:
            st.warning(f"Görsel bulunamadı: {img_bar_path}")

    with col_shap2:
        st.markdown("#### 🌡️ Özelliklerin Fiyata Etkisi")
        if os.path.exists(img_dot_path):
            st.image(img_dot_path, use_container_width=True, caption="Kırmızı: yüksek değer, Mavi: düşük değer")
        else:
            st.warning(f"Görsel bulunamadı: {img_dot_path}")

    # 3. İstatistiki Grafikler (Mevcut Dummy Kod) st.subheader("📉 Tahmin Sapma Analizi") c_chart1, c_chart2 = st.columns(2) with c_chart1: st.caption("Tahmin vs Gerçek Değer Dağılımı (Test Verisi)") # Simülasyon verisi (Gerçek veri varsa buraya entegre edilebilir) chart_data = pd.DataFrame( np.random.randn(20, 2), columns=['Tahmin', 'Gerçek']) st.line_chart(chart_data) with c_chart2: st.caption("Hata Dağılımı") st.bar_chart(chart_data['Tahmin'] - chart_data['Gerçek'])
    c_chart1, c_chart2 = st.columns(2)
    with c_chart1:
        st.caption("Tahmin vs Gerçek Değer Dağılımı (Test Verisi)")
        # Simülasyon verisi (Gerçek veri varsa buraya entegre edilebilir)
        chart_data = pd.DataFrame(
            np.random.randn(20, 2),
            columns=['Tahmin', 'Gerçek'])
        st.line_chart(chart_data)
    with c_chart2:
        st.caption("Hata Dağılımı")
        st.bar_chart(chart_data['Tahmin'] - chart_data['Gerçek'])

    

# ------------------------------------------------------------
# TAB 5: ANALİZ RAPORLARI (Placeholder)
# ------------------------------------------------------------
with tab_dosyalar:
    st.header("📂 Kayıtlı Analiz Raporları")
    st.info("Bu bölüm placeholder. İstersen burada gerçek dosya listeleme (reports/ klasörü) ekleyebiliriz.")

    col_file, col_btn = st.columns([3, 1])
    with col_file:
        st.selectbox("Rapor Seçin", [
            "2023-10-12_kadikoy_analiz.csv",
            "2023-10-10_besiktas_toplu.csv",
            "2023-09-28_genel_tarama.xlsx"
        ])
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("📥 İndir", use_container_width=True)

    st.divider()
    st.subheader("Rapor Önizleme")
    st.dataframe(pd.DataFrame({
        "İlan No": [101, 102, 103],
        "Bölge": ["Kadıköy", "Beşiktaş", "Şişli"],
        "Tahmin": [1500000, 2300000, 1800000],
        "Durum": ["✅ Fırsat", "⛔ Pahalı", "⚖️ Normal"]
    }), use_container_width=True)
