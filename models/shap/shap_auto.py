import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
import os
import re

# ==========================================
# USER SETTINGS
# ==========================================
DATA_PATH = r".\data\hackathon_train_set.csv" 
OUT_DIR = r".\shap_results_auto"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_price_to_float(x):
    if pd.isna(x): return np.nan
    s = str(x)
    # Remove currency symbols, keep digits and dots/commas
    digits = re.sub(r"[^\d]", "", s)
    return float(digits) if digits else np.nan

def parse_rooms(value):
    if pd.isna(value): return np.nan
    s = str(value).strip()
    m = re.match(r"^\s*(\d+)\s*\+\s*(\d+)\s*$", s)
    if m:
        return float(int(m.group(1)) + int(m.group(2)))
    try:
        return float(s)
    except:
        return np.nan

def robust_load_csv(path):
    # Try reading with different separators
    try:
        df = pd.read_csv(path, sep=',', low_memory=False)
        if len(df.columns) <= 1:
            raise ValueError("Comma didn't work")
    except:
        try:
            print("[INFO] Comma failed, trying semicolon (;)...")
            df = pd.read_csv(path, sep=';', low_memory=False)
        except:
            print("[INFO] Semicolon failed, trying tab...")
            df = pd.read_csv(path, sep='\t', low_memory=False)
    
    # Clean column names (remove spaces)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_price_column(df):
    # List of possible names for the target
    candidates = ["Price", "price", "Fiyat", "fiyat", "PRICE", "Target", "target", "Salary", "bedel"]
    
    for c in candidates:
        if c in df.columns:
            return c
    return None

def quick_preprocess(df):
    print("[INFO] Cleaning data...")
    df = df.copy()
    
    # 1. Find and Clean Target
    price_col = find_price_column(df)
    
    if price_col:
        print(f"[INFO] Found target column: '{price_col}'")
        df["Price_num"] = df[price_col].apply(clean_price_to_float)
        df = df[df["Price_num"].notna()]
        y = np.log1p(df["Price_num"]) # Log transform
        X = df.drop(columns=[price_col, "Price_num"])
    else:
        print("\n[CRITICAL ERROR] Could not find a 'Price' column.")
        print(f"Columns found in CSV: {list(df.columns)}")
        print("Please check if your CSV uses a different name for the target.\n")
        return None, None

    # 2. Feature Engineering (Basic)
    room_cols = [c for c in X.columns if "room" in c.lower()]
    if room_cols:
        X["total_rooms"] = X[room_cols[0]].apply(parse_rooms)

    net_col = next((c for c in X.columns if "net" in c.lower() and ("m2" in c.lower() or "area" in c.lower())), None)
    gross_col = next((c for c in X.columns if "gross" in c.lower() and ("m2" in c.lower() or "area" in c.lower())), None)
    
    if net_col: X[net_col] = pd.to_numeric(X[net_col], errors='coerce')
    if gross_col: X[gross_col] = pd.to_numeric(X[gross_col], errors='coerce')

    # 3. Handle Categories
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
            
    return X, y

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    print(f"[INFO] Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print(f"[CRITICAL] File not found: {DATA_PATH}")
        return

    # Load with smart separator detection
    df = robust_load_csv(DATA_PATH)

    # Preprocess
    X, y = quick_preprocess(df)
    if X is None: return

    print(f"[INFO] Training Data Shape: {X.shape}")

    # Train Fresh Model
    print("[INFO] Training LightGBM model (fast)...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(X, y)

    # Run SHAP
    print("[INFO] Calculating SHAP values (using 2000 samples)...")
    explainer = shap.TreeExplainer(model)
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    shap_values = explainer.shap_values(X_sample)

    # Plot
    print(f"[INFO] Saving plots to: {OUT_DIR}")
    
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(os.path.join(OUT_DIR, "shap_summary_dot.png"), bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig(os.path.join(OUT_DIR, "shap_importance_bar.png"), bbox_inches='tight')
    plt.close()

    print("\n[SUCCESS] ! Your SHAP plots are finally ready.")

if __name__ == "__main__":
    main()