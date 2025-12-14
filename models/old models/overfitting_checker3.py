# overfitting_checker3.py
# Comprehensive overfitting checker for model6_balanced
# Tests the model on multiple validation strategies to ensure results aren't just lucky

import os
import sys
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb

# Import functions from model6_balanced
from model6 import (
    set_thread_env, log, robust_read_csv, normalize_colnames,
    clean_price_to_float, filter_loan_eligible, Preprocessor,
    RobustTargetEncoder, eval_reg, interpret_overfit_index
)


@dataclass
class ValidationResult:
    """Store results from a single validation fold"""
    fold: int
    train_rmse_log: float
    val_rmse_log: float
    train_r2_log: float
    val_r2_log: float
    train_rmse_tl: float
    val_rmse_tl: float
    overfit_index: float
    fold_type: str


@dataclass
class ValidationSummary:
    """Summary statistics across all validation folds"""
    mean_train_rmse_log: float
    mean_val_rmse_log: float
    mean_overfit_index: float
    std_overfit_index: float
    min_overfit_index: float
    max_overfit_index: float
    stability_score: float  # 1 - (std/mean), higher is better
    results: List[ValidationResult]


def load_model_bundle(bundle_path: str):
    """Load the trained model bundle"""
    log(f"Loading model bundle: {bundle_path}")
    bundle = joblib.load(bundle_path)
    return bundle


def load_and_preprocess_data(data_path: str, preprocessor, target_encoder, bundle):
    """Load and preprocess data using the same pipeline as training"""
    df = robust_read_csv(data_path)
    df.columns = normalize_colnames(df.columns)
    
    # Clean price
    df["Price_num"] = df["Price"].apply(clean_price_to_float)
    df = df[df["Price_num"].notna()].copy()
    
    # Apply loan filter
    df, did_filter, loan_col = filter_loan_eligible(df)
    
    # Apply same clipping as training
    lo = bundle["meta"]["clip_lo_value"]
    hi = bundle["meta"]["clip_hi_value"]
    y_tl_raw = df["Price_num"].astype(float).to_numpy()
    y_tl_clip = np.clip(y_tl_raw, lo, hi)
    y_log = np.log1p(y_tl_clip).astype(float)
    
    X = df.drop(columns=["Price", "Price_num"], errors="ignore").copy()
    
    # Preprocess
    X_processed = preprocessor.transform(X)
    if target_encoder is not None:
        X_processed = target_encoder.transform(X_processed)
    
    return X_processed, y_log, y_tl_clip, df.index.tolist()


def kfold_cross_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    model_params: Dict[str, Any],
    n_splits: int = 5,
    seed: int = 42,
    threads: int = 10
) -> ValidationSummary:
    """
    Perform K-Fold cross validation to check model stability
    """
    set_thread_env(threads)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    results = []
    fold_scores = []
    
    log(f"Starting {n_splits}-Fold Cross Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        log(f"  Fold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train new model with same parameters
        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=800,
            verbose=False
        )
        
        # Get best iteration
        best_iter = getattr(model, "best_iteration_", 0) or model_params.get("n_estimators", 15000)
        
        # Predictions
        pred_train = model.predict(X_train, num_iteration=best_iter)
        pred_val = model.predict(X_val, num_iteration=best_iter)
        
        # Calculate metrics
        train_metrics = eval_reg(y_train, pred_train)
        val_metrics = eval_reg(y_val, pred_val)
        
        overfit_idx = val_metrics["rmse_log"] / max(train_metrics["rmse_log"], 1e-12)
        fold_scores.append(overfit_idx)
        
        results.append(ValidationResult(
            fold=fold,
            train_rmse_log=train_metrics["rmse_log"],
            val_rmse_log=val_metrics["rmse_log"],
            train_r2_log=train_metrics["r2_log"],
            val_r2_log=val_metrics["r2_log"],
            train_rmse_tl=train_metrics["rmse_tl"],
            val_rmse_tl=val_metrics["rmse_tl"],
            overfit_index=overfit_idx,
            fold_type="kfold"
        ))
        
        log(f"    Fold {fold}: Overfit Index = {overfit_idx:.3f}")
    
    # Calculate summary statistics
    mean_train_rmse = np.mean([r.train_rmse_log for r in results])
    mean_val_rmse = np.mean([r.val_rmse_log for r in results])
    mean_o_idx = np.mean(fold_scores)
    std_o_idx = np.std(fold_scores)
    
    stability = 1 - (std_o_idx / mean_o_idx) if mean_o_idx > 0 else 0
    
    summary = ValidationSummary(
        mean_train_rmse_log=mean_train_rmse,
        mean_val_rmse_log=mean_val_rmse,
        mean_overfit_index=mean_o_idx,
        std_overfit_index=std_o_idx,
        min_overfit_index=np.min(fold_scores),
        max_overfit_index=np.max(fold_scores),
        stability_score=stability,
        results=results
    )
    
    return summary


def time_based_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    dates: pd.Series,
    model_params: Dict[str, Any],
    test_size: float = 0.2,
    threads: int = 10
) -> ValidationSummary:
    """
    Time-based validation: train on older data, test on newer data
    """
    set_thread_env(threads)
    
    if dates is None or len(dates) != len(X):
        log("No date information for time-based validation")
        return None
    
    # Sort by date
    sorted_idx = dates.argsort()
    X_sorted = X.iloc[sorted_idx]
    y_sorted = y[sorted_idx]
    
    # Split: 80% oldest for training, 20% newest for validation
    split_idx = int(len(X_sorted) * (1 - test_size))
    
    X_train = X_sorted.iloc[:split_idx]
    X_val = X_sorted.iloc[split_idx:]
    y_train = y_sorted[:split_idx]
    y_val = y_sorted[split_idx:]
    
    log(f"Time-based validation: Train on {len(X_train)} oldest, Test on {len(X_val)} newest")
    
    # Train model
    model = lgb.LGBMRegressor(**model_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=800,
        verbose=False
    )
    
    best_iter = getattr(model, "best_iteration_", 0) or model_params.get("n_estimators", 15000)
    
    # Predictions
    pred_train = model.predict(X_train, num_iteration=best_iter)
    pred_val = model.predict(X_val, num_iteration=best_iter)
    
    # Calculate metrics
    train_metrics = eval_reg(y_train, pred_train)
    val_metrics = eval_reg(y_val, pred_val)
    
    overfit_idx = val_metrics["rmse_log"] / max(train_metrics["rmse_log"], 1e-12)
    
    result = ValidationResult(
        fold=1,
        train_rmse_log=train_metrics["rmse_log"],
        val_rmse_log=val_metrics["rmse_log"],
        train_r2_log=train_metrics["r2_log"],
        val_r2_log=val_metrics["r2_log"],
        train_rmse_tl=train_metrics["rmse_tl"],
        val_rmse_tl=val_metrics["rmse_tl"],
        overfit_index=overfit_idx,
        fold_type="time_based"
    )
    
    summary = ValidationSummary(
        mean_train_rmse_log=train_metrics["rmse_log"],
        mean_val_rmse_log=val_metrics["rmse_log"],
        mean_overfit_index=overfit_idx,
        std_overfit_index=0,
        min_overfit_index=overfit_idx,
        max_overfit_index=overfit_idx,
        stability_score=1.0,
        results=[result]
    )
    
    return summary


def stratified_price_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    y_tl: np.ndarray,
    model_params: Dict[str, Any],
    n_splits: int = 5,
    seed: int = 42,
    threads: int = 10
) -> ValidationSummary:
    """
    Stratified validation based on price bins to ensure distribution consistency
    """
    set_thread_env(threads)
    
    # Create price bins for stratification
    n_bins = min(10, len(np.unique(y_tl)) // 10)
    if n_bins < 2:
        n_bins = 2
    
    price_bins = pd.qcut(y_tl, q=n_bins, labels=False, duplicates='drop')
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    results = []
    fold_scores = []
    
    log(f"Starting Stratified Price Validation ({n_splits} folds)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, price_bins), 1):
        log(f"  Fold {fold}/{n_splits}")
        
        # Split data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=800,
            verbose=False
        )
        
        best_iter = getattr(model, "best_iteration_", 0) or model_params.get("n_estimators", 15000)
        
        # Predictions
        pred_train = model.predict(X_train, num_iteration=best_iter)
        pred_val = model.predict(X_val, num_iteration=best_iter)
        
        # Calculate metrics
        train_metrics = eval_reg(y_train, pred_train)
        val_metrics = eval_reg(y_val, pred_val)
        
        overfit_idx = val_metrics["rmse_log"] / max(train_metrics["rmse_log"], 1e-12)
        fold_scores.append(overfit_idx)
        
        results.append(ValidationResult(
            fold=fold,
            train_rmse_log=train_metrics["rmse_log"],
            val_rmse_log=val_metrics["rmse_log"],
            train_r2_log=train_metrics["r2_log"],
            val_r2_log=val_metrics["r2_log"],
            train_rmse_tl=train_metrics["rmse_tl"],
            val_rmse_tl=val_metrics["rmse_tl"],
            overfit_index=overfit_idx,
            fold_type="stratified_price"
        ))
        
        log(f"    Fold {fold}: Overfit Index = {overfit_idx:.3f}")
    
    # Calculate summary
    mean_train_rmse = np.mean([r.train_rmse_log for r in results])
    mean_val_rmse = np.mean([r.val_rmse_log for r in results])
    mean_o_idx = np.mean(fold_scores)
    std_o_idx = np.std(fold_scores)
    
    stability = 1 - (std_o_idx / mean_o_idx) if mean_o_idx > 0 else 0
    
    summary = ValidationSummary(
        mean_train_rmse_log=mean_train_rmse,
        mean_val_rmse_log=mean_val_rmse,
        mean_overfit_index=mean_o_idx,
        std_overfit_index=std_o_idx,
        min_overfit_index=np.min(fold_scores),
        max_overfit_index=np.max(fold_scores),
        stability_score=stability,
        results=results
    )
    
    return summary


def bootstrap_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    model_params: Dict[str, Any],
    n_iterations: int = 10,
    sample_size: float = 0.8,
    seed: int = 42,
    threads: int = 10
) -> ValidationSummary:
    """
    Bootstrap validation: random sampling with replacement
    """
    set_thread_env(threads)
    np.random.seed(seed)
    
    results = []
    fold_scores = []
    n_samples = int(len(X) * sample_size)
    
    log(f"Starting Bootstrap Validation ({n_iterations} iterations)...")
    
    for i in range(1, n_iterations + 1):
        log(f"  Iteration {i}/{n_iterations}")
        
        # Bootstrap sample (with replacement)
        sample_idx = np.random.choice(len(X), size=n_samples, replace=True)
        # Out-of-bag sample (not in bootstrap)
        oob_idx = np.setdiff1d(np.arange(len(X)), sample_idx)
        
        if len(oob_idx) == 0:
            continue
        
        X_train = X.iloc[sample_idx]
        X_val = X.iloc[oob_idx]
        y_train = y[sample_idx]
        y_val = y[oob_idx]
        
        # Train model
        model = lgb.LGBMRegressor(**model_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=800,
            verbose=False
        )
        
        best_iter = getattr(model, "best_iteration_", 0) or model_params.get("n_estimators", 15000)
        
        # Predictions
        pred_train = model.predict(X_train, num_iteration=best_iter)
        pred_val = model.predict(X_val, num_iteration=best_iter)
        
        # Calculate metrics
        train_metrics = eval_reg(y_train, pred_train)
        val_metrics = eval_reg(y_val, pred_val)
        
        overfit_idx = val_metrics["rmse_log"] / max(train_metrics["rmse_log"], 1e-12)
        fold_scores.append(overfit_idx)
        
        results.append(ValidationResult(
            fold=i,
            train_rmse_log=train_metrics["rmse_log"],
            val_rmse_log=val_metrics["rmse_log"],
            train_r2_log=train_metrics["r2_log"],
            val_r2_log=val_metrics["r2_log"],
            train_rmse_tl=train_metrics["rmse_tl"],
            val_rmse_tl=val_metrics["rmse_tl"],
            overfit_index=overfit_idx,
            fold_type="bootstrap"
        ))
        
        log(f"    Iteration {i}: Overfit Index = {overfit_idx:.3f} (OOB size: {len(oob_idx)})")
    
    # Calculate summary
    mean_train_rmse = np.mean([r.train_rmse_log for r in results])
    mean_val_rmse = np.mean([r.val_rmse_log for r in results])
    mean_o_idx = np.mean(fold_scores)
    std_o_idx = np.std(fold_scores)
    
    stability = 1 - (std_o_idx / mean_o_idx) if mean_o_idx > 0 else 0
    
    summary = ValidationSummary(
        mean_train_rmse_log=mean_train_rmse,
        mean_val_rmse_log=mean_val_rmse,
        mean_overfit_index=mean_o_idx,
        std_overfit_index=std_o_idx,
        min_overfit_index=np.min(fold_scores),
        max_overfit_index=np.max(fold_scores),
        stability_score=stability,
        results=results
    )
    
    return summary


def feature_permutation_test(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    preprocessor,
    n_repeats: int = 10,
    seed: int = 42
) -> Dict[str, float]:
    """
    Test model sensitivity to feature permutation
    A stable model should not be too sensitive to random feature noise
    """
    np.random.seed(seed)
    
    # Get baseline performance
    pred = model.predict(X)
    baseline_rmse = np.sqrt(mean_squared_error(y, pred))
    
    feature_importance = {}
    
    log("Starting Feature Permutation Test...")
    
    for feature in tqdm(X.columns, desc="Permuting features"):
        original_values = X[feature].copy()
        feature_scores = []
        
        for _ in range(n_repeats):
            # Shuffle the feature
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature].values)
            
            # Predict with permuted feature
            pred_perm = model.predict(X_permuted)
            perm_rmse = np.sqrt(mean_squared_error(y, pred_perm))
            
            # Calculate importance score
            importance = perm_rmse / max(baseline_rmse, 1e-12)
            feature_scores.append(importance)
        
        feature_importance[feature] = np.mean(feature_scores)
    
    # Normalize importance scores
    max_imp = max(feature_importance.values()) if feature_importance else 1.0
    for feat in feature_importance:
        feature_importance[feat] /= max_imp
    
    return feature_importance


def plot_validation_results(all_results: Dict[str, ValidationSummary], output_dir: str):
    """Create visualization plots for validation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Overfit Index Distribution across validation methods
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
    
    for i, (method, summary) in enumerate(all_results.items()):
        # Collect overfit indices from all folds
        oi_values = [r.overfit_index for r in summary.results]
        
        # Plot box plot
        bp = plt.boxplot(oi_values, positions=[i], widths=0.6, patch_artist=True)
        bp['boxes'][0].set_facecolor(colors[i])
        bp['medians'][0].set_color('black')
        
        # Add mean point
        plt.scatter([i], [summary.mean_overfit_index], color='red', s=100, zorder=3, marker='X')
    
    plt.axhline(y=1.05, color='green', linestyle='--', alpha=0.7, label='Excellent threshold (1.05)')
    plt.axhline(y=1.15, color='orange', linestyle='--', alpha=0.7, label='Good threshold (1.15)')
    plt.axhline(y=1.25, color='red', linestyle='--', alpha=0.7, label='Moderate threshold (1.25)')
    
    plt.xticks(range(len(all_results)), all_results.keys(), rotation=45, ha='right')
    plt.ylabel('Overfit Index (Val RMSE / Train RMSE)')
    plt.title('Overfitting Check: Multiple Validation Methods')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overfit_indices_distribution.png'), dpi=150, bbox_inches='tight')
    
    # Plot 2: Train vs Validation RMSE comparison
    plt.figure(figsize=(10, 8))
    
    for method, summary in all_results.items():
        # Plot each fold
        for result in summary.results:
            plt.scatter(result.train_rmse_log, result.val_rmse_log, 
                       alpha=0.6, label=f'{method} (fold {result.fold})' if method not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Plot ideal line (train == val)
    min_val = min([min(r.train_rmse_log for r in summary.results) for summary in all_results.values()])
    max_val = max([max(r.val_rmse_log for r in summary.results) for summary in all_results.values()])
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal (Train=Val)')
    
    plt.xlabel('Train RMSE (log)')
    plt.ylabel('Validation RMSE (log)')
    plt.title('Train vs Validation RMSE Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_vs_val_rmse.png'), dpi=150, bbox_inches='tight')
    
    # Plot 3: Stability scores
    plt.figure(figsize=(10, 6))
    
    methods = list(all_results.keys())
    stability_scores = [summary.stability_score for summary in all_results.values()]
    
    bars = plt.bar(methods, stability_scores, color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    
    plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, label='Excellent stability (0.9)')
    plt.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Good stability (0.8)')
    plt.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Poor stability (0.7)')
    
    # Add value labels on bars
    for bar, score in zip(bars, stability_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.ylabel('Stability Score (1 - std/mean)')
    plt.title('Model Stability Across Validation Methods')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stability_scores.png'), dpi=150, bbox_inches='tight')
    
    plt.close('all')
    log(f"Plots saved to {output_dir}")


def generate_comprehensive_report(
    all_results: Dict[str, ValidationSummary],
    original_overfit_idx: float,
    feature_importance: Dict[str, float],
    output_path: str
):
    """Generate a comprehensive HTML report"""
    
    # Calculate overall statistics
    all_overfit_indices = []
    for summary in all_results.values():
        all_overfit_indices.extend([r.overfit_index for r in summary.results])
    
    mean_overall = np.mean(all_overfit_indices)
    std_overall = np.std(all_overfit_indices)
    
    # Create HTML report
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Overfitting Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1200px; margin: auto; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .metric {{ background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .good {{ border-left: 5px solid #4CAF50; }}
            .warning {{ border-left: 5px solid #FF9800; }}
            .bad {{ border-left: 5px solid #F44336; }}
            .section {{ margin-top: 30px; }}
            .plot {{ margin: 20px 0; text-align: center; }}
            img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Comprehensive Overfitting Analysis Report</h1>
                <p>Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="metric {'good' if original_overfit_idx < 1.05 else 'warning' if original_overfit_idx < 1.15 else 'bad'}">
                <h2>Original Model Performance</h2>
                <p><strong>Overfit Index:</strong> {original_overfit_idx:.3f} - {interpret_overfit_index(original_overfit_idx)}</p>
            </div>
            
            <div class="section">
                <h2>Validation Results Summary</h2>
                <table>
                    <tr>
                        <th>Validation Method</th>
                        <th>Mean Overfit Index</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>Stability Score</th>
                        <th>Interpretation</th>
                    </tr>
    """
    
    for method, summary in all_results.items():
        interpretation = interpret_overfit_index(summary.mean_overfit_index)
        html_content += f"""
                    <tr>
                        <td>{method.replace('_', ' ').title()}</td>
                        <td>{summary.mean_overfit_index:.3f}</td>
                        <td>{summary.std_overfit_index:.3f}</td>
                        <td>{summary.min_overfit_index:.3f}</td>
                        <td>{summary.max_overfit_index:.3f}</td>
                        <td>{summary.stability_score:.3f}</td>
                        <td>{interpretation}</td>
                    </tr>
        """
    
    html_content += f"""
                </table>
            </div>
            
            <div class="metric {'good' if mean_overall < 1.05 else 'warning' if mean_overall < 1.15 else 'bad'}">
                <h2>Overall Assessment</h2>
                <p><strong>Mean Overfit Index (All Methods):</strong> {mean_overall:.3f} ± {std_overall:.3f}</p>
                <p><strong>Interpretation:</strong> {interpret_overfit_index(mean_overall)}</p>
                <p><strong>Conclusion:</strong> {'✅ Model appears stable and not overfitting' if mean_overall < 1.05 else 
                                                 '⚠️ Model shows moderate overfitting - consider additional regularization' if mean_overall < 1.15 else 
                                                 '❌ Model shows significant overfitting - needs architectural changes'}</p>
            </div>
            
            <div class="section">
                <h2>Visualizations</h2>
                <div class="plot">
                    <img src="overfit_indices_distribution.png" alt="Overfit Indices Distribution">
                </div>
                <div class="plot">
                    <img src="train_vs_val_rmse.png" alt="Train vs Validation RMSE">
                </div>
                <div class="plot">
                    <img src="stability_scores.png" alt="Stability Scores">
                </div>
            </div>
    """
    
    # Add top features by importance
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:20]
        html_content += """
            <div class="section">
                <h2>Top 20 Feature Importance (Permutation Test)</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Importance Score</th>
                    </tr>
        """
        
        for feature, importance in top_features:
            html_content += f"""
                    <tr>
                        <td>{feature}</td>
                        <td>{importance:.3f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
    
    html_content += """
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
    """
    
    if mean_overall < 1.05:
        html_content += """
                    <li>✅ Model is well-regularized and shows low overfitting</li>
                    <li>✅ Continue with current architecture for production</li>
                    <li>✅ Consider experimenting with slightly larger models for better accuracy</li>
        """
    elif mean_overall < 1.15:
        html_content += """
                    <li>⚠️ Model shows slight overfitting</li>
                    <li>⚠️ Consider increasing regularization (higher lambda, lower learning rate)</li>
                    <li>⚠️ Add more dropout or reduce model complexity slightly</li>
        """
    else:
        html_content += """
                    <li>❌ Significant overfitting detected</li>
                    <li>❌ Reduce model complexity (fewer leaves, shallower trees)</li>
                    <li>❌ Increase regularization parameters significantly</li>
                    <li>❌ Add more training data or use data augmentation</li>
                    <li>❌ Consider ensemble methods to reduce variance</li>
        """
    
    html_content += """
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    log(f"HTML report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive overfitting checker for model6_balanced')
    parser.add_argument('--bundle', type=str, required=True,
                       help='Path to model bundle (.joblib file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--output_dir', type=str, default='overfitting_reports',
                       help='Directory to save reports and plots')
    parser.add_argument('--threads', type=int, default=10,
                       help='Number of threads for parallel processing')
    parser.add_argument('--kfold_splits', type=int, default=5,
                       help='Number of splits for K-Fold validation')
    parser.add_argument('--bootstrap_iterations', type=int, default=10,
                       help='Number of bootstrap iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set up
    set_thread_env(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)
    
    log("=" * 60)
    log("COMPREHENSIVE OVERFITTING CHECKER v3")
    log("=" * 60)
    
    # Load model bundle
    bundle = load_model_bundle(args.bundle)
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    target_encoder = bundle.get("target_encoder", None)
    original_overfit_idx = bundle["meta"]["overfit_index"]
    
    log(f"Original overfit index from training: {original_overfit_idx:.3f}")
    log(f"Interpretation: {interpret_overfit_index(original_overfit_idx)}")
    
    # Load and preprocess data
    log("\n[1/6] Loading and preprocessing data...")
    X, y_log, y_tl, indices = load_and_preprocess_data(
        args.data, preprocessor, target_encoder, bundle
    )
    log(f"Preprocessed data shape: {X.shape}")
    
    # Extract model parameters
    model_params = model.get_params()
    model_params['n_jobs'] = args.threads
    model_params['verbosity'] = -1
    
    # Store all validation results
    all_results = {}
    
    # 1. K-Fold Cross Validation
    log("\n[2/6] Performing K-Fold Cross Validation...")
    kfold_results = kfold_cross_validation(
        X, y_log, model_params,
        n_splits=args.kfold_splits,
        seed=args.seed,
        threads=args.threads
    )
    all_results['kfold'] = kfold_results
    
    # 2. Stratified Price Validation
    log("\n[3/6] Performing Stratified Price Validation...")
    stratified_results = stratified_price_validation(
        X, y_log, y_tl, model_params,
        n_splits=args.kfold_splits,
        seed=args.seed,
        threads=args.threads
    )
    all_results['stratified_price'] = stratified_results
    
    # 3. Bootstrap Validation
    log("\n[4/6] Performing Bootstrap Validation...")
    bootstrap_results = bootstrap_validation(
        X, y_log, model_params,
        n_iterations=args.bootstrap_iterations,
        seed=args.seed,
        threads=args.threads
    )
    all_results['bootstrap'] = bootstrap_results
    
    # 4. Try Time-based validation if date column exists
    log("\n[5/6] Checking for time-based validation...")
    # Look for date columns in original data
    df_raw = robust_read_csv(args.data)
    df_raw.columns = normalize_colnames(df_raw.columns)
    
    date_columns = [c for c in df_raw.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_columns:
        try:
            # Use the first date column found
            date_col = date_columns[0]
            dates = pd.to_datetime(df_raw[date_col], errors='coerce')
            if dates.notna().sum() > len(dates) * 0.5:  # If at least 50% valid dates
                time_results = time_based_validation(
                    X, y_log, dates, model_params,
                    test_size=0.2,
                    threads=args.threads
                )
                if time_results:
                    all_results['time_based'] = time_results
        except Exception as e:
            log(f"Time-based validation failed: {e}")
    
    # 5. Feature Permutation Test
    log("\n[6/6] Performing Feature Permutation Test...")
    feature_importance = feature_permutation_test(
        model, X, y_log, preprocessor,
        n_repeats=5,
        seed=args.seed
    )
    
    # Generate visualizations
    log("\nGenerating visualizations...")
    plot_validation_results(all_results, args.output_dir)
    
    # Generate comprehensive report
    report_path = os.path.join(args.output_dir, 'overfitting_report.html')
    generate_comprehensive_report(
        all_results, original_overfit_idx, feature_importance, report_path
    )
    
    # Print summary
    log("\n" + "=" * 60)
    log("OVERFITTING CHECK SUMMARY")
    log("=" * 60)
    
    for method, summary in all_results.items():
        log(f"\n{method.upper().replace('_', ' ')}:")
        log(f"  Mean Overfit Index: {summary.mean_overfit_index:.3f}")
        log(f"  Range: [{summary.min_overfit_index:.3f}, {summary.max_overfit_index:.3f}]")
        log(f"  Stability: {summary.stability_score:.3f}")
        log(f"  Interpretation: {interpret_overfit_index(summary.mean_overfit_index)}")
    
    # Overall assessment
    all_indices = []
    for summary in all_results.values():
        all_indices.extend([r.overfit_index for r in summary.results])
    
    mean_overall = np.mean(all_indices)
    log(f"\nOVERALL ASSESSMENT:")
    log(f"  Mean across all methods: {mean_overall:.3f}")
    log(f"  Original vs Multi-Validation: {original_overfit_idx:.3f} vs {mean_overall:.3f}")
    log(f"  Difference: {abs(original_overfit_idx - mean_overall):.3f}")
    
    if mean_overall < 1.05:
        log("✅ RESULT: Model is NOT overfitting! The good results are real!")
    elif mean_overall < 1.15:
        log("⚠️ RESULT: Model shows MODERATE overfitting. Consider minor adjustments.")
    else:
        log("❌ RESULT: Model IS overfitting significantly. Need to address this.")
    
    log(f"\nDetailed report: {report_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()