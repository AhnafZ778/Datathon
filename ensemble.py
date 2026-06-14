"""
ensemble.py — Stacking Meta-Learner, Rank-Average Blending, and Probability Calibration

This module loads the out-of-fold (OOF) predictions from each base model,
compares Stacking meta-models and Rank-Average Blending, calibrates the final output
via Isotonic Regression, and generates the submission CSV.

Includes Optuna-based weight optimization and power-rank blending.
"""
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from scipy.stats import rankdata
from scipy.optimize import minimize

# Try importing Optuna
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

def load_predictions():
    """Load OOF and test predictions from the model zoo."""
    oof = pd.read_parquet("./predictions/oof_predictions.parquet")
    test = pd.read_parquet("./predictions/test_predictions.parquet")
    return oof, test


def optimize_weights_optuna(oof_ranks, y, n_models, n_trials=200):
    """Use Optuna to find optimal blend weights maximizing OOF AUC."""
    print(f"  Running Optuna weight optimization ({n_trials} trials)...")
    
    def objective(trial):
        weights = np.array([trial.suggest_float(f"w{i}", 0.01, 1.0) for i in range(n_models)])
        weights = weights / weights.sum()
        blend = np.zeros(len(y))
        for i in range(n_models):
            blend += weights[i] * oof_ranks[i]
        return roc_auc_score(y, blend)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    best_weights = np.array([study.best_params[f"w{i}"] for i in range(n_models)])
    best_weights = best_weights / best_weights.sum()
    print(f"  Optuna best AUC: {study.best_value:.5f}")
    return best_weights


def optimize_weights_hillclimb(oof_ranks, y, n_models, n_iter=5000):
    """Hill climbing weight optimization as fallback."""
    print(f"  Running hill-climbing weight optimization ({n_iter} iterations)...")
    
    # Start from equal weights
    weights = np.ones(n_models) / n_models
    best_auc = 0
    best_weights = weights.copy()
    
    rng = np.random.RandomState(42)
    
    for i in range(n_iter):
        # Perturb weights
        new_weights = weights + rng.normal(0, 0.02, n_models)
        new_weights = np.clip(new_weights, 0.001, None)
        new_weights = new_weights / new_weights.sum()
        
        blend = np.zeros(len(y))
        for j in range(n_models):
            blend += new_weights[j] * oof_ranks[j]
        
        auc = roc_auc_score(y, blend)
        if auc > best_auc:
            best_auc = auc
            best_weights = new_weights.copy()
            weights = new_weights.copy()
    
    print(f"  Hill-climb best AUC: {best_auc:.5f}")
    return best_weights


def optimize_power_rank(oof_preds_list, y, stack_cols):
    """Find optimal power parameter for rank transformation (rank^power)."""
    print("  Optimizing rank power parameter...")
    
    best_auc = 0
    best_power = 1.0
    
    for power in np.arange(0.3, 3.0, 0.05):
        rank_sum = np.zeros(len(y))
        for preds in oof_preds_list:
            r = (rankdata(preds) / len(preds)) ** power
            rank_sum += r
        auc = roc_auc_score(y, rank_sum)
        if auc > best_auc:
            best_auc = auc
            best_power = power
    
    print(f"  Best rank power: {best_power:.2f} (AUC: {best_auc:.5f})")
    return best_power


def build_stacking_ensemble():
    """Train a Logistic Regression/Ridge meta-classifier on stacked OOF predictions and compare with Rank-Average Blending."""
    oof, test = load_predictions()
    
    # Identify available base models dynamically
    stack_cols = [c for c in oof.columns if c.endswith("_oof")]
    test_stack_cols = [c.replace("_oof", "_test") for c in stack_cols]
    
    print("Detected base models for ensembling:")
    for col in stack_cols:
        print(f"  - {col[:-4]}")
        
    X_stack = oof[stack_cols].values
    y = oof["CHURN"].values
    X_test_stack = test[test_stack_cols].values
    
    n_models = len(stack_cols)
    
    # Results dictionary
    results = {}
    
    # 1. Individual Model Metrics
    print("\n--- Individual Base Model AUCs ---")
    individual_aucs = {}
    for col in stack_cols:
        auc = roc_auc_score(y, oof[col].values)
        brier = brier_score_loss(y, oof[col].values)
        individual_aucs[col] = auc
        print(f"  {col[:-4]:<20}: AUC = {auc:.5f}, Brier = {brier:.5f}")
        results[f"Base: {col[:-4]}"] = {
            "auc": auc,
            "brier": brier,
            "oof": oof[col].values,
            "test": test[col.replace('_oof', '_test')].values
        }

    # 2. Stacking Meta-Classifier (Logistic Regression)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_oof = np.zeros(len(X_stack))
    meta_test = np.zeros(len(X_test_stack))
    
    print("\nTraining Stacking Meta-Classifier (Logistic Regression)...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X_stack, y)):
        X_train, y_train = X_stack[train_idx], y[train_idx]
        X_val = X_stack[val_idx]
        
        meta_model = LogisticRegression(max_iter=1000, random_state=42 + fold)
        meta_model.fit(X_train, y_train)
        
        meta_oof[val_idx] = meta_model.predict_proba(X_val)[:, 1]
        meta_test += meta_model.predict_proba(X_test_stack)[:, 1] / cv.n_splits
    
    # Calibrate Stacking
    iso_meta = IsotonicRegression(out_of_bounds="clip")
    iso_meta.fit(meta_oof, y)
    calibrated_meta_oof = iso_meta.predict(meta_oof)
    calibrated_meta_test = iso_meta.predict(meta_test)
    
    results["Stacking (Calibrated LR)"] = {
        "auc": roc_auc_score(y, calibrated_meta_oof),
        "brier": brier_score_loss(y, calibrated_meta_oof),
        "oof": calibrated_meta_oof,
        "test": calibrated_meta_test
    }
    
    # 3. Simple Weighted Average Ensemble
    total_auc = sum(individual_aucs.values())
    weights = {col: auc / total_auc for col, auc in individual_aucs.items()}
    
    weighted_oof = np.zeros(len(X_stack))
    weighted_test = np.zeros(len(X_test_stack))
    for oof_col, test_col in zip(stack_cols, test_stack_cols):
        w = weights[oof_col]
        weighted_oof += w * oof[oof_col].values
        weighted_test += w * test[test_col].values
        
    results["Weighted Average"] = {
        "auc": roc_auc_score(y, weighted_oof),
        "brier": brier_score_loss(y, weighted_oof),
        "oof": weighted_oof,
        "test": weighted_test
    }

    # 4. Rank-Average Blending with AUC-squared weights (baseline)
    print("\nComputing Rank-Average Blend (AUC-squared weights)...")
    rank_oof_baseline = np.zeros(len(X_stack))
    rank_test_baseline = np.zeros(len(X_test_stack))
    
    for oof_col, test_col in zip(stack_cols, test_stack_cols):
        w = (individual_aucs[oof_col] ** 2) / sum(auc ** 2 for auc in individual_aucs.values())
        r_oof = rankdata(oof[oof_col].values) / len(oof)
        r_test = rankdata(test[test_col].values) / len(test)
        rank_oof_baseline += w * r_oof
        rank_test_baseline += w * r_test
        
    iso_rank_base = IsotonicRegression(out_of_bounds="clip")
    iso_rank_base.fit(rank_oof_baseline, y)
    
    results["Rank-Avg (AUC² weights)"] = {
        "auc": roc_auc_score(y, iso_rank_base.predict(rank_oof_baseline)),
        "brier": brier_score_loss(y, iso_rank_base.predict(rank_oof_baseline)),
        "oof": iso_rank_base.predict(rank_oof_baseline),
        "test": iso_rank_base.predict(rank_test_baseline)
    }

    # 5. === NEW: Optuna/Hill-Climbing Optimized Rank-Average Blend ===
    print("\n--- Optimized Rank-Average Blend ---")
    
    # Pre-compute OOF and test ranks
    oof_ranks = []
    test_ranks = []
    for oof_col, test_col in zip(stack_cols, test_stack_cols):
        oof_ranks.append(rankdata(oof[oof_col].values) / len(oof))
        test_ranks.append(rankdata(test[test_col].values) / len(test))
    
    # Find optimal power for rank transformation
    oof_preds_list = [oof[col].values for col in stack_cols]
    best_power = optimize_power_rank(oof_preds_list, y, stack_cols)
    
    # Apply power transform to ranks
    oof_ranks_power = [(rankdata(oof[col].values) / len(oof)) ** best_power for col in stack_cols]
    test_ranks_power = [(rankdata(test[col].values) / len(test)) ** best_power for col in test_stack_cols]
    
    # Optimize weights
    if OPTUNA_AVAILABLE:
        opt_weights = optimize_weights_optuna(oof_ranks_power, y, n_models, n_trials=300)
    else:
        opt_weights = optimize_weights_hillclimb(oof_ranks_power, y, n_models, n_iter=8000)
    
    print("  Optimized weights:")
    for col, w in zip(stack_cols, opt_weights):
        print(f"    {col[:-4]:<15}: {w:.4f}")
    
    # Apply optimized blend
    opt_rank_oof = np.zeros(len(X_stack))
    opt_rank_test = np.zeros(len(X_test_stack))
    for i, (oof_col, test_col) in enumerate(zip(stack_cols, test_stack_cols)):
        opt_rank_oof += opt_weights[i] * oof_ranks_power[i]
        opt_rank_test += opt_weights[i] * test_ranks_power[i]
    
    # Calibrate
    iso_opt = IsotonicRegression(out_of_bounds="clip")
    iso_opt.fit(opt_rank_oof, y)
    calibrated_opt_oof = iso_opt.predict(opt_rank_oof)
    calibrated_opt_test = iso_opt.predict(opt_rank_test)
    
    results["Optimized Power-Rank Blend"] = {
        "auc": roc_auc_score(y, calibrated_opt_oof),
        "brier": brier_score_loss(y, calibrated_opt_oof),
        "oof": calibrated_opt_oof,
        "test": calibrated_opt_test
    }
    
    # --- Cost-Sensitive Threshold Optimization on final best OOF predictions ---
    best_method = max(results, key=lambda k: results[k]["auc"])
    best_oof_proba = results[best_method]["oof"]
    
    print(f"\nBest ensemble method: {best_method} (AUC: {results[best_method]['auc']:.5f})")
    
    print("\n--- Cost-Sensitive Threshold Optimization ---")
    best_threshold = 0.5
    best_cost = float("inf")
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (best_oof_proba >= threshold).astype(int)
        fn = ((y == 1) & (y_pred == 0)).sum()
        fp = ((y == 0) & (y_pred == 1)).sum()
        cost = 5 * fn + 1 * fp
        if cost < best_cost:
            best_cost = cost
            best_threshold = threshold
            
    print(f"Optimal threshold: {best_threshold:.2f} (Expected Loss: {best_cost})")
    
    # --- Generate Submission ---
    final_test_proba = results[best_method]["test"]
    
    submission = pd.DataFrame({
        "ACCOUNT_ID": test["ACCOUNT_ID"],
        "CHURN_PROB": final_test_proba
    })
    submission.to_csv("./predictions.csv", index=False)
    print(f"\nSubmission saved to predictions.csv with CHURN_PROB probabilities.")
    
    # Also save probabilities for SHAP and segment analysis
    proba_df = pd.DataFrame({
        "ACCOUNT_ID": test["ACCOUNT_ID"],
        "churn_probability": final_test_proba
    })
    os.makedirs("./predictions", exist_ok=True)
    proba_df.to_csv("./predictions/test_probabilities.csv", index=False)
    
    # Print comparison summary
    print("\n=== ENSEMBLE COMPARISON SUMMARY ===")
    print(f"{'Method':<35} {'AUC':>8} {'Brier':>8}")
    print("-" * 55)
    for method, vals in results.items():
        marker = " <<<" if method == best_method else ""
        print(f"{method:<35} {vals['auc']:>8.5f} {vals['brier']:>8.5f}{marker}")

if __name__ == "__main__":
    build_stacking_ensemble()
