"""
explain.py — SHAP Explainability, Leakage Prevention, and Business Recommendations

This module:
1. Loads the pre-trained XGBoost model and processed features.
2. Recreates the identical 10-fold Stratified Cross-Validation split to isolate
   the Fold 0 validation set (ensuring zero data leakage from the training set).
3. Selects a background dataset from the training fold for TreeExplainer.
4. Computes SHAP values for the held-out validation set.
5. Generates Global Interpretability Plots:
   - Beeswarm Summary Plot (plots/shap_beeswarm.png)
   - Mean Absolute SHAP Bar Plot (plots/shap_bar.png)
6. Generates Local Interpretability Plots:
   - Waterfall Plot for the highest-risk customer (plots/shap_waterfall_local.png)
7. Evaluates local SHAP values to recommend personalized win-back actions.
"""
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold

def get_winback_action(customer_row, customer_shap_vals, feature_names):
    """
    Analyzes local SHAP values and outputs a list of personalized win-back recommendations.
    Ties specific feature triggers to action items.
    """
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "feature_value": customer_row.values,
        "shap_value": customer_shap_vals
    })
    
    # Sort by SHAP value descending (highest contributors to churn probability)
    shap_df = shap_df.sort_values("shap_value", ascending=False).reset_index(drop=True)
    
    # Select top contributors to churn risk (SHAP > 0)
    top_drivers = shap_df[shap_df["shap_value"] > 0.01].head(3)
    
    recommendations = []
    
    for _, row in top_drivers.iterrows():
        feat = row["feature"]
        val = row["feature_value"]
        sh_val = row["shap_value"]
        
        # 1. Recency Gap
        if "days_since_last" in feat or "recency" in feat:
            recommendations.append(
                f"• High Inactivity (Driver: {feat} = {val:.1f} days, SHAP = +{sh_val:.3f}): "
                f"Offer a 'Welcome Back' transaction cashback voucher (e.g., get 50 TK cashback on the next transaction) to re-engage."
            )
        # 1b. Low Transaction Volume
        elif "count" in feat and ("out_count" in feat or "in_count" in feat or "trx_count" in feat) and val <= 1.0:
            recommendations.append(
                f"• Low Transaction Volume (Driver: {feat} = {val:.1f}, SHAP = +{sh_val:.3f}): "
                f"The customer has low activity in key metrics. Offer a transaction cashback campaign (e.g., get 20 TK cashback on each of your next 3 transactions) to rebuild usage habits."
            )
        # 2. Draining Balance
        elif "balance_trend" in feat and val < 0:
            recommendations.append(
                f"• Declining Wallet Balance (Driver: {feat} = {val:.2f}, SHAP = +{sh_val:.3f}): "
                f"Target with a Balance Booster campaign: Offer 5% interest or cashback if they maintain a daily average balance > 2,000 TK for 30 days."
            )
        elif "balance_drop" in feat and val < 0:
            recommendations.append(
                f"• Significant Balance Drop (Driver: {feat} = {val:.1f} TK, SHAP = +{sh_val:.3f}): "
                f"Offer a high-yield savings campaign or a transaction discount incentive to retain deposit volumes."
            )
        # 3. Empty Wallet / Low Balance
        elif "zero_balance_days" in feat and val > 5:
            recommendations.append(
                f"• Frequent Zero Balance (Driver: {feat} = {val:.1f} days, SHAP = +{sh_val:.3f}): "
                f"Recommend zero-balance fee waivers and send a Micro-credit reload voucher (e.g., zero-interest cash advance up to 500 TK)."
            )
        # 4. Zero Bill Pay
        elif "bill_pay" in feat and ("flag_zero" in feat or val == 0 or "ratio" in feat):
            recommendations.append(
                f"• Under-utilization of Bill Pay (Driver: {feat} = {val}, SHAP = +{sh_val:.3f}): "
                f"Bill payments are a sticky retention anchor. Offer a 10% discount voucher on their next utility bill (electricity/gas/water) when paid via FictiPay."
            )
        # 5. Zero Merchant Pay
        elif "merchant" in feat and ("flag_zero" in feat or val == 0 or "ratio" in feat):
            recommendations.append(
                f"• Low Merchant Payment Activity (Driver: {feat} = {val}, SHAP = +{sh_val:.3f}): "
                f"Promote retail wallet adoption: Offer partner merchant discount vouchers (e.g., grocery/superstore discount) for their next merchant payment."
            )
        # 6. High Activity Decay
        elif "decay" in feat and val > 1.2:
            recommendations.append(
                f"• Sudden Activity Drop (Driver: {feat} = {val:.2f}, SHAP = +{sh_val:.3f}): "
                f"Outbound customer support call: Initiate an outbound care agent check-in to investigate technical issues or service complaints."
            )
        # 7. Low tenure
        elif "tenure" in feat and val < 180:
            recommendations.append(
                f"• New User Friction (Driver: {feat} = {val:.1f} days, SHAP = +{sh_val:.3f}): "
                f"Onboarding flow optimization: Send a guided notification with new-user tutorials and onboarding task rewards."
            )
            
    # Default fall back if no specific rule matched
    if not recommendations:
        top_feat = shap_df.iloc[0]["feature"]
        top_val = shap_df.iloc[0]["feature_value"]
        top_sh = shap_df.iloc[0]["shap_value"]
        recommendations.append(
            f"• General Retention Alert (Driver: {top_feat} = {top_val}, SHAP = +{top_sh:.3f}): "
            f"Provide general transaction fee discount (e.g., 50% off next cash-out fee) to lower friction."
        )
        
    return recommendations

def run_explainability():
    print("Loading pre-trained XGBoost model and processed features...")
    
    # Load model and data
    xgb_model = joblib.load("./models/xgb_model.pkl")
    train_df = pd.read_parquet("./processed_data/train_features.parquet")
    
    exclude_cols = ["ACCOUNT_ID", "CHURN"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Extract features and targets, cleaning names exactly like training
    X = train_df[feature_cols].copy()
    X.columns = [str(col).replace("[", "").replace("]", "").replace("<", "").replace(" ", "_") for col in X.columns]
    y = train_df["CHURN"].copy()
    
    print(f"Dataset Shape: {X.shape}")
    
    # Recreate the Stratified 10-Fold split used during model training to isolate the Fold 0 validation set
    print("Recreating Stratified K-Fold splits (n_splits=10, seed=42)...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        if fold == 0:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            break
            
    print(f"Isolated Fold 0 Training Samples: {X_train.shape[0]}")
    print(f"Isolated Fold 0 Validation (Held-out) Samples: {X_val.shape[0]}")
    
    # Ensure zero data leakage: select background dataset solely from the training partition
    print("Sampling 200 representative training instances as background data...")
    background_data = shap.sample(X_train, 200, random_state=42)
    
    # Sub-sample the held-out validation set for performance stability during SHAP calculation
    sample_size = min(2000, len(X_val))
    print(f"Sampling {sample_size} validation samples for explanation...")
    np.random.seed(42)
    val_sample_idx = np.random.choice(X_val.index, sample_size, replace=False)
    X_val_sample = X_val.loc[val_sample_idx]
    y_val_sample = y_val.loc[val_sample_idx]
    
    # Initialize TreeExplainer
    # TreeExplainer is highly optimized for gradient boosted trees. Passing the background
    # dataset ensures SHAP values map output differences relative to the reference cohort.
    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(xgb_model, data=background_data)
    
    print("Computing SHAP values on validation sample...")
    raw_shap_values = explainer.shap_values(X_val_sample)
    
    # Handle list outputs (e.g. multi-class format sometimes returned for binary)
    if isinstance(raw_shap_values, list):
        shap_vals = raw_shap_values[1]
    else:
        shap_vals = raw_shap_values
        
    os.makedirs("./plots", exist_ok=True)
    
    # Format the SHAP values into an Explanation object (preferred by modern SHAP plotting API)
    expected_val = explainer.expected_value
    if isinstance(expected_val, (list, np.ndarray)):
        expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]
        
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=expected_val,
        data=X_val_sample.values,
        feature_names=X_val_sample.columns.tolist()
    )
    
    # ----------------------------------------------------
    # Plot 1: SHAP Beeswarm Plot (Global Attributions)
    # ----------------------------------------------------
    print("Generating SHAP Beeswarm summary plot...")
    plt.figure(figsize=(12, 10))
    # Override show=False to save to disk
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.title("SHAP Beeswarm Summary Plot (Fold 0 Validation Set)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("./plots/shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: ./plots/shap_beeswarm.png")
    
    # ----------------------------------------------------
    # Plot 2: Mean Absolute SHAP Bar Plot (Global Importance)
    # ----------------------------------------------------
    print("Generating Mean Absolute SHAP Bar Plot...")
    plt.figure(figsize=(12, 10))
    shap.plots.bar(explanation, max_display=20, show=False)
    plt.title("Mean Absolute SHAP Feature Importance (Fold 0 Validation Set)", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig("./plots/shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: ./plots/shap_bar.png")
    
    # ----------------------------------------------------
    # Local Explainability (Specific Customer Analysis)
    # ----------------------------------------------------
    print("Identifying the highest-risk customer in the validation sample...")
    preds_proba = xgb_model.predict_proba(X_val_sample)[:, 1]
    highest_risk_idx = np.argmax(preds_proba)
    
    high_risk_customer_row = X_val_sample.iloc[highest_risk_idx]
    high_risk_customer_shap = shap_vals[highest_risk_idx]
    high_risk_customer_prob = preds_proba[highest_risk_idx]
    high_risk_customer_id = val_sample_idx[highest_risk_idx]
    
    # Fetch account ID
    account_id = train_df.loc[high_risk_customer_id, "ACCOUNT_ID"]
    
    print(f"\n==================================================")
    print(f"LOCAL ATTRIBUTION AUDIT: HIGHEST-RISK CUSTOMER")
    print(f"==================================================")
    print(f"Account ID: {account_id}")
    print(f"Predicted Churn Probability: {high_risk_customer_prob:.4f}")
    
    # Create Local Explanation object for waterfall
    local_explanation = shap.Explanation(
        values=high_risk_customer_shap,
        base_values=expected_val,
        data=high_risk_customer_row.values,
        feature_names=X_val_sample.columns.tolist()
    )
    
    # Generate Waterfall Plot
    print("\nGenerating local waterfall plot...")
    plt.figure(figsize=(10, 8))
    shap.plots.waterfall(local_explanation, max_display=12, show=False)
    plt.title(f"SHAP Waterfall Local Attribution: Account {account_id}", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig("./plots/shap_waterfall_local.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved: ./plots/shap_waterfall_local.png")
    
    # Integrate Business Retention Rules
    print("\nEvaluating top risk factors to trigger win-back recommendations...")
    recs = get_winback_action(high_risk_customer_row, high_risk_customer_shap, X_val_sample.columns.tolist())
    print("\n--- RECOMMENDED WIN-BACK ACTIONS ---")
    for r in recs:
        print(r)
    print("==================================================")

if __name__ == "__main__":
    run_explainability()
