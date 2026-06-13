"""
segment.py — Autoencoder + K-Means Customer Segmentation

This module:
1. Trains a symmetric PyTorch Autoencoder to map behavioral features
   into a compressed latent space
2. Applies K-Means clustering on the latent embeddings to segment
   customers into distinct behavioral groups
3. Generates segment profiles and targeted retention recommendations
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class Autoencoder(nn.Module):
    """Symmetric Autoencoder: input -> 32 -> 16 -> 32 -> input."""
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
    def encode(self, x):
        return self.encoder(x)

def train_autoencoder(X_scaled, epochs=20, batch_size=512, latent_dim=16):
    """Train the autoencoder and return latent embeddings."""
    input_dim = X_scaled.shape[1]
    model = Autoencoder(input_dim, latent_dim)
    
    dataset = TensorDataset(torch.FloatTensor(X_scaled))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs} — Reconstruction Loss: {avg_loss:.6f}")
    
    # Extract latent embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(torch.FloatTensor(X_scaled)).numpy()
    
    return embeddings, model

def run_segmentation():
    """Full segmentation pipeline."""
    print("Loading train features for segmentation...")
    train_df = pd.read_parquet("./processed_data/train_features.parquet")
    
    # Select behavioral features for clustering (exclude IDs, labels, one-hot categories)
    behavioral_cols = [
        "tenure_days",
        "out_count_total", "out_sum_total", "in_count_total", "in_sum_total",
        "trx_count_decay_feb_march", "trx_type_diversity",
        "merchant_pay_ratio", "bill_pay_ratio",
        "mean_bal_march", "std_bal_march", "final_balance_march",
        "balance_trend_march", "zero_balance_days_march", "balance_stability_march"
    ]
    
    X = train_df[behavioral_cols].fillna(0).values
    y = train_df["CHURN"].values
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train autoencoder
    print("Training Autoencoder (20 epochs)...")
    embeddings, ae_model = train_autoencoder(X_scaled, epochs=20, latent_dim=16)
    print(f"Latent embeddings shape: {embeddings.shape}")
    
    # K-Means on latent space — try K=3,4,5 and select by silhouette
    print("\nFinding optimal K via Silhouette Score...")
    best_k, best_score = 3, -1
    for k in [3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels, sample_size=5000, random_state=42)
        print(f"  K={k}: Silhouette = {sil:.4f}")
        if sil > best_score:
            best_k = k
            best_score = sil
    
    print(f"\nSelected K={best_k} (Silhouette = {best_score:.4f})")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(embeddings)
    
    # Add cluster labels to DataFrame
    train_df["segment"] = cluster_labels
    
    # Segment Profiles
    print("\n=== CUSTOMER SEGMENT PROFILES ===")
    profile_cols = behavioral_cols + ["CHURN"]
    profile = train_df.groupby("segment")[profile_cols].agg(["mean", "count"]).round(3)
    
    # Simplified profile table
    summary_rows = []
    for seg in range(best_k):
        seg_data = train_df[train_df["segment"] == seg]
        summary_rows.append({
            "Segment": seg,
            "Count": len(seg_data),
            "Churn Rate (%)": round(seg_data["CHURN"].mean() * 100, 2),
            "Avg Tenure (days)": round(seg_data["tenure_days"].mean(), 1),
            "Avg Trx Count": round(seg_data["out_count_total"].mean(), 1),
            "Avg Trx Diversity": round(seg_data["trx_type_diversity"].mean(), 2),
            "Avg March Balance": round(seg_data["mean_bal_march"].mean(), 2),
            "Avg Balance Trend": round(seg_data["balance_trend_march"].mean(), 4),
            "Avg Zero-Balance Days": round(seg_data["zero_balance_days_march"].mean(), 1)
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv("./plots/segment_profiles.csv", index=False)
    
    # Plot: Segment Churn Rates
    os.makedirs("./plots", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Churn rate by segment
    churn_by_seg = train_df.groupby("segment")["CHURN"].mean() * 100
    axes[0].bar(churn_by_seg.index, churn_by_seg.values, color=["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"][:best_k])
    axes[0].set_xlabel("Segment")
    axes[0].set_ylabel("Churn Rate (%)")
    axes[0].set_title("Churn Rate by Customer Segment")
    
    # 2. Segment sizes
    seg_sizes = train_df["segment"].value_counts().sort_index()
    axes[1].bar(seg_sizes.index, seg_sizes.values, color=["#2ecc71", "#e74c3c", "#f39c12", "#3498db", "#9b59b6"][:best_k])
    axes[1].set_xlabel("Segment")
    axes[1].set_ylabel("Customer Count")
    axes[1].set_title("Segment Sizes")
    
    # 3. Balance vs Transaction Count colored by segment
    sample_idx = np.random.choice(len(train_df), min(5000, len(train_df)), replace=False)
    sample = train_df.iloc[sample_idx]
    scatter = axes[2].scatter(
        sample["out_count_total"], sample["mean_bal_march"],
        c=sample["segment"], cmap="Set2", alpha=0.5, s=10
    )
    axes[2].set_xlabel("Total Transaction Count")
    axes[2].set_ylabel("Mean March Balance")
    axes[2].set_title("Customer Segments in Feature Space")
    plt.colorbar(scatter, ax=axes[2], label="Segment")
    
    plt.suptitle("Customer Segmentation Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("./plots/segmentation.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("\nSegmentation plot saved to plots/segmentation.png")
    
    # Business Recommendations
    print("\n=== TARGETED RETENTION RECOMMENDATIONS ===")
    for _, row in summary_df.iterrows():
        seg = int(row["Segment"])
        churn = row["Churn Rate (%)"]
        trx = row["Avg Trx Count"]
        bal = row["Avg March Balance"]
        
        if churn > 20 and trx > 10:
            label = "High-Value At-Risk"
            action = "Personal loyalty cashback + high-touch relationship manager outreach"
        elif churn > 20 and trx <= 10:
            label = "Low-Engagement Dormant"
            action = "Re-engagement campaign with free CashIn vouchers + gamified rewards"
        elif churn <= 20 and trx > 10:
            label = "Active Power User"
            action = "Cross-sell premium products (merchant subscriptions, micro-loans)"
        else:
            label = "Stable Low-Activity"
            action = "Introduce P2P/MerchantPay migration rewards to increase stickiness"
        
        print(f"\n  Segment {seg} ({label}):")
        print(f"    Churn Rate: {churn}% | Avg Transactions: {trx} | Avg Balance: {bal:.0f}")
        print(f"    → Recommendation: {action}")

if __name__ == "__main__":
    run_segmentation()
