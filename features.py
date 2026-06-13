import os
import gc
import glob
import pandas as pd
import numpy as np

base_path = "/home/ahnaf-zakaria/Desktop/Datathon/bkash-presents-nsucec-datathon/public"

def load_kyc(target_ids):
    """Load and preprocess KYC metadata for target customers."""
    print("Loading and preprocessing KYC...")
    kyc = pd.read_parquet(os.path.join(base_path, "kyc.parquet"))
    kyc = kyc[kyc["ACCOUNT_ID"].isin(target_ids)].copy()
    
    ref_date = pd.to_datetime("2024-03-31")
    kyc["tenure_days"] = (ref_date - pd.to_datetime(kyc["ACCOUNT_OPEN_DATE"])).dt.days
    kyc["GENDER"] = kyc["GENDER"].fillna("Unknown")
    kyc["REGION"] = kyc["REGION"].fillna("Unknown")
    kyc = pd.get_dummies(kyc, columns=["GENDER", "REGION"], prefix=["gender", "region"], dtype=float)
    kyc = kyc.drop(columns=["ACCOUNT_TYPE", "ACCOUNT_OPEN_DATE"])
    kyc = kyc.set_index("ACCOUNT_ID")
    return kyc


def compute_recency(target_ids):
    """Separate lightweight pass to compute days_since_last_trx.
    Only loads SRC_ACCOUNT, DST_ACCOUNT, TRX_DATETIME — no TRX_AMT."""
    print("Computing recency (lightweight datetime pass)...")
    trx_files = sorted(glob.glob(os.path.join(base_path, "transactions", "*.parquet")))
    
    last_trx = {}  # ACCOUNT_ID -> max datetime
    
    for filepath in trx_files:
        fname = os.path.basename(filepath)
        print(f"  Scanning {fname} for recency...")
        
        # Only load the two ID columns and datetime — skip TRX_AMT and TRX_TYPE
        df = pd.read_parquet(filepath, columns=["SRC_ACCOUNT", "DST_ACCOUNT", "TRX_DATETIME"])
        df["TRX_DATETIME"] = pd.to_datetime(df["TRX_DATETIME"])
        
        # Source side
        mask_src = df["SRC_ACCOUNT"].isin(target_ids)
        src_last = df.loc[mask_src].groupby("SRC_ACCOUNT")["TRX_DATETIME"].max()
        for acct, dt in src_last.items():
            if acct not in last_trx or dt > last_trx[acct]:
                last_trx[acct] = dt
        del src_last, mask_src
        
        # Destination side
        mask_dst = df["DST_ACCOUNT"].isin(target_ids)
        dst_last = df.loc[mask_dst].groupby("DST_ACCOUNT")["TRX_DATETIME"].max()
        for acct, dt in dst_last.items():
            if acct not in last_trx or dt > last_trx[acct]:
                last_trx[acct] = dt
        del dst_last, mask_dst, df
        gc.collect()
    
    ref_date = pd.to_datetime("2024-03-31")
    recency_series = pd.Series(last_trx)
    recency_series.index.name = "ACCOUNT_ID"
    recency_df = pd.DataFrame({"days_since_last_trx": (ref_date - recency_series).dt.days})
    
    print(f"  Recency computed for {len(recency_df):,} customers.")
    del last_trx, recency_series
    gc.collect()
    return recency_df


def compute_advanced_march_features(target_ids):
    """Load March transactions and extract highly predictive recency, velocity, and micro-window features.
    This runs fast because it is limited to a single month (March)."""
    print("Computing advanced March transaction features...")
    filepath = os.path.join(base_path, "transactions", "trx_2024-03.parquet")
    if not os.path.exists(filepath):
        print("  March transaction file not found! Skipping advanced March features.")
        return pd.DataFrame(index=list(target_ids))
        
    df = pd.read_parquet(filepath, columns=["SRC_ACCOUNT", "DST_ACCOUNT", "TRX_DATETIME", "TRX_TYPE", "TRX_AMT"])
    df["TRX_DATETIME"] = pd.to_datetime(df["TRX_DATETIME"])
    
    # Filter for target customer IDs
    df_src = df[df["SRC_ACCOUNT"].isin(target_ids)].copy()
    df_dst = df[df["DST_ACCOUNT"].isin(target_ids)].copy()
    
    ref_date = pd.to_datetime("2024-03-31 23:59:59")
    march_adv = pd.DataFrame(index=list(target_ids))
    march_adv.index.name = "ACCOUNT_ID"
    
    # 1. Outbound and Inbound Recencies
    print("  Calculating directional recencies...")
    out_last = df_src.groupby("SRC_ACCOUNT")["TRX_DATETIME"].max()
    in_last = df_dst.groupby("DST_ACCOUNT")["TRX_DATETIME"].max()
    
    march_adv["days_since_last_outbound"] = (ref_date - out_last).dt.days
    march_adv["days_since_last_inbound"] = (ref_date - in_last).dt.days
    
    # 2. Recency by transaction type (outbound and inbound)
    print("  Calculating recency by transaction types...")
    trx_types = ["P2P", "MerchantPay", "BillPay", "CashIn", "CashOut"]
    for ttype in trx_types:
        # Outbound types: P2P, MerchantPay, BillPay, CashOut (initiated by customer)
        if ttype in ["P2P", "MerchantPay", "BillPay", "CashOut"]:
            last_dt = df_src[df_src["TRX_TYPE"] == ttype].groupby("SRC_ACCOUNT")["TRX_DATETIME"].max()
            march_adv[f"days_since_last_{ttype}"] = (ref_date - last_dt).dt.days
        # Inbound types: CashIn, P2P (received by customer)
        if ttype in ["CashIn", "P2P"]:
            last_dt = df_dst[df_dst["TRX_TYPE"] == ttype].groupby("DST_ACCOUNT")["TRX_DATETIME"].max()
            prefix = "received" if ttype == "P2P" else "last"
            march_adv[f"days_since_{prefix}_{ttype}"] = (ref_date - last_dt).dt.days
            
    # 3. Micro-window counts and sums (Last 7 days: March 25-31, Last 14 days: March 18-31)
    print("  Calculating micro-window statistics (7d and 14d)...")
    date_7d = pd.to_datetime("2024-03-25 00:00:00")
    date_14d = pd.to_datetime("2024-03-18 00:00:00")
    
    # Outbound 7d/14d
    df_src_7d = df_src[df_src["TRX_DATETIME"] >= date_7d]
    df_src_14d = df_src[df_src["TRX_DATETIME"] >= date_14d]
    
    src_7d_agg = df_src_7d.groupby("SRC_ACCOUNT")["TRX_AMT"].agg(
        out_count_last_7d="count",
        out_sum_last_7d="sum"
    )
    src_14d_agg = df_src_14d.groupby("SRC_ACCOUNT")["TRX_AMT"].agg(
        out_count_last_14d="count",
        out_sum_last_14d="sum"
    )
    
    march_adv = march_adv.join(src_7d_agg, how="left").join(src_14d_agg, how="left")
    
    # Inbound 7d/14d
    df_dst_7d = df_dst[df_dst["TRX_DATETIME"] >= date_7d]
    df_dst_14d = df_dst[df_dst["TRX_DATETIME"] >= date_14d]
    
    dst_7d_agg = df_dst_7d.groupby("DST_ACCOUNT")["TRX_AMT"].agg(
        in_count_last_7d="count",
        in_sum_last_7d="sum"
    )
    dst_14d_agg = df_dst_14d.groupby("DST_ACCOUNT")["TRX_AMT"].agg(
        in_count_last_14d="count",
        in_sum_last_14d="sum"
    )
    
    march_adv = march_adv.join(dst_7d_agg, how="left").join(dst_14d_agg, how="left").fillna(0)
    
    # 4. Outbound transaction velocity in March
    # We will load the total March count and sum to calculate velocity ratios
    march_total_src = df_src.groupby("SRC_ACCOUNT")["TRX_AMT"].agg(
        out_count_march="count",
        out_sum_march="sum"
    )
    
    march_adv = march_adv.join(march_total_src, how="left").fillna(0)
    
    march_adv["out_count_velocity_7d"] = march_adv["out_count_last_7d"] / (march_adv["out_count_march"] + 1e-5)
    march_adv["out_count_velocity_14d"] = march_adv["out_count_last_14d"] / (march_adv["out_count_march"] + 1e-5)
    march_adv["out_sum_velocity_7d"] = march_adv["out_sum_last_7d"] / (march_adv["out_sum_march"] + 1e-5)
    march_adv["out_sum_velocity_14d"] = march_adv["out_sum_last_14d"] / (march_adv["out_sum_march"] + 1e-5)
    
    # Drop columns that are duplicated in the main aggregation
    march_adv = march_adv.drop(columns=["out_count_march", "out_sum_march"])
    
    # Fill missing recency values with 31 days (max for March)
    recency_cols = [c for c in march_adv.columns if c.startswith("days_since_") or c.startswith("days_since_last_")]
    for col in recency_cols:
        march_adv[col] = march_adv[col].fillna(31)
        
    print(f"  Advanced March features computed: {march_adv.shape[1]} features.")
    del df, df_src, df_dst, out_last, in_last, df_src_7d, df_src_14d, src_7d_agg, src_14d_agg
    del df_dst_7d, df_dst_14d, dst_7d_agg, dst_14d_agg, march_total_src
    gc.collect()
    
    return march_adv


def aggregate_transactions_one_month(filepath, month, target_ids):
    """Process a SINGLE month of transactions — minimal memory footprint.
    Does NOT load TRX_DATETIME (handled separately in compute_recency)."""
    print(f"  Aggregating {month} transactions ({os.path.basename(filepath)})...")
    
    # Only load what we need for aggregation — NO datetime column
    df = pd.read_parquet(filepath, columns=["SRC_ACCOUNT", "DST_ACCOUNT", "TRX_TYPE", "TRX_AMT"])
    
    # ---- Outbound (SRC) ----
    df_src = df[df["SRC_ACCOUNT"].isin(target_ids)]
    
    out_grp = df_src.groupby("SRC_ACCOUNT")
    out_agg = out_grp["TRX_AMT"].agg(
        out_count="count",
        out_sum="sum",
        out_avg="mean",
        out_std="std",
        out_max="max"
    )
    out_agg.index.name = "ACCOUNT_ID"
    out_agg["out_std"] = out_agg["out_std"].fillna(0)
    
    type_counts = df_src.groupby(["SRC_ACCOUNT", "TRX_TYPE"]).size().unstack(fill_value=0)
    type_counts.index.name = "ACCOUNT_ID"
    type_counts.columns = [f"count_{col}_{month}" for col in type_counts.columns]
    
    del df_src
    
    # ---- Inbound (DST) ----
    df_dst = df[df["DST_ACCOUNT"].isin(target_ids)]
    
    in_agg = df_dst.groupby("DST_ACCOUNT")["TRX_AMT"].agg(
        in_count="count",
        in_sum="sum"
    )
    in_agg.index.name = "ACCOUNT_ID"
    
    del df_dst, df
    gc.collect()
    
    # ---- Merge into single month DataFrame ----
    month_agg = pd.DataFrame(index=list(target_ids))
    month_agg.index.name = "ACCOUNT_ID"
    month_agg = month_agg.join(out_agg, how="left").fillna(0)
    month_agg = month_agg.join(type_counts, how="left").fillna(0)
    month_agg = month_agg.join(in_agg, how="left").fillna(0)
    
    del out_agg, type_counts, in_agg
    gc.collect()
    
    # Rename columns with month suffix (skip type_counts already suffixed)
    new_cols = []
    for col in month_agg.columns:
        if col.startswith("count_") and col.endswith(f"_{month}"):
            new_cols.append(col)
        else:
            new_cols.append(f"{col}_{month}")
    month_agg.columns = new_cols
    
    return month_agg


def aggregate_transactions(target_ids):
    """Aggregate transactions across all available months, one at a time."""
    print("Aggregating transactions (memory-safe, one month at a time)...")
    trx_files = sorted(glob.glob(os.path.join(base_path, "transactions", "*.parquet")))
    months = ["jan", "feb", "march"]
    
    trx_agg = None
    
    for month_idx, filepath in enumerate(trx_files):
        if month_idx >= len(months):
            break
        month = months[month_idx]
        
        month_df = aggregate_transactions_one_month(filepath, month, target_ids)
        
        if trx_agg is None:
            trx_agg = month_df
        else:
            trx_agg = trx_agg.join(month_df, how="outer").fillna(0)
            del month_df
            gc.collect()
    
    # ---- Cross-Month Features ----
    available_months = months[:min(len(trx_files), len(months))]
    
    # Totals
    trx_agg["out_count_total"] = sum(trx_agg.get(f"out_count_{m}", 0) for m in available_months)
    trx_agg["out_sum_total"] = sum(trx_agg.get(f"out_sum_{m}", 0) for m in available_months)
    trx_agg["in_count_total"] = sum(trx_agg.get(f"in_count_{m}", 0) for m in available_months)
    trx_agg["in_sum_total"] = sum(trx_agg.get(f"in_sum_{m}", 0) for m in available_months)
    
    # Decay features
    if "out_count_jan" in trx_agg.columns and "out_count_feb" in trx_agg.columns:
        trx_agg["trx_count_decay_jan_feb"] = (
            (trx_agg["out_count_feb"] - trx_agg["out_count_jan"]) / (trx_agg["out_count_jan"] + 1)
        )
    if "out_count_feb" in trx_agg.columns and "out_count_march" in trx_agg.columns:
        trx_agg["trx_count_decay_feb_march"] = (
            (trx_agg["out_count_march"] - trx_agg["out_count_feb"]) / (trx_agg["out_count_feb"] + 1)
        )
    if "out_count_jan" in trx_agg.columns and "out_count_march" in trx_agg.columns:
        trx_agg["trx_count_decay_jan_march"] = (
            (trx_agg["out_count_march"] - trx_agg["out_count_jan"]) / (trx_agg["out_count_jan"] + 1)
        )
    
    # March activity share
    if "out_count_march" in trx_agg.columns:
        trx_agg["march_activity_share"] = trx_agg["out_count_march"] / (trx_agg["out_count_total"] + 1e-5)
    
    # Service diversity
    trx_types = ["P2P", "MerchantPay", "BillPay", "CashIn", "CashOut"]
    for ttype in trx_types:
        total = 0
        for m in available_months:
            col = f"count_{ttype}_{m}"
            if col in trx_agg.columns:
                total = total + trx_agg[col]
        trx_agg[f"count_{ttype}_total"] = total
    
    total_type_cols = [f"count_{ttype}_total" for ttype in trx_types]
    trx_agg["trx_type_diversity"] = (trx_agg[total_type_cols] > 0).sum(axis=1)
    
    # Ratios
    denom = trx_agg["out_count_total"] + 1e-5
    trx_agg["merchant_pay_ratio"] = trx_agg["count_MerchantPay_total"] / denom
    trx_agg["bill_pay_ratio"] = trx_agg["count_BillPay_total"] / denom
    trx_agg["p2p_ratio"] = trx_agg["count_P2P_total"] / denom
    trx_agg["cashout_ratio"] = trx_agg["count_CashOut_total"] / denom
    
    # Net flow March
    if "in_sum_march" in trx_agg.columns and "out_sum_march" in trx_agg.columns:
        trx_agg["net_flow_march"] = trx_agg["in_sum_march"] - trx_agg["out_sum_march"]
    
    return trx_agg


def aggregate_balances(target_ids):
    """Aggregate daily balances — one month at a time, chunked pivot for March with advanced trend features."""
    print("Aggregating daily balances...")
    bal_files = sorted(glob.glob(os.path.join(base_path, "dayend_balance", "*.parquet")))
    months = ["jan", "feb", "march"]
    
    bal_agg = pd.DataFrame(index=list(target_ids))
    bal_agg.index.name = "ACCOUNT_ID"
    
    for month_idx, filepath in enumerate(bal_files):
        if month_idx >= len(months):
            break
        month = months[month_idx]
        print(f"  Processing balances for {month}...")
        
        df = pd.read_parquet(filepath, columns=["ACCOUNT_ID", "DATE", "AVAILABLE_BALANCE"])
        df = df[df["ACCOUNT_ID"].isin(target_ids)]
        
        # Monthly base stats
        stats = df.groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"].agg(
            mean_bal="mean",
            std_bal="std",
            min_bal="min",
            max_bal="max"
        )
        stats.columns = [f"{col}_{month}" for col in stats.columns]
        bal_agg = bal_agg.join(stats, how="left").fillna(0)
        del stats
        
        # March detailed features — CHUNKED pivot to save memory
        if month == "march":
            df["day"] = pd.to_datetime(df["DATE"]).dt.day
            
            # Process pivot in chunks of 100K customers
            target_list = list(target_ids)
            chunk_size = 100_000
            
            final_bal = pd.Series(0.0, index=target_list, name="final_balance_march")
            bal_drop = pd.Series(0.0, index=target_list, name="balance_drop_march")
            bal_trend = pd.Series(0.0, index=target_list, name="balance_trend_march")
            zero_days = pd.Series(0, index=target_list, name="zero_balance_days_march")
            
            # New Advanced micro-window balance stats
            mean_bal_7d = pd.Series(0.0, index=target_list, name="mean_balance_last_7d_march")
            mean_bal_14d = pd.Series(0.0, index=target_list, name="mean_balance_last_14d_march")
            zero_days_7d = pd.Series(0, index=target_list, name="zero_balance_days_last_7d_march")
            bal_trend_14d = pd.Series(0.0, index=target_list, name="balance_trend_last_14d_march")
            
            for chunk_start in range(0, len(target_list), chunk_size):
                chunk_ids = set(target_list[chunk_start:chunk_start + chunk_size])
                chunk_df = df[df["ACCOUNT_ID"].isin(chunk_ids)]
                
                if len(chunk_df) == 0:
                    continue
                
                pivot = chunk_df.pivot(
                    index="ACCOUNT_ID", columns="day", values="AVAILABLE_BALANCE"
                ).fillna(0)
                
                # Align days to 1 to 31
                all_days = list(range(1, 32))
                pivot = pivot.reindex(columns=all_days, fill_value=0.0)
                
                # Final balance
                final_bal.update(pivot[31])
                
                # Balance drop
                bal_drop.update(pivot[31] - pivot[1])
                
                # Linear trend slope (31 days)
                t_centered = np.array(all_days) - 16.0
                denom_val = (t_centered ** 2).sum()
                weights = t_centered / denom_val
                trend = pd.Series(np.dot(pivot.values, weights), index=pivot.index)
                bal_trend.update(trend)
                
                # Zero balance days (31 days)
                zd = (pivot < 10.0).sum(axis=1)
                zero_days.update(zd)
                
                # Mean balance last 7 days (days 25-31)
                mean_7 = pivot[list(range(25, 32))].mean(axis=1)
                mean_bal_7d.update(mean_7)
                
                # Mean balance last 14 days (days 18-31)
                mean_14 = pivot[list(range(18, 32))].mean(axis=1)
                mean_bal_14d.update(mean_14)
                
                # Zero balance days last 7 days (days 25-31)
                zd_7 = (pivot[list(range(25, 32))] < 10.0).sum(axis=1)
                zero_days_7d.update(zd_7)
                
                # Linear trend last 14 days (days 18-31)
                t_14 = np.array(range(1, 15)) - 7.5
                denom_14 = (t_14 ** 2).sum()
                weights_14 = t_14 / denom_14
                trend_14 = pd.Series(np.dot(pivot[list(range(18, 32))].values, weights_14), index=pivot.index)
                bal_trend_14d.update(trend_14)
                
                del pivot, chunk_df
                gc.collect()
            
            bal_agg["final_balance_march"] = final_bal
            bal_agg["balance_drop_march"] = bal_drop
            bal_agg["balance_trend_march"] = bal_trend
            bal_agg["zero_balance_days_march"] = zero_days
            bal_agg["mean_balance_last_7d_march"] = mean_bal_7d
            bal_agg["mean_balance_last_14d_march"] = mean_bal_14d
            bal_agg["zero_balance_days_last_7d_march"] = zero_days_7d
            bal_agg["balance_trend_last_14d_march"] = bal_trend_14d
            
            del final_bal, bal_drop, bal_trend, zero_days, mean_bal_7d, mean_bal_14d, zero_days_7d, bal_trend_14d
        
        del df
        gc.collect()
    
    # Stability
    bal_agg["balance_stability_march"] = bal_agg["std_bal_march"] / (bal_agg["mean_bal_march"] + 1e-5)
    bal_agg["balance_change_jan_march"] = bal_agg["mean_bal_march"] - bal_agg["mean_bal_jan"]
    bal_agg["balance_change_feb_march"] = bal_agg["mean_bal_march"] - bal_agg["mean_bal_feb"]
    
    # New Final balance to mean balance ratio
    bal_agg["final_to_mean_balance_ratio_march"] = bal_agg["final_balance_march"] / (bal_agg["mean_bal_march"] + 1e-5)
    
    return bal_agg


def build_features(sample_fraction=1.0, seed=42):
    """Run end-to-end feature engineering with memory-safe processing and advanced features."""
    print(f"Starting feature engineering pipeline (sample={sample_fraction}, seed={seed})...")
    
    train_labels = pd.read_csv(os.path.join(base_path, "train_labels.csv"))
    test_labels = pd.read_csv(os.path.join(base_path, "test.csv"))
    
    if sample_fraction < 1.0:
        print(f"Sampling {sample_fraction*100}% of the data...")
        sampled_indices = train_labels.groupby("CHURN", group_keys=False).apply(
            lambda x: x.sample(frac=sample_fraction, random_state=seed)
        ).index
        train_sampled = train_labels.loc[sampled_indices].reset_index(drop=True)
        test_sampled = test_labels.sample(frac=sample_fraction, random_state=seed)
    else:
        train_sampled = train_labels
        test_sampled = test_labels
    
    target_ids = set(train_sampled["ACCOUNT_ID"]).union(set(test_sampled["ACCOUNT_ID"]))
    print(f"Total target customers: {len(target_ids):,}")
    
    # 1. KYC
    kyc_df = load_kyc(target_ids)
    gc.collect()
    
    # 2. Recency (lightweight pass across all 3 months)
    recency_df = compute_recency(target_ids)
    gc.collect()
    
    # 3. Advanced March transaction features
    march_adv_df = compute_advanced_march_features(target_ids)
    gc.collect()
    
    # 4. General transaction aggregations (Jan + Feb + March sums/counts)
    trx_df = aggregate_transactions(target_ids)
    trx_df = trx_df.join(recency_df, how="left")
    trx_df["days_since_last_trx"] = trx_df["days_since_last_trx"].fillna(90)
    del recency_df
    
    # Join advanced March transaction features
    trx_df = trx_df.join(march_adv_df, how="left")
    del march_adv_df
    gc.collect()
    
    # 5. Balance aggregation
    bal_df = aggregate_balances(target_ids)
    gc.collect()
    
    # 6. Join all
    features = kyc_df.join(trx_df, how="left").join(bal_df, how="left").fillna(0)
    del kyc_df, trx_df, bal_df
    gc.collect()
    
    # 7. Sparsity flags
    features["flag_zero_bill_pay"] = (features["bill_pay_ratio"] == 0).astype(float)
    features["flag_zero_merchant_pay"] = (features["merchant_pay_ratio"] == 0).astype(float)
    if "out_count_march" in features.columns:
        features["flag_zero_march_trx"] = (features["out_count_march"] == 0).astype(float)
        
    # Zero activity in the last 7 days of March flag
    if "out_count_last_7d" in features.columns:
        features["flag_zero_trx_last_7d_march"] = (features["out_count_last_7d"] == 0).astype(float)
    
    # 8. Log transforms for skewed monetary columns
    skewed_cols = [
        "out_sum_total", "in_sum_total", "out_sum_march", "in_sum_march",
        "mean_bal_march", "std_bal_march", "final_balance_march",
        "out_sum_last_7d", "out_sum_last_14d", "in_sum_last_7d", "in_sum_last_14d",
        "mean_balance_last_7d_march", "mean_balance_last_14d_march"
    ]
    for col in skewed_cols:
        if col in features.columns:
            features[f"{col}_log"] = np.log1p(features[col].clip(lower=0))
    
    # 9. Split back
    features = features.reset_index()
    
    train_features = features[features["ACCOUNT_ID"].isin(train_sampled["ACCOUNT_ID"])].copy()
    train_features = train_features.merge(train_sampled, on="ACCOUNT_ID", how="left")
    
    test_features = features[features["ACCOUNT_ID"].isin(test_sampled["ACCOUNT_ID"])].copy()
    
    del features
    gc.collect()
    
    print(f"\nTrain features: {train_features.shape}, Test features: {test_features.shape}")
    
    os.makedirs("./processed_data", exist_ok=True)
    train_features.to_parquet("./processed_data/train_features.parquet")
    test_features.to_parquet("./processed_data/test_features.parquet")
    print("Features saved to processed_data/.")
    
    generate_feature_catalog()


def generate_feature_catalog():
    """Write the features.md catalog."""
    catalog = """# FictiPay Churn Prediction: Advanced Feature Catalog

## 1. Tenure & Demographics (KYC)
- **tenure_days**: Days from ACCOUNT_OPEN_DATE to 2024-03-31.
- **gender_*, region_***: One-hot encoded demographics.

## 2. General Transaction Activity (Jan + Feb + March)
- **out_count_[jan/feb/march/total]**: Outbound transaction counts per month and total.
- **out_sum_[jan/feb/march/total]**: Total outbound amount (TK) per month.
- **out_avg_[jan/feb/march]**: Average transaction size per month.
- **out_std_[jan/feb/march]**: Std dev of transaction amounts.
- **out_max_[jan/feb/march]**: Max single transaction.
- **in_count_[jan/feb/march/total]**: Inbound transaction counts.
- **in_sum_[jan/feb/march/total]**: Total inbound volume.

## 3. Directional & Type Recency (NEW - March Advanced)
- **days_since_last_trx**: Days between last transaction (any direction) and 2024-03-31.
- **days_since_last_outbound**: Days since last outbound transaction.
- **days_since_last_inbound**: Days since last inbound transaction.
- **days_since_last_P2P / days_since_last_MerchantPay / days_since_last_BillPay / days_since_last_CashOut**: Outbound recency per type.
- **days_since_last_CashIn / days_since_received_P2P**: Inbound recency per type.

## 4. Micro-Windows & Velocity (NEW - March Advanced)
- **out_count_last_7d / out_sum_last_7d**: Count and sum of outbound transactions in the last week of March (March 25-31).
- **out_count_last_14d / out_sum_last_14d**: Count and sum of outbound transactions in the last 2 weeks of March (March 18-31).
- **in_count_last_7d / in_sum_last_7d / in_count_last_14d / in_sum_last_14d**: Inbound micro-window stats.
- **out_count_velocity_7d / out_count_velocity_14d / out_sum_velocity_7d / out_sum_velocity_14d**: Ratio of micro-window spend/count to the total March spend/count.

## 5. Temporal Decay (3-Month Trends)
- **trx_count_decay_jan_feb / trx_count_decay_feb_march / trx_count_decay_jan_march**: Rate of activity change.
- **march_activity_share**: March count / total count.

## 6. Service Diversity & Type Ratios
- **count_[P2P/MerchantPay/BillPay/CashIn/CashOut]_total**: Type-level total activity.
- **trx_type_diversity**: Number of distinct types used (0-5).
- **merchant_pay_ratio / bill_pay_ratio / p2p_ratio / cashout_ratio**: Spend composition.
- **flag_zero_bill_pay / flag_zero_merchant_pay / flag_zero_march_trx / flag_zero_trx_last_7d_march**: Zero-inflation/zero-activity flags.

## 7. Net Flow
- **net_flow_march**: March inbound minus outbound. Negative = wallet draining.

## 8. Balance Trends & Micro-Windows (NEW)
- **mean_bal_[jan/feb/march]**: Average daily balance per month.
- **mean_balance_last_7d_march / mean_balance_last_14d_march**: Average balance in the last week and two weeks of March.
- **final_balance_march**: Balance on March 31.
- **balance_drop_march**: March 31 - March 1 change.
- **balance_trend_march**: Linear slope of daily balance over March (31 days).
- **balance_trend_last_14d_march**: Linear slope of daily balance over last 14 days.
- **zero_balance_days_march / zero_balance_days_last_7d_march**: Days with balance < 10 TK.
- **balance_stability_march**: CV (std/mean) of March balance.
- **balance_change_jan_march / balance_change_feb_march**: Cross-month balance trends.
- **final_to_mean_balance_ratio_march**: Final balance relative to mean balance.
"""
    with open("/home/ahnaf-zakaria/Desktop/Datathon/features.md", "w") as f:
        f.write(catalog)
    print("Feature catalog features.md updated.")


if __name__ == "__main__":
    build_features(sample_fraction=1.0)
