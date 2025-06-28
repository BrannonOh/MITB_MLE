import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, avg, stddev, min, max, count, lit

from sklearn.metrics import f1_score, roc_auc_score
import pandas as pd

#####################
# PERFORMANCE DRIFT # 
#####################

def check_performance_drift(spark, perf_path, snapshotdate, modelname):
    # Get all prediction files
    pred_paths = sorted(glob.glob(f"datamart/gold/model_predictions/{modelname}/{modelname}_predictions_*.parquet"))

    if not pred_paths:
        print(f"⚠️ [SKIP] No prediction files found up till {snapshotdate} for model: '{modelname}'. Skipping performance drift check.")
        return

    # Load all label files
    label_paths = sorted(glob.glob(f"datamart/gold/label_store/GOLD_label_store_*.parquet"))
    labels_df = spark.read.parquet(*label_paths).withColumn("snapshot_date", col("snapshot_date").cast("date"))

    # Load all prediction files
    pred_paths = sorted(glob.glob(f"datamart/gold/model_predictions/credit_model_2024_09_01/credit_model_2024_09_01_predictions_*.parquet"))
    preds_df = spark.read.parquet(*pred_paths).withColumn("snapshot_date", col("snapshot_date").cast("date"))

    # Threshold into binary predictions
    preds_df = preds_df.withColumn("binary_prediction", when(col("model_predictions") > 0.5, lit(1)).otherwise(lit(0)))
    
     # Join with labels
    joined_df = preds_df.join(labels_df, on=["Customer_ID", "snapshot_date"], how="inner")
    
    # Convert to Pandas for metric computation
    monthly_data = joined_df.select("Customer_ID", "snapshot_date", "model_predictions", "binary_prediction", "label") \
                            .toPandas()
    
    # Compute metrics per snapshot_date
    results = []
    for date in sorted(monthly_data["snapshot_date"].unique()):
        df = monthly_data[monthly_data["snapshot_date"] == date]
        if df["label"].nunique() < 2:
            continue

        y_true = df["label"]
        y_pred = df["binary_prediction"]
        y_score = df["model_predictions"]

        results.append({
            "snapshot_date": date,
            "f1": f1_score(y_true, y_pred),
            "auc": roc_auc_score(y_true, y_score),
        })

    # Save metrics
    perf_df = pd.DataFrame(results).sort_values("snapshot_date")
    output_path = f"datamart/gold/model_monitoring/{modelname}/performance_drift_summary/performance_metrics.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    perf_df.to_csv(output_path, index=False)

    # Load and evaluate snapshot-specific drift
    df = pd.read_csv(output_path)
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    snapshot_dt = pd.to_datetime(snapshotdate)

    row = df[df["snapshot_date"] == snapshot_dt]

    if row.empty:
        print(f"⚠️ [SKIP] No performance drift monitoring as performance metrics not found for {snapshotdate}.")
        return

    auc = row["auc"].values[0]
    snapshot = row["snapshot_date"].values[0]

    if auc < 0.70:
        # Save drift flag file 
        drift_flag = f"datamart/gold/model_monitoring/{modelname}/drift_flags/{snapshotdate}_performance_drift.txt"
        os.makedirs(os.path.dirname(drift_flag), exist_ok=True)
        with open(drift_flag, "w") as f:
            f.write(f"AUC dropped to {auc:.3f} on {snapshot}.\n")
        raise ValueError(f"\U0001F534 Critical: AUC dropped to {auc:.3f} on {snapshot}. Log saved @ {drift_flag}.")
    elif auc < 0.75:
        print(f"\U0001F7E1 Warning: AUC = {auc:.3f} on {snapshot}. Monitor closely.")
    else:
        print(f"\U0001F7E2 Healthy: AUC = {auc:.3f} on {snapshot}.")

#################
# FEATURE DRIFT # 
#################

def check_feature_drift(snapshotdate, spark, modelname):
    baseline_date = datetime(2024, 6, 1) 
    snap = datetime.strptime(snapshotdate, "%Y-%m-%d")
    if snap <= baseline_date: 
        print(f"⚠️ [SKIP] No feature drift monitoring for {snapshotdate} (before or on baseline date {baseline_date})")
        return 

    baseline_date_str = "2024_06_01"
    baseline_path = f"datamart/gold/feature_store/GOLD_feature_store_{baseline_date_str}.parquet"
    baseline_df = spark.read.parquet(baseline_path).toPandas().dropna()

    def calculate_psi(expected, actual, buckets=10):
        expected = np.array(expected)
        actual = np.array(actual)

        quantiles = np.linspace(0, 1, buckets + 1)
        bin_edges = np.quantile(expected, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        expected_counts = np.histogram(expected, bins=bin_edges)[0]
        actual_counts = np.histogram(actual, bins=bin_edges)[0]

        expected_pct = expected_counts / expected_counts.sum()
        actual_pct = actual_counts / actual_counts.sum()

        psi_values = (actual_pct - expected_pct) * np.log((actual_pct + 1e-6) / (expected_pct + 1e-6))
        return np.sum(psi_values)

    # Load current snapshot features
    current_path = f"datamart/gold/feature_store/GOLD_feature_store_{snapshotdate.replace('-', '_')}.parquet"
    current_df = spark.read.parquet(current_path).toPandas().dropna()

    psi_results = []
    common_cols = [col for col in baseline_df.columns if col in current_df.columns and col not in ["Customer_ID", "snapshot_date"]]

    for col_name in common_cols:
        try:
            psi = calculate_psi(baseline_df[col_name], current_df[col_name])
            psi_results.append({"snapshot_date": pd.to_datetime(snapshotdate), "feature": col_name, "psi": psi})
        except Exception as e:
            continue

    psi_df = pd.DataFrame(psi_results).sort_values(["snapshot_date", "feature"])

    # Save PSI results
    output_dir = f"datamart/gold/model_monitoring/{modelname}/feature_drift_summary"
    os.makedirs(output_dir, exist_ok=True)
    psi_log_path = f"{output_dir}/feature_drift_psi.csv"
    if os.path.exists(psi_log_path):
        # Load existing log and append new rows, avoiding duplicates
        psi_log = pd.read_csv(psi_log_path, parse_dates=["snapshot_date"])
        # Remove existing rows for this snapshot_date to prevent duplication
        psi_log = psi_log[psi_log["snapshot_date"] != pd.to_datetime(snapshotdate)]
        psi_log = pd.concat([psi_log, psi_df], ignore_index=True)
    else:
        psi_log = psi_df

    psi_log = psi_log.sort_values(["snapshot_date", "feature"])
    psi_log.to_csv(psi_log_path, index=False)

    # Evaluate Drift Alert
    alerts = []
    grouped = psi_df.groupby("snapshot_date")
    for snapshot, group in grouped:
        n_features = group['feature'].nunique()
        n_critical = (group['psi'] > 0.2).sum()
        n_warning = ((group["psi"] > 0.1) & (group["psi"] <= 0.2)).sum()
        pct_drifted = (n_critical + n_warning) / n_features

        if n_critical >= 5 or pct_drifted >= 0.3:
            alerts.append(f"{snapshot.date()}: {n_critical} critical, {n_warning} warning, {pct_drifted:.0%} drifted")

    if alerts:
        drift_flag = f"datamart/gold/model_monitoring/{modelname}/drift_flags/{snapshotdate}_feature_drift.txt"
        os.makedirs(os.path.dirname(drift_flag), exist_ok=True)
        with open(drift_flag, "w") as f:
            f.write("Feature drift alert triggered:\n" + "\n".join(alerts))
        msg = "\U0001F6A8 Feature drift alert triggered:\n" + "\n".join(alerts)
        raise ValueError(msg)
    else:
        print(f"\U0001F7E2 No significant feature drift on {snapshotdate}.")

def main(snapshotdate, modelname):
    print(f"""
    ================================
    STARTING DRIFT MONITORING FOR: {snapshotdate}  
    ================================
    """)

    spark = pyspark.sql.SparkSession.builder \
        .appName("drift-monitor") \
        .master("local[*]") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    performance_path = f"datamart/gold/model_monitoring/{modelname}/performance_drift_summary/performance_metrics.csv"

    check_performance_drift(spark, performance_path, snapshotdate, modelname)
    check_feature_drift(snapshotdate, spark, modelname)

    spark.stop()

    print(f"""
    ================================
    COMPLETED DRIFT MONITORING FOR: {snapshotdate}  
    ================================
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run drift monitoring job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="Model version name")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
