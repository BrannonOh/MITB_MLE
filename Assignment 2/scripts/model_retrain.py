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
import pickle

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, avg, stddev, min, max, count, lit


from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

######################
# Retrain Model Code # 
######################

def retrain_model(spark, snapshotdate):
    # 0. Set up config
    model_train_date_str = snapshotdate
    train_test_period_months = 12
    oot_period_months = 1
    train_test_ratio = 0.8

    config = {}
    config["model_train_date_str"] = model_train_date_str
    config["train_test_period_months"] = train_test_period_months
    config["oot_period_months"] =  oot_period_months
    config["model_train_date"] =  datetime.strptime(model_train_date_str, "%Y-%m-%d")
    config["oot_end_date"] =  config['model_train_date'] - timedelta(days = 1)
    config["oot_start_date"] =  config['model_train_date'] - relativedelta(months = oot_period_months)
    config["train_test_end_date"] =  config["oot_start_date"] - timedelta(days = 1)
    config["train_test_start_date"] =  config["oot_start_date"] - relativedelta(months = train_test_period_months)
    config["train_test_ratio"] = train_test_ratio 

    # 1. Get labels from the label store; Connect to the label store
    folder_path = "datamart/gold/label_store/"
    files_list = [folder_path+os.path.basename(f) for f in glob.glob(os.path.join(folder_path, '*'))]
    label_store_sdf = spark.read.option("header", "true").parquet(*files_list)
    labels_sdf = label_store_sdf.filter((col("snapshot_date") >= config["train_test_start_date"]) & (col("snapshot_date") <= config["oot_end_date"]))

    # 2. Get the features from feature store 
    # All montly files are in the same folder 
    feature_store_path = "datamart/gold/feature_store/"
    features_store_sdf = spark.read.option('mergeSchema', 'true').parquet(f'{feature_store_path}/*.parquet')
    features_sdf = features_store_sdf.filter( (col('snapshot_date') >= config['train_test_start_date']) & (col('snapshot_date') <= config['oot_end_date']) )

    # 3. Prepare data for modelling 
    # For labels, snapshot_date is unique for each Customer_ID. And, Customer_ID is unique per entry. No duplicates. 
    data_pdf = labels_sdf.join(features_sdf, on=['Customer_ID', 'snapshot_date'], how='left').toPandas()
    data_pdf = data_pdf.dropna()

    # Split dataset into Train, Test, & OOT 
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config['train_test_start_date'].date()) & (data_pdf['snapshot_date'] <= config['train_test_end_date'].date())]
    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config['oot_start_date'].date()) & (data_pdf['snapshot_date'] <= config['oot_end_date'].date())]
    cols_to_drop = [
        'Customer_ID', 'snapshot_date', 'loan_id', 'label', 'label_def', 'loan_id', 
        'payment_ratio', 'overdue_ratio', 'balance', 'installments_missed', 'dpd'
    ]
    # Exclude these features as it'll cause data leakage (Perfect AUC score when classifying):
    # 'payment_ratio', 'overdue_ratio', 'balance', 'installments_missed', 'dpd'

    # Split train_test_pdf into X_train, X_val, y_train, y_val
    X_train_val = train_test_pdf.drop(columns=cols_to_drop)
    y_train_val = train_test_pdf['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size = 1 - config['train_test_ratio'],
        random_state=42, 
        shuffle=True,
        stratify=y_train_val
    )

    # Split oot_pdf into X_oot, y_oot
    X_oot = oot_pdf.drop(columns=cols_to_drop)
    y_oot = oot_pdf['label']

    # 4. Preprocessing (don't need for XGBoost)
    # 5. Train the model 
    # Define the XGBoost classifier
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    # Define the hyperparameter space to search
    param_dist = {
        'n_estimators': [100, 200, 300, 500],             # Number of boosting rounds
        'max_depth': [3, 5, 7, 9],                        # Tree depth
        'learning_rate': [0.01, 0.05, 0.1, 0.2],          # Step size shrinkage
        'subsample': [0.6, 0.8, 1.0],                     # Row sampling per tree
        'colsample_bytree': [0.6, 0.8, 1.0],              # Feature sampling per tree
        'gamma': [0, 0.1, 0.3, 1],                        # Minimum loss reduction (regularization)
        'min_child_weight': [1, 3, 5, 10],                # Minimum sum of instance weight
        'reg_alpha': [0, 0.01, 0.1, 1],                   # L1 regularization
        'reg_lambda': [1, 1.5, 2, 5],                     # L2 regularization
        'scale_pos_weight': [1, 2, 5]                     # To handle class imbalance
    }
    # Create a scorer based on AUC score
    auc_scorer = make_scorer(roc_auc_score)

    # Set up the random search with cross-validation
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=100,  # Number of iterations for random search
        cv=3,       # Number of folds in cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1   # Use all available cores
    )
    # Perform the random search
    random_search.fit(X_train, y_train)

    # Output the best parameters and best score
    print("Best parameters found: ", random_search.best_params_)
    print("Best AUC score:", random_search.best_score_)

    # Evaluate the model on the train set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_train)[:, 1]
    train_auc_score = roc_auc_score(y_train, y_pred_proba)
    print("Train AUC score:", train_auc_score)

    # Evaluate the model on the val set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    val_auc_score = roc_auc_score(y_val, y_pred_proba)
    print("Val AUC score:", val_auc_score)

    # Evaluate the model on the oot set
    best_model = random_search.best_estimator_
    y_pred_proba = best_model.predict_proba(X_oot)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("OOT AUC score:", oot_auc_score)

    print("TRAIN GINI score:", round(2*train_auc_score-1,3))
    print("VAL GINI score:", round(2*val_auc_score-1,3))
    print("OOT GINI score:", round(2*oot_auc_score-1,3))

    # 6. Save Model Artefact 
    model_artefact = {}
    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_"+config["model_train_date_str"].replace('-','_')
    model_artefact['preprocessing_transformers'] = {}
    model_artefact['preprocessing_transformers']['stdscaler'] = {} #transformer_stdscaler
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {}
    model_artefact['data_stats']['X_train'] = X_train.shape[0]
    model_artefact['data_stats']['X_val'] = X_val.shape[0]
    model_artefact['data_stats']['X_oot'] = X_oot.shape[0]
    model_artefact['data_stats']['y_train'] = round(y_train.mean(),2)
    model_artefact['data_stats']['y_val'] = round(y_val.mean(),2)
    model_artefact['data_stats']['y_oot'] = round(y_oot.mean(),2)
    model_artefact['results'] = {}
    model_artefact['results']['auc_train'] = train_auc_score
    model_artefact['results']['auc_val'] = val_auc_score
    model_artefact['results']['auc_oot'] = oot_auc_score
    model_artefact['results']['gini_train'] = round(2*train_auc_score-1,3)
    model_artefact['results']['gini_val'] = round(2*val_auc_score-1,3)
    model_artefact['results']['gini_oot'] = round(2*oot_auc_score-1,3)
    model_artefact['hp_params'] = random_search.best_params_

    # Create model_bank dir
    model_bank_directory = "model_bank/"
    if not os.path.exists(model_bank_directory):
        os.makedirs(model_bank_directory)

    # PKL
    # Full path to the file
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')

    # Write the model to a pickle file
    with open(file_path, 'wb') as file:
        pickle.dump(model_artefact, file)

    print(f"Model saved to: {file_path}")

    # 8. Test load pickle and make model inference
    # Load the model from the pickle file
    with open(file_path, 'rb') as file:
        loaded_model_artefact = pickle.load(file)

    y_pred_proba = loaded_model_artefact['model'].predict_proba(X_oot)[:, 1]
    oot_auc_score = roc_auc_score(y_oot, y_pred_proba)
    print("OOT AUC score:", oot_auc_score)

    print("Model loaded successfully!")

# #################################
# # Check drift flag for the date # 
# #################################

# def should_trigger_retraining(snapshotdate, modelname): 
#     # Compose flag file paths 
#     drift_flag_dir = f"datamart/gold/model_monitoring/{modelname}/drift_flags"
#     feature_flag = os.path.join(drift_flag_dir, f"{snapshotdate}_feature_drift.txt")
#     perf_flag = os.path.join(drift_flag_dir, f"{snapshotdate}_performance_drift.txt")
#     # Check for either flag 
#     return os.path.exists(feature_flag) or os.path.exists(perf_flag)

#####################################
# Retrain only if drift is detected # 
#####################################

# def maybe_retrain_model(spark, snapshotdate):
#     snapshotdate = snapshotdate
#     if should_trigger_retraining(snapshotdate): 
#         # Load data for these months, then train
#         # E.g., loop through months, read parquet files, concat
#         # After retraining, save and register the new model
#         print(f"Drift detected. Retraining with data from 13 months up till {snapshotdate}")
#         retrain_model(snapshotdate)
#     else:
#         print("No drift flag found. Skipping retraining.")

def main(snapshotdate, modelname):
    print(f"""
    =============================================
    STARTING RETRAINING FOR: {snapshotdate}  
    =============================================
    """)

    spark = pyspark.sql.SparkSession.builder \
        .appName("drift-monitor") \
        .master("local[*]") \
        .config("spark.sql.ansi.enabled", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    retrain_model(spark, snapshotdate)

    spark.stop()

    print(f"""
    =======================================
    COMPLETED RETRAINING FOR: {snapshotdate}  
    =======================================
    """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run model retraining job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="Model version name")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
