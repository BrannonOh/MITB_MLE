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

import utils.data_processing_gold_feature_store_loan_daily
import utils.data_processing_gold_feature_store_financials
import utils.data_processing_gold_feature_store_attributes
import utils.data_processing_gold_feature_store_clickstream

# To call this script: python bronze_label_store.py --snapshotdate "2023-01-01"

def main(snapshotdate):
    print(f"""
    ================================
    STARTING JOB FOR: {snapshotdate}  
    ================================
    """)
    
    # Initialize SparkSession
    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    
    # Set log level to ERROR to hide warnings
    spark.sparkContext.setLogLevel("ERROR")

    # load arguments
    date_str = snapshotdate
    
    # create gold datalake
    gold_output_directory = "datamart/gold/feature_store/"
    
    if not os.path.exists(gold_output_directory):
        os.makedirs(gold_output_directory)

    loan_daily_gold_df = utils.data_processing_gold_feature_store_loan_daily.process_gold_features(
    date_str, 'loan_daily', 'datamart/silver/loan_daily/', None, spark, mob=6)
    
    financials_gold_df = utils.data_processing_gold_feature_store_financials.process_gold_features(
    'financials', 'datamart/silver/financials/', None, spark)
    
    attributes_gold_df = utils.data_processing_gold_feature_store_attributes.process_gold_features(
    'attributes', 'datamart/silver/attributes/', None, spark)
    
    clickstream_gold_df = utils.data_processing_gold_feature_store_clickstream.process_gold_features(
    date_str, 'clickstream', 'datamart/silver/clickstream/', None, spark)
        
    gold_df = loan_daily_gold_df.join(financials_gold_df, on='Customer_ID', how='left') \
    .join(attributes_gold_df, on='Customer_ID', how='left') \
    .join(clickstream_gold_df, on=['Customer_ID', 'snapshot_date'], how='left')
 
    partition_name = f"GOLD_feature_store_{date_str.replace('-','_')}.parquet"
    filepath = gold_output_directory + partition_name
    gold_df.write.mode("overwrite").parquet(filepath)

    # end spark session
    spark.stop()
    
    print(f"""
    ================================
    COMPLETED JOB FOR: {snapshotdate}  
    ================================
    """)

if __name__ == "__main__":
    # Setup argparse to parse command-line arguments
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    
    args = parser.parse_args()
    
    # Call main with arguments explicitly passed
    main(args.snapshotdate)