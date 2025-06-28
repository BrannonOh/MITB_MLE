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

import utils.data_processing_gold_label_store_loan_daily

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
    
    # create bronze datalake
    gold_lms_directory = "datamart/gold/"
    
    if not os.path.exists(gold_lms_directory):
        os.makedirs(gold_lms_directory)

    # run data processing
    utils.data_processing_gold_label_store_loan_daily.process_labels_gold_table(
        snapshot_date_str=date_str, 
        silver_loan_daily_directory='datamart/silver/loan_daily/', 
        gold_label_store_directory='datamart/gold/label_store/',
        spark=spark,
        dpd=30,
        mob=6
    )
    
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