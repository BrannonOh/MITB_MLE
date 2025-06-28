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

import utils.data_processing_bronze_table_lms_clickstream

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
    bronze_lms_directory = "datamart/bronze/"
    
    if not os.path.exists(bronze_lms_directory):
        os.makedirs(bronze_lms_directory)

    # run data processing
    utils.data_processing_bronze_table_lms_clickstream.process_bronze_table(
        snapshot_date_str=date_str, 
        input_dir='data/lms_loan_daily.csv', 
        table_name='loan_daily',
        output_dir=bronze_lms_directory,
        spark=spark
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