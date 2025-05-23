# main.py

# -------------------------------
# 1. Imports
# -------------------------------
import os 
import glob 
import pprint 
import pandas as pd 
import numpy as np
import random 
from datetime import datetime, timedelta 
from dateutil.relativedelta import relativedelta 
import pyspark 
import pyspark.sql.functions as F 
from pyspark.sql.functions import col 
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType
from utils import(
    data_processing_bronze_table_lms_clickstream,
    data_processing_bronze_table_financials_attributes,
    data_processing_silver_table_loan_daily, 
    data_processing_silver_table_financials,
    data_processing_silver_table_attributes,
    data_processing_silver_table_clickstream,
    data_processing_gold_feature_store_loan_daily,
    data_processing_gold_feature_store_financials,
    data_processing_gold_feature_store_clickstream,
    data_processing_gold_feature_store_attributes,
    data_processing_gold_label_store_loan_daily
)

# -------------------------------
# 2. Spark Session Setup
# -------------------------------
spark = pyspark.sql.SparkSession.builder\
    .appName('dev') \
    .master('local[*]') \
    .getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# -------------------------------
# 3. Parameters
# -------------------------------
snapshot_date_str = '2023-01-01'
start_date_str = '2023-01-01'
end_date_str = '2024-12-01'
bronze_output_dir = 'datamart/bronze/'
silver_output_dir = 'datamart/silver/'
gold_output_dir = 'datamart/gold/feature_store/'
os.makedirs((bronze_output_dir), exist_ok=True)
os.makedirs((silver_output_dir), exist_ok=True)
os.makedirs((gold_output_dir), exist_ok=True)

# -------------------------------
# 4. Main Execution Block
# -------------------------------
if __name__ == "__main__":

    def generate_first_day_of_month_dates(start_date_str, end_date_str):
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        first_of_month_dates = []
        current_date = datetime(start_date.year, start_date.month, 1)
    
        while current_date <= end_date: 
            first_of_month_dates.append(current_date.strftime('%Y-%m-%d'))
            
            if current_date.month == 12: 
                current_date = datetime(current_date.year+1, 1, 1)
            else: 
                current_date = datetime(current_date.year, current_date.month+1, 1)
                
        return first_of_month_dates
        
    dates_str_lst = generate_first_day_of_month_dates(start_date_str, end_date_str)
# -------------------------------
# 4.1 Bronze Layer Execution Block
# -------------------------------
    print("""
    ================================
    PHASE 1: BRONZE LAYER INITIATED  
    ================================
    """)
    
    bronze_config_split = {
        'lms_loan_daily': {
            'input_dir': 'data/lms_loan_daily.csv',
            'table_name': 'loan_daily'
        },
    
        'feature_clickstream': {
            'input_dir': 'data/feature_clickstream.csv',
            'table_name': 'clickstream'
        }
    }
    
    bronze_config_no_split = {
        'features_financials': {
            'input_dir': 'data/features_financials.csv',
            'table_name': 'financials'
        },
        'features_attributes': {
            'input_dir': 'data/features_attributes.csv',
            'table_name': 'attributes'
        }
    }
    
    for table_key, cfg in bronze_config_split.items(): 
        for date_str in dates_str_lst: 
            data_processing_bronze_table_lms_clickstream.process_bronze_table(
                snapshot_date_str = date_str, 
                input_dir = cfg['input_dir'],
                table_name = cfg['table_name'],
                output_dir = bronze_output_dir,
                spark = spark
            )
    
    for table_key, cfg in bronze_config_no_split.items(): 
        data_processing_bronze_table_financials_attributes.process_bronze_table(
            input_dir = cfg['input_dir'],
            table_name = cfg['table_name'],
            output_dir = bronze_output_dir,
            spark = spark
        )
        
    print("""
    ================================
    PHASE 1: BRONZE LAYER COMPLETED  
    ================================
    """)
# -------------------------------
# 4.2 Silver Layer Execution Block
# -------------------------------
    print("""
    ================================
    PHASE 2: SILVER LAYER INITIATED  
    ================================
    """)
    
    silver_config_split = {
        'lms_loan_daily': {
            'input_dir': 'datamart/bronze/loan_daily/',
            'table_name': 'loan_daily',
            'processor': data_processing_silver_table_loan_daily
        },
        'feature_clickstream': {
            'input_dir': 'datamart/bronze/clickstream/',
            'table_name': 'clickstream',
            'processor': data_processing_silver_table_clickstream
        }
    }
    
    silver_config_no_split = {
        'features_financials': {
            'input_dir': 'datamart/bronze/financials/',
            'table_name': 'financials',
            'processor': data_processing_silver_table_financials 
        },
        'features_attributes': {
            'input_dir': 'datamart/bronze/attributes/',
            'table_name': 'attributes',
            'processor': data_processing_silver_table_attributes
        }
    }
     
    for table_key, cfg in silver_config_split.items(): 
        for date_str in dates_str_lst: 
            cfg['processor'].process_silver_table(
                snapshot_date_str = date_str, 
                input_dir = cfg['input_dir'],
                table_name = cfg['table_name'],
                output_dir = silver_output_dir,
                spark = spark
            )
    
    for table_key, cfg in silver_config_no_split.items(): 
        cfg['processor'].process_silver_table(
            input_dir = cfg['input_dir'],
            table_name = cfg['table_name'],
            output_dir = silver_output_dir,
            spark = spark
        )
        
    print("""
    ================================
    PHASE 2: SILVER LAYER COMPLETED  
    ================================
    """)
# -------------------------------
# 4.3 Gold Layer Execution Block
# -------------------------------
    print("""
    ================================
    PHASE 3: GOLD LAYER INITIATED  
    ================================
    """)

    # Gold Label Store 
    for date_str in dates_str_lst: 
        snapshot_date_str = date_str
        silver_loan_daily_directory = 'datamart/silver/loan_daily/'
        gold_label_store_directory = 'datamart/gold/label_store/'
        spark = spark 
        dpd = 30
        mob = 6
        data_processing_gold_label_store_loan_daily.process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob)

    # Gold Feature Store 
    for date_str in dates_str_lst: 
        loan_daily_gold_df = data_processing_gold_feature_store_loan_daily.process_gold_features(
            date_str, 'loan_daily', 'datamart/silver/loan_daily/', None, spark, mob=6)
        financials_gold_df = data_processing_gold_feature_store_financials.process_gold_features(
            'financials', 'datamart/silver/financials/', None, spark)
        attributes_gold_df = data_processing_gold_feature_store_attributes.process_gold_features(
            'attributes', 'datamart/silver/attributes/', None, spark)
        clickstream_gold_df = data_processing_gold_feature_store_clickstream.process_gold_features(
            date_str, 'clickstream', 'datamart/silver/clickstream/', None, spark)
        
        gold_df = loan_daily_gold_df.join(financials_gold_df, on='Customer_ID', how='left') \
                          .join(attributes_gold_df, on='Customer_ID', how='left') \
                          .join(clickstream_gold_df, on=['Customer_ID', 'snapshot_date'], how='left')
     
        partition_name = f'GOLD_feature_store_{date_str.replace('-','_')}.parquet'
        filepath = gold_output_dir + partition_name
        gold_df.write.mode("overwrite").parquet(filepath)
        # df.toPandas().to_parquet(filepath,
        #           compression='gzip')
        print('Saved to:', filepath)
        
    print("""
    ================================
    PHASE 3: GOLD LAYER COMPLETED  
    ================================
    """)