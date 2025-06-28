# IMPORT ALL PACKAGES
# 1 - General-Purpose & Utility Libraries 
# Interacting with the OS - file paths, environment variables etc. 
import os 
# Finding file paths using wildcard patterns (e.g. 'data/*.csv')
import glob 
# Pretty-printing Python objects, useful for inspecting nested structures (like dicts).
import pprint 

# 2 - Data & Visualization 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import random 

# 3 - Date and Time 
# Creating and manipulating dates/times 
from datetime import datetime, timedelta 
# More flexible relative date operations (e.g. 'add 1 month')
from dateutil.relativedelta import relativedelta 

# 4 - PySpark (Distributed Data Processing) 
# Core PySpark module 
import pyspark 
# Built-in function for Spark DataFrame manipulation 
import pyspark.sql.functions as F 
# Shortcut for selecting and transforming columns 
from pyspark.sql.functions import col 
# Explicitly specifiying data types for schema definitions in Spark 
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType, DoubleType

def process_gold_features(snapshot_date_str, table_name, input_dir, output_dir, spark, mob):
    
    # Prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # Connect to silver table
    partition_name = f"SILVER_{table_name}_{snapshot_date_str.replace('-','_')}.parquet"
    filepath = input_dir + partition_name
    df = spark.read.parquet(filepath)
    print('Loaded from:', filepath, '| Row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)
    
    # Feature Engineering 
    df = df.withColumn('payment_ratio', 
                       F.when(col('due_amt') != 0, col('paid_amt') / col('due_amt'))
                       .otherwise(0).cast(FloatType()))
    # df = df.withColumn('loan_progress',
    #                    F.when(col('tenure') != 0, col('installment_num') / col('tenure'))
    #                    .otherwise(0).cast(FloatType()))
    df = df.withColumn('overdue_ratio', 
                       F.when(col('loan_amt') != 0, col('overdue_amt') / col('loan_amt'))
                       .otherwise(0).cast(FloatType()))
    # df = df.withColumn('is_active', # If balance is > 0, it means the user might be an active user.
    #                    F.when(col('balance') > 0, 1)
    #                    .otherwise(0).cast(IntegerType()))
    
    # Clean data: enforce schema / data type 
    # Dictionary specifying columns and their desired datatypes 
    column_type_map = {
        'loan_id': StringType(),
        'Customer_ID': StringType(),
        'payment_ratio': FloatType(),
        'overdue_ratio': FloatType(),
        'balance': IntegerType(),
        'snapshot_date': DateType(),
        'installments_missed': IntegerType(),
        'dpd': IntegerType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # Select columns to save
    df = df.select('loan_id', 'Customer_ID', 'payment_ratio', 'overdue_ratio', 'balance', 'snapshot_date',
                   'installments_missed', 'dpd')
    
    return df