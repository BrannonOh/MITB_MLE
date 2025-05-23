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

def process_silver_table(snapshot_date_str, input_dir, table_name, output_dir, spark):

    # Connect to the Bronze table 
    partition_name = f'BRONZE_{table_name}_{snapshot_date_str.replace('-','_')}.csv'
    filepath = input_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, '| Row count:', df.count())

    # df = df.withColumn('snapshot_date', F.to_date(col('snapshot_date').cast('string'), 'd/M/yyyy'))

    # DATA CLEANING BEFORE CASTING DESIRED DATATYPES 
    # Given the fact that this is a bronze layer/bronze table and assuming that each 
    # clickstream feature are features derived from raw user activity (e.g. page visits, 
    # session times, click counts, etc.: Most clickstream features should be >= 0. Assume 
    # that they are not Z-scores or mean-centered values as it is a bronze layer/table.
    # Likely data errors. 
    ls_of_clickstream_features = ['fe_1', 'fe_2', 'fe_3', 'fe_4', 'fe_5', 'fe_6', 'fe_7', 'fe_8', 'fe_9', 'fe_10', 'fe_11', 'fe_12', 'fe_13', 'fe_14', 'fe_15', 'fe_16', 'fe_17', 'fe_18', 'fe_19','fe_20']

    for clickstream_feature in ls_of_clickstream_features: 
        df = df.withColumn(clickstream_feature, F.when(col(clickstream_feature) > 0, col(clickstream_feature)).otherwise(0))
    
    # Clean data: enforce schema / data type 
    # Dictionary specifying columns and their desired datatypes 
    column_type_map = {
        'fe_1': IntegerType(),
        'fe_2': IntegerType(),
        'fe_3': IntegerType(),
        'fe_4': IntegerType(),
        'fe_5': IntegerType(),
        'fe_6': IntegerType(),
        'fe_7': IntegerType(),
        'fe_8': IntegerType(),
        'fe_9': IntegerType(),
        'fe_10': IntegerType(),
        'fe_11': IntegerType(),
        'fe_12': IntegerType(),
        'fe_13': IntegerType(), 
        'fe_14': IntegerType(),
        'fe_15': IntegerType(),
        'fe_16': IntegerType(),
        'fe_17': IntegerType(),
        'fe_18': IntegerType(),
        'fe_19': IntegerType(),
        'fe_20': IntegerType(),
        'Customer_ID': StringType(),
        'snapshot_date': DateType()         
    }
    
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Save Silver table - IRL connect to database to write 
    partition_name = f"SILVER_{table_name}_{snapshot_date_str.replace('-','_')}.parquet"
    filepath = output_dir + table_name + '/' + partition_name 
    # Creates the folder or the file directory if it doesn't exists, else continue 
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    df.write.mode('overwrite').parquet(filepath)
    print('Saved to:', filepath, '\n')
    
    return df 