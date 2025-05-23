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
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

def process_gold_features(table_name, input_dir, output_dir, spark):

    # Connect to silver table
    partition_name = f'SILVER_{table_name}.parquet'
    filepath = input_dir + partition_name
    df = spark.read.parquet(filepath)
    print('Loaded from:', filepath, '| Row count:', df.count())

    # Data Preprocessing 
    # Feature: Occupation 
    # One-hot Encoding 
    indexer = StringIndexer(inputCol='Occupation', outputCol='Occupation_index', handleInvalid='keep')
    encoder = OneHotEncoder(inputCol='Occupation_index', outputCol='Occupation_ohe')
    pipeline = Pipeline(stages=[indexer, encoder])
    df = pipeline.fit(df).transform(df)

    # Clean data: enforce schema / data type 
    # Dictionary specifying columns and their desired datatypes 
    column_type_map = {
        'Customer_ID': StringType(),
        'Age': IntegerType(),
        'snapshot_date': DateType()       
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # Select columns to save
    df = df.select('Customer_ID', 'Age', 'Occupation_ohe')
    
    return df