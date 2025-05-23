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

def process_silver_table(input_dir, table_name, output_dir, spark):

    # Connect to the Bronze table 
    partition_name = f'BRONZE_{table_name}.csv'
    filepath = input_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, '| Row count:', df.count())

    # df = df.withColumn('snapshot_date', F.to_date(col('snapshot_date').cast('string'), 'd/M/yyyy'))

    # Convert 'loan_start_date' to DateType() first before casting it as DateType() below as Spark treats it as a StringType(). 
    df = df.withColumn('snapshot_date', F.to_date('snapshot_date', 'd/M/yyyy'))

    df = df.drop('Name') # Sensitive personal data (PII)
    
    # For feature 'Age': 
    # Remove any non-digit characters 
    df = df.withColumn('Age', F.regexp_replace(col('Age'), '_', ''))
    df = df.withColumn('Age', col('Age').cast(IntegerType()))
    df = df.withColumn('Age', F.when((col('Age') >= 0) & (col('Age') <= 100), col('Age')).otherwise(F.lit(None)))
    # Impute missing values with 'median' age 
    median_age = df.approxQuantile("Age", [0.5], 0.01)[0] 
    df = df.withColumn("Age", F.when(col("Age").isNull(), median_age).otherwise(col("Age")))

    # # For feature 'SSN': 
    # # Make sure it is in the format ^\d{3}-\d{2}-\d{4}$, else replace with NULL. 
    # df = df.withColumn('SSN_cleaned', F.when(col('SSN').rlike(r'^\d{3}-\d{2}-\d{4}$'), col('SSN')).otherwise(F.lit(None)))
    df = df.drop('SSN') # Sensitive personal data (PII)
                                            
    # For feature 'Occupation':
    # Remove those '_______' with NULL. 
    df = df.withColumn('Occupation', F.when(col('Occupation').rlike(r'_______'), F.lit(None)).otherwise(col('Occupation')))
    df = df.fillna({'Occupation': 'Unknown'})

    # Clean data: enforce schema / data type 
    # Dictionary specifying columns and their desired datatypes 
    column_type_map = {
        'Customer_ID': StringType(),
        'Age': IntegerType(),
        'Occupation': StringType(),
        'snapshot_date': DateType()       
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Save Silver table - IRL connect to database to write 
    partition_name = f"SILVER_{table_name}.parquet"
    filepath = output_dir + table_name + '/' + partition_name 
    # Creates the folder or the file directory if it doesn't exists, else continue 
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    df.write.mode('overwrite').parquet(filepath)
    print('Saved to:', filepath, '\n')
    
    return df 