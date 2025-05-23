# Import all the packages

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

def process_bronze_table(input_dir, table_name, output_dir, spark):
   
    # Points to your raw source data - Simulates a real source system (like a batch job export from a banking database. Loads the full CSV and filters only the rows where snapshot_date matches the one you want - Simulates extracting data for a single monthly snapshot. 
    df = spark.read.csv(input_dir, header=True, inferSchema=True)
    df = df.withColumn('snapshot_date', F.to_date('snapshot_date', 'd/M/yyyy'))
   
    # Logs how many rows are found for that month and year - It is useful for debugging data volume or missing partitions 
    print('Loaded from:', input_dir, '| Row count:', df.count())
   
    # Save each partition for every month of each year
    partition_name = f"BRONZE_{table_name}.csv"
    filepath = output_dir + table_name + '/' + partition_name 

    # Creates the folder if it doesn't exist, else continue
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert Spark DataFrame to Pandas -> saves as .csv 
    df.toPandas().to_csv(filepath, index=False)
    print('Saved to:', filepath, '\n')
    
    return df 