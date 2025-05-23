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
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType

def process_silver_table(snapshot_date_str, input_dir, table_name, output_dir, spark):

    # Connect to the Bronze table 
    partition_name = f'BRONZE_{table_name}_{snapshot_date_str.replace('-','_')}.csv'
    filepath = input_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, '| Row count:', df.count())

    # Convert 'loan_start_date' to DateType() first before casting it as DateType() below as Spark treats it as a StringType(). 
    df = df.withColumn('loan_start_date', F.to_date('loan_start_date', 'd/M/yyyy'))
    
    # Clean data: enforce schema / data type 
    # Dictionary specifying columns and their desired datatypes 
    column_type_map = {
        "loan_id": StringType(),
        "Customer_ID": StringType(),
        "loan_start_date": DateType(),
        "tenure": IntegerType(),
        "installment_num": IntegerType(),
        "loan_amt": FloatType(),
        "due_amt": FloatType(),
        "paid_amt": FloatType(),
        "overdue_amt": FloatType(),
        "balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # Augment data: Add month on book (How many months the loan has been active)
    df = df.withColumn('mob', col('installment_num').cast(IntegerType()))

    # Augment data: Add installments_missed, first_missed_date and days past due (dpd)
    # Estimates how many monthly payments the customer has missed.
    df = df.withColumn('installments_missed', F.ceil(col('overdue_amt') / col('due_amt')).cast(IntegerType())).fillna(0)
    # Approximates when the customer first missed a payment.
    df = df.withColumn("first_missed_date", F.when(col("installments_missed") > 0, F.add_months(col("snapshot_date"), -1 * col("installments_missed"))).cast(DateType()))
    # How long (in days) the customer has been overdue.
    df = df.withColumn("dpd", F.when(col("overdue_amt") > 0.0, F.datediff(col("snapshot_date"), col("first_missed_date"))).otherwise(0).cast(IntegerType()))
   
    # Save Silver table - IRL connect to database to write 
    partition_name = f'SILVER_{table_name}_{snapshot_date_str.replace('-','_')}.parquet'
    filepath = output_dir + table_name + '/' + partition_name 
    # Creates the folder or the file directory if it doesn't exists, else continue 
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    df.write.mode('overwrite').parquet(filepath)
    print('Saved to:', filepath, '\n')
    
    return df 