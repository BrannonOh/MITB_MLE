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

    # CONNECT TO THE BRONZE TABLE
    partition_name = f'BRONZE_{table_name}.csv'
    filepath = input_dir + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('Loaded from:', filepath, '| Row count:', df.count())
    
    # DATA CLEANING BEFORE CASTING DESIRED DATATYPES 
    # df = df.withColumn('Customer_ID', F.trim(col('Customer_ID'))) # Strip spaces
    # duplicate_ids = df.groupBy('Customer_ID').count().filter('count > 1')
    # duplicate_ids.show() # Check for duplicates 

    df = df.withColumn('Annual_Income', F.when(col('Annual_Income').isin('NA', 'nan', ''), 0)
                       .otherwise(F.regexp_replace('Annual_Income', '_', '')))
    
    df = df.withColumn('Num_Bank_Accounts', F.when(col('Num_Bank_Accounts') >= 0, col('Num_Bank_Accounts'))
                       .otherwise(F.lit(None))) # Replace -1 with None (or Null), assuming null.
    df = df.filter(col("Num_Bank_Accounts") <= 10) 
    df = df.filter(col('Num_Bank_Accounts').isNotNull()) 

    df = df.withColumn('Num_Credit_Card', F.when(col('Num_Credit_Card').cast('string').rlike(r'^\d+$'), col('Num_Credit_Card')).otherwise(0)) # Safeguard even though manual check didn't show typos
    df = df.withColumn('Num_Credit_Card', col('Num_Credit_Card').cast('int'))
    df = df.filter(col("Num_Credit_Card") <= 20) # Drop instead as there are only 284 rows that are > 20. 

    # df = df.withColumn('Interest_Rate', col('Interest_Rate').cast('int'))
    # df = df.filter(col('Interest_Rate') <= 100) # Drop instead as there are only 254 rows that are > 100. 
    df = df.withColumn('Interest_Rate', col('Interest_Rate').cast(DoubleType()))

    df = df.withColumn('Num_of_Loan', F.regexp_replace(col('Num_of_Loan'), '_', ''))
    df = df.withColumn('Num_of_Loan', F.when(col('Num_of_Loan').rlike(r'^\d+$'), col('Num_of_Loan'))
                       .otherwise(F.lit(None))) # Replace -100 or those without a proper value as null. 
    df = df.withColumn("Num_of_Loan", col("Num_of_Loan").cast("int"))
    df = df.withColumn("Num_of_Loan", F.when((col("Num_of_Loan") >= 0) & (col("Num_of_Loan") <= 50), col("Num_of_Loan")).otherwise(None))
    df = df.filter(col('Num_of_Loan').isNotNull()) # Drop missing values, 478 rows. Less than 5% of the whole dataset. 

    df = df.withColumn('Type_of_Loan', F.when(col('Type_of_Loan').isNull() | (col('Type_of_Loan') == ''), 'Unknown').otherwise(col('Type_of_Loan')))

    df = df.withColumn('Delay_from_due_date', F.when(col('Delay_from_due_date') >= 0, col('Delay_from_due_date'))
                       .otherwise(F.lit(None))) # Treat -ve values or other values not +ve numeric as null. 
    df = df.filter(col('Delay_from_due_date').isNotNull()) # Drop missing values, 83 rows.

    df = df.withColumn('Num_of_Delayed_Payment', F.regexp_replace(col('Num_of_Delayed_Payment'), '_', ''))
    df = df.withColumn('Num_of_Delayed_Payment', F.when(col('Num_of_Delayed_Payment').rlike(r'^\d+$'), col('Num_of_Delayed_Payment')).otherwise(F.lit(None))) # Treat -ve values or other values not +ve numeric as null. 
    df = df.withColumn('Num_of_Delayed_Payment', col('Num_of_Delayed_Payment').cast('int'))
    df = df.withColumn("Num_of_Delayed_Payment", 
                       F.when((col("Num_of_Delayed_Payment") >= 0) & (col("Num_of_Delayed_Payment") <= 100),
                              col("Num_of_Delayed_Payment"))
                       .otherwise(F.lit(None)))
    df = df.filter(col('Num_of_Delayed_Payment').isNotNull()) # Drop missing values. Less than 1% of the whole dataset. 
    
    df = df.withColumn('Changed_Credit_Limit', F.when(col('Changed_Credit_Limit').rlike(r'^-?\d+(\.\d+)?$'), col('Changed_Credit_Limit')).otherwise(F.lit(None))) # Only accept +ve and -ve float numbers. 
    df = df.filter(col('Changed_Credit_Limit').isNotNull()) # Drop missing values, 224 rows.

    df = df.withColumn('Num_Credit_Inquiries', F.when(col('Num_Credit_Inquiries') >= 0, col('Num_Credit_Inquiries')).otherwise(0)) # Only +ve numbers. If -ve numbers, most likely no activity so impute it with 0. 
    df = df.withColumn("Num_Credit_Inquiries", col("Num_Credit_Inquiries").cast("int"))
    df = df.withColumn("Num_Credit_Inquiries",
                       F.when((col("Num_Credit_Inquiries") >= 0) & 
                              (col("Num_Credit_Inquiries") <= 50), col("Num_Credit_Inquiries"))
                       .otherwise(None))

    df = df.withColumn('Credit_Mix', F.when(col('Credit_Mix') == '_', 'Unknown') # Replace it 'Unknown'  
                       .otherwise(col('Credit_Mix')))
    
    df = df.withColumn('Outstanding_Debt', F.regexp_replace(col('Outstanding_Debt'), '_', '')) 

    df = df.withColumn('Payment_of_Min_Amount', F.when(col('Payment_of_Min_Amount') == 'NM', 'Unknown')
                       .otherwise(col('Payment_of_Min_Amount')))

    df = df.withColumn('Total_EMI_per_month', F.when(col('Total_EMI_per_month') >= 0, col('Total_EMI_per_month'))
                       .otherwise(F.lit(None))) 

    df = df.withColumn('Amount_invested_monthly', F.when(col('Amount_invested_monthly') == '__10000__', 0).otherwise(col('Amount_invested_monthly'))) # Treat missing values as 0.

    df = df.withColumn('Payment_Behaviour', F.when(col('Payment_Behaviour') == '!@9#%8', 'Unknown').otherwise(col('Payment_Behaviour')))

    df = df.withColumn('Monthly_Balance', col("Monthly_Balance").cast(StringType()))
    df = df.withColumn('Monthly_Balance', F.regexp_replace(col('Monthly_Balance'), '_', '')) # Remove any underscores, if any. 
    df = df.withColumn('Monthly_Balance', F.when(col('Monthly_Balance').cast('float') < 0, 0).otherwise(col('Monthly_Balance'))) # If it's < 0, change it to 0. 
    df = df.withColumn('Monthly_Balance', F.when(col('Monthly_Balance').isNull(), 0)
                       .otherwise(col('Monthly_Balance'))) # Impute 0 if null. 

    # CLEAN DATA: ENFORCE SCHEMA / DATA TYPE
    # Dictionary specifying columns and their desired datatypes 
    column_type_map = {
        'Customer_ID': StringType(),
        'Annual_Income': FloatType(),
        'Monthly_Inhand_Salary': FloatType(),
        'Num_Bank_Accounts': IntegerType(),
        'Num_Credit_Card': IntegerType(),
        'Interest_Rate': DoubleType(),
        'Num_of_Loan': IntegerType(),
        'Type_of_Loan': StringType(),
        'Delay_from_due_date': IntegerType(),
        'Num_of_Delayed_Payment': IntegerType(),
        'Changed_Credit_Limit': FloatType(),
        'Num_Credit_Inquiries': IntegerType(),
        'Credit_Mix': StringType(), 
        'Outstanding_Debt': FloatType(),
        'Credit_Utilization_Ratio': DoubleType(),
        'Credit_History_Age': StringType(),
        'Payment_of_Min_Amount': StringType(),
        'Total_EMI_per_month': DoubleType(),
        'Amount_invested_monthly': DoubleType(),
        'Payment_Behaviour': StringType(),
        'Monthly_Balance': DoubleType(),
        'snapshot_date': DateType()         
    }
    
    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # DATA AUGMENTATION 
    # Augment data: Convert from text to numeric months 
    # Extract years and months 
    df = df.withColumn('years', F.regexp_extract(col('Credit_History_Age'), r'(\d+)\s*Years', 1).cast('int'))
    df = df.withColumn('months', F.regexp_extract(col('Credit_History_Age'), r'(\d+)\s*Months', 1).cast('int'))
    # Compute total months 
    df = df.withColumn('Credit_History_Age_In_Months', (col('years')*12 + col('months')).cast('int'))
    # Drop 'years' and 'months' columns 
    df = df.drop('years', 'months', 'Credit_History_Age')

    # Augment data: Calculate Investment Ratio 
    df = df.withColumn('Investment Ratio', F.when((col('Monthly_Inhand_Salary').isNull()) | (col('Monthly_Inhand_Salary') == 0), 0.0).otherwise((col('Amount_invested_monthly') / col('Monthly_Inhand_Salary')).cast('float')))

    # Save Silver table - IRL connect to database to write 
    partition_name = f"SILVER_{table_name}.parquet"
    filepath = output_dir + table_name + '/' + partition_name 
    # Creates the folder or the file directory if it doesn't exists, else continue 
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    df.write.mode('overwrite').parquet(filepath)
    print('Saved to:', filepath, '\n')
    
    return df 