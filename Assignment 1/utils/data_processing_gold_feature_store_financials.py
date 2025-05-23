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

    # Feature Engineering 
    # Feature 'Credit_Mix': One-hot-encoding 
    # Step 1: StringIndexer to convert categories to numeric indices
    credit_mix_indexer = StringIndexer(inputCol='Credit_Mix', outputCol='Credit_Mix_index', handleInvalid='keep')
    # Step 2: OneHotEncoder to convert index to one-hot vector
    credit_mix_encoder = OneHotEncoder(inputCol='Credit_Mix_index', outputCol='Credit_Mix_ohe')
    # # Step 3: Build and apply pipeline
    # pipeline = Pipeline(stages=[indexer, encoder])
    # df = pipeline.fit(df).transform(df)

    # Feature 'Payment_of_Min_Amount'
    # Define indexer and encoder
    payment_indexer = StringIndexer(inputCol='Payment_of_Min_Amount', outputCol='Payment_of_Min_Amount_index', handleInvalid='keep')
    payment_encoder = OneHotEncoder(inputCol='Payment_of_Min_Amount_index', outputCol='Payment_of_Min_Amount_ohe')

    # Feature 'Payment_Behaviour'
    # Define indexer and encoder
    payment_behavior_indexer = StringIndexer(inputCol='Payment_Behaviour', outputCol='Payment_Behaviour_index', handleInvalid='keep')
    payment_behavior_encoder = OneHotEncoder(inputCol='Payment_Behaviour_index', outputCol='Payment_Behaviour_ohe')
    
    # Build pipeline
    pipeline = Pipeline(stages=[credit_mix_indexer, credit_mix_encoder, payment_indexer, payment_encoder, 
                               payment_behavior_indexer, payment_behavior_encoder])
    df = pipeline.fit(df).transform(df)

    # Clean data: enforce schema / data type 
    # Dictionary specifying columns and their desired datatypes 
    column_type_map = {
        'Customer_ID': StringType(),
        'Annual_Income': FloatType(),
        'Monthly_Inhand_Salary': FloatType(),        
        'Num_Bank_Accounts': IntegerType(),
        'Num_Credit_Card': IntegerType(),
        'Interest_Rate': IntegerType(),
        'Num_of_Loan': IntegerType(),
        'Delay_from_due_date': IntegerType(),
        'Num_of_Delayed_Payment': IntegerType(),
        'Changed_Credit_Limit': FloatType(),
        'Num_Credit_Inquiries': IntegerType(),         
        'Outstanding_Debt': FloatType(),
        'Credit_Utilization_Ratio': DoubleType(),
        'Total_EMI_per_month': DoubleType(),
        'Amount_invested_monthly': DoubleType(),
        'Monthly_Balance': DoubleType(),
        'Credit_History_Age_In_Months': IntegerType(),
        'Investment Ratio': DoubleType()
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))
    
    # Select columns to save
    df = df.select('Customer_ID', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 
                   'Interest_Rate', 'Num_of_Loan','Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit',
                  'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
                  'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_In_Months', 'Investment Ratio',
                  'Credit_Mix_ohe', 'Payment_of_Min_Amount_ohe', 'Payment_Behaviour_ohe')

    return df
