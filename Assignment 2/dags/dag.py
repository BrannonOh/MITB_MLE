# Imports the main DAG class to define your workflow.
from airflow import DAG
# Imports the operator used to run bash shell commands as tasks.
from airflow.operators.bash import BashOperator
# Imports a placeholder operator that does nothing—used for structuring (e.g., start/end nodes).
from airflow.operators.dummy import DummyOperator
# Imports date/time utilities to define start times and intervals.
from datetime import datetime, timedelta

from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import BranchPythonOperator
import os 

default_args = {
    'owner': 'airflow', # Sets the task owner metadata to "airflow" (just for tracking/documentation).
    'depends_on_past': False, # Ensures each run of a task does not depend on the success of the previous run.
    'retries': 1, # If a task fails, Airflow will retry it once.
    'retry_delay': timedelta(minutes=5), # Wait 5 minutes before retrying the failed task.
}

with DAG(
    # Creates a new DAG with ID 'dag'. This is how it will appear in the Airflow UI.
    'dag', 
    
    # Applies the default arguments (owner, retries, etc.) you defined earlier to all tasks in the DAG.
    default_args=default_args, 
    
    # Adds a human-readable description for the DAG in the Airflow UI.
    description='Data pipeline run once a month ', 
    
    # At 00:00 on day-of-month 1. Sets the DAG to run monthly, at midnight on the 1st day of each month. This is in cron syntax.
    schedule_interval='0 0 1 * *',  
    
     # Airflow will create DAG runs from Jan 1, 2023 to Dec 1, 2024 (inclusive). No runs will be scheduled before or after this range.
    start_date=datetime(2023, 1, 1), 
    end_date=datetime(2024, 12, 1),

    # If the DAG is turned on later (after the start_date), Airflow will backfill and run all missed schedule intervals up to the current date.
    catchup=True,
    
#Assigns the DAG instance to a variable dag, used internally to attach tasks.
) as dag:
    
    # data pipeline

    # 🏷️ --- label store ---
        # --- Step 1: Dependency Check --- 
    # dep_check_souce_label_data = DummyOperator(task_id='dep_check_source_lms') 
    dep_check_source_label_data = FileSensor(
        task_id="dep_check_source_lms",
        filepath="lms_loan_daily.csv",
        fs_conn_id='fs_label_data',
        poke_interval=60,
        timeout=600,
        mode='poke'
    )
    
        # --- Step 2: Bronze Layer ---
    bronze_label_store = BashOperator(
        task_id='bronze_table_lms',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 bronze_label_store.py '
            '--snapshotdate "{{ ds }}"'  # This is a templated string in Airflow's BashOperator. It injects the DAG execution date (ds) into your shell command.
        ),
    )

        # --- Step 3: Silver Layer --- 
    silver_label_store = BashOperator(
        task_id='silver_table_lms',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 silver_label_store.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )
    
        # --- Step 4: Gold Label Store --- 
    gold_label_store = BashOperator(
        task_id='gold_label_store', 
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 gold_label_store.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )

        # --- Step 5: Completion Marker ---
    label_store_completed = DummyOperator(task_id='label_store_completed')

        # Dependencies (Defines task execution order for labels: check → bronze → silver → gold → complete.)
    dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed
 
    # ✨ --- features store ---
            # --- Step 1: Dependency Check --- 
    # dep_check_source_data_bronze_1 = DummyOperator(task_id="dep_check_source_data_attributes")
    # dep_check_source_data_bronze_2 = DummyOperator(task_id="dep_check_source_data_clickstream")
    # dep_check_source_data_bronze_3 = DummyOperator(task_id="dep_check_source_data_financials")
    dep_check_source_data_attributes = FileSensor(
        task_id="dep_check_source_data_attributes",
        filepath="features_attributes.csv",
        fs_conn_id='fs_label_data',
        poke_interval=60,
        timeout=600,
        mode='poke'
    )

    dep_check_source_data_clickstream = FileSensor(
        task_id="dep_check_source_data_clickstream",
        filepath="feature_clickstream.csv",
        fs_conn_id='fs_label_data',
        poke_interval=60,
        timeout=600,
        mode='poke'
    )

    dep_check_source_data_financials = FileSensor(
        task_id="dep_check_source_data_financials",
        filepath="features_financials.csv",
        fs_conn_id='fs_label_data',
        poke_interval=60,
        timeout=600,
        mode='poke'
    )

            # 🥉 --- Step 2: Bronze Layer ---
            # attributes 
    bronze_table_1 = BashOperator(
        task_id="bronze_table_attributes",
        bash_command=(
            'cd /opt/airflow/scripts && ' 
            'python3 bronze_attributes.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )
            # clickstream
    bronze_table_2 = BashOperator(
        task_id="bronze_table_clickstream",
        bash_command=(
            'cd /opt/airflow/scripts && ' 
            'python3 bronze_clickstream.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )    
            # financials 
    bronze_table_3 = BashOperator(
        task_id="bronze_table_financials",
        bash_command=(
            'cd /opt/airflow/scripts && ' 
            'python3 bronze_financials.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )        

            # 🥈 --- Step 2: Silver Layer ---
            # attributes 
    silver_table_1 = BashOperator(
        task_id="silver_table_attributes",
        bash_command=(
            'cd /opt/airflow/scripts && ' 
            'python3 silver_attributes.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )

            # clickstream 
    silver_table_2 = BashOperator(
        task_id="silver_table_clickstream",
        bash_command=(
            'cd /opt/airflow/scripts && ' 
            'python3 silver_clickstream.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )
            # financials 
    silver_table_3 = BashOperator(
        task_id="silver_table_financials",
        bash_command=(
            'cd /opt/airflow/scripts && ' 
            'python3 silver_financials.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )
            # 🥇 --- Step 1: Gold Layer ---
    gold_feature_store = BashOperator(
        task_id="gold_feature_store",
        bash_command=(
            'cd /opt/airflow/scripts && ' 
            'python3 gold_feature_store.py '
            '--snapshotdate "{{ ds }}"' 
        ),
    )

    feature_store_completed = DummyOperator(task_id="feature_store_completed")
    
    # Define task dependencies to run scripts sequentially
    dep_check_source_data_attributes >> bronze_table_1 >> silver_table_1 >> gold_feature_store
    dep_check_source_data_clickstream >> bronze_table_2 >> silver_table_2 >> gold_feature_store
    dep_check_source_data_financials >> bronze_table_3 >> silver_table_3 >> gold_feature_store
    gold_feature_store >> feature_store_completed
    
    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")
    
    model_xgb_inference = BashOperator(task_id='model_xgb_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname credit_model_2024_09_01.pkl'
        ),
    )

    model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_inference_start
    model_inference_start >> model_xgb_inference >> model_inference_completed
    

    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_xgb_monitor = BashOperator(task_id='model_xgb_monitor',
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 model_monitoring.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname credit_model_2024_09_01'
        ),
    )

    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    label_store_completed >> model_monitor_start
    model_monitor_start >> model_xgb_monitor >> model_monitor_completed


    # --- model auto training ---
    model_automl_start = DummyOperator(task_id='model_automl_start')

    def check_drift_flag(**context):
        snapshotdate = context['ds']
        modelname = 'credit_model_2024_09_01'
        drift_flag_dir = f"/opt/airflow/scripts/datamart/gold/model_monitoring/{modelname}/drift_flags"
        feature_flag = os.path.join(drift_flag_dir, f"{snapshotdate}_feature_drift.txt")
        perf_flag = os.path.join(drift_flag_dir, f"{snapshotdate}_performance_drift.txt")
        if os.path.exists(feature_flag) or os.path.exists(perf_flag):
            return "model_xgb_automl_retrain"
        else:
            return "skip_retrain"

    model_xgb_automl = BashOperator(
    task_id="model_xgb_automl_retrain",
    bash_command=(
        'cd /opt/airflow/scripts && '
        'python3 model_retrain.py '
        '--snapshotdate "{{ ds }}" '
        '--modelname credit_model_2024_09_01'
    ),
)
    
    branch_retrain = BranchPythonOperator(
        task_id='check_drift_flag',
        python_callable=check_drift_flag,
        provide_context=True
    )

    skip_retrain = DummyOperator(task_id="skip_retrain")

    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    # Task dependencies
    model_monitor_completed >> model_automl_start
    model_automl_start >> branch_retrain
    branch_retrain >> [model_xgb_automl, skip_retrain]
    model_xgb_automl >> model_automl_completed
    skip_retrain >> model_automl_completed

