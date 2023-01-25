import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago
import os

PROJECT_DIR=r"/home/siewe/Documents/France/Efrei/Cours/M2/'Semestre 9'/'Applications of Big Data'/app-bd"
MLFLOW_TRACKING_URI="http://localhost:5000"

default_args = {
    'owner': 'siewe',    
    #'start_date': airflow.utils.dates.days_ago(2),
    # 'end_date': datetime(),
    # 'depends_on_past': False,
    #'email': ['airflow@example.com'],
    #'email_on_failure': False,
    #'email_on_retry': False,
    # If a task fails, retry it once after waiting
    # at least 5 minutes
    #'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag_python = DAG(
	dag_id = "training_pipeline",
	default_args=default_args,
	schedule_interval='*/30 * * * *',
	dagrun_timeout=timedelta(minutes=60),
	description='training Pipeline',
	start_date = airflow.utils.dates.days_ago(1),
    catchup=False
)

pipeline_creation = BashOperator(
    task_id="pipeline_creationv",
    bash_command=f"""
        export MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}; 
        mlflow run {PROJECT_DIR} -e create_pipeline
    """,
    dag=dag_python
)

data_preparation = BashOperator(
    task_id="data_preparation",
    bash_command=f"""
        export MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}; 
        mlflow run {PROJECT_DIR} -e data_prep
    """,
    dag=dag_python
)


training = BashOperator(
    task_id="training",
    bash_command=f"""
        export MLFLOW_TRACKING_URI={MLFLOW_TRACKING_URI}; 
        mlflow run {PROJECT_DIR}
    """,
    dag=dag_python
)

pipeline_creation >> data_preparation >> training


if __name__ == "__main__":
    dag_python.cli()