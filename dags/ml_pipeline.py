import airflow
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import timedelta
from airflow.utils.dates import days_ago
import os

PROJECT_DIR=r"/home/siewe/Documents/France/Efrei/Cours/M2/'Semestre 9'/'Applications of Big Data'/app-bd"


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
	dag_id = "ML_pipeline",
	default_args=default_args,
	schedule_interval='*/30 * * * *',
	dagrun_timeout=timedelta(minutes=60),
	description='ML Pipeline',
	start_date = airflow.utils.dates.days_ago(1),
    catchup=False
)

#python_task = PythonOperator(task_id='python_task',python_callable=my_func, dag=dag_python)



data_preparation = BashOperator(
    task_id="data_preparation",
    bash_command=f"python {PROJECT_DIR}/src/features/build_features.py",
    dag=dag_python
)

#feature_engineering = BashOperator(
#    task_id="feature_engineering",
#    bash_command="python /app/twitter_scraper/twitter_scraper.py",
#    dag=dag_python
#)

training = BashOperator(
    task_id="training",
    bash_command=f"python {PROJECT_DIR}/src/models/train_model.py",
    dag=dag_python
)

data_preparation >> training


if __name__ == "__main__":
    dag_python.cli()