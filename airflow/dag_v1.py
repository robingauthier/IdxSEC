from datetime import datetime, timedelta
from airflow import DAG
import time
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from pprint import pprint

# standalone | Starting Airflow Standalone
# standalone | Checking database is initialized
# INFO  [alembic.runtime.migration] Context impl SQLiteImpl.
# INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
# WARNI [unusual_prefix_08a146ac3e25b59a88555e217db3dc8f203f3211_example_kubernetes_executor] The example_kubernetes_executor example DAG requires the kubernetes provider. Please install it with: pip install apache-airflow[cncf.kubernetes]
# WARNI [unusual_prefix_90be04794c26a7e763e9fa7ddcc42db26bc094e8_example_python_operator] The virtalenv_python example task requires virtualenv, please install it.
# WARNI [unusual_prefix_c64d26ee53b6e18cd9441eed85d3fbeec730aa80_example_local_kubernetes_executor] Could not import DAGs in example_local_kubernetes_executor.py
# Traceback (most recent call last):
#   File "/home/sachadrevet.linux/anaconda3/lib/python3.11/site-packages/airflow/example_dags/example_local_kubernetes_executor.py", line 37, in <module>
#     from kubernetes.client import models as k8s
# ModuleNotFoundError: No module named 'kubernetes'
# WARNI [unusual_prefix_c64d26ee53b6e18cd9441eed85d3fbeec730aa80_example_local_kubernetes_executor] Install Kubernetes dependencies with: pip install apache-airflow[cncf.kubernetes]
# WARNI [unusual_prefix_7d388b871a8b03b6bf997b22333e46ec93d99300_tutorial_taskflow_api_virtualenv] The tutorial_taskflow_api_virtualenv example DAG requires virtualenv, please install it.
# pip install apache-airflow-providers-cncf-kubernetes

def print_hello():
    return "Hello world!"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2019, 4, 30),
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

dag = DAG(
    "hello_world",
    description="Simple tutorial DAG",
    schedule_interval="0 12 * * *",
    default_args=default_args,
    catchup=False,
)

t1 = DummyOperator(task_id="dummy_task", retries=3, dag=dag)

t2 = PythonOperator(task_id="hello_task", python_callable=print_hello, dag=dag)

def print_context(ds, **kwargs):
    pprint(kwargs)
    time.sleep(100)
    print(ds)
    return 'Whatever you return gets printed in the logs'


run_this = PythonOperator(
    task_id='print_the_context',
    provide_context=True,
    python_callable=print_context,
    dag=dag,
)

# sets downstream foe t1
t1 >> t2
