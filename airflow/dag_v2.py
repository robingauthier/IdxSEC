import datetime
import pendulum
import os
import pandas as pd
from airflow.decorators import dag, task
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator

# Airflow seems to have issues with pythonpath
import sys
sys.path.append('/Users/sachadrevet/src/')

from IdxSEC.edgar_dwnd_v1 import main as edgar_download_main
from IdxSEC.extract_table_content_main import extract_table_content
from IdxSEC import g_edgar_folder


# ls -lhtr  /Users/sachadrevet/src/IdxSEC/data | tail -n 20


@task(multiple_outputs=True)
def edgar_download_task():
    lnfiles=edgar_download_main(redownload=True,fromdt=pd.to_datetime('2024-01-20'))
    return {'fname':lnfiles}


@task()
def find_content_table(d:dict):
    # (Pdb) p d
    # {'fname': ['/Users/sachadrevet/src/IdxSEC/data/formA14A_cik1750153_asof20240118_0001140361-24-002758.txt',
    # '/Users/sachadrevet/src/IdxSEC/data/form14A_cik1750153_asof20240118_0001140361-24-002743.txt',
    # '/Users/sachadrevet/src/IdxSEC/data/form14A_cik1537140_asof20240118_0001580642-24-000296.txt']}
    if 'fname' not in d.keys():
        raise(ValueError('missing fname key'))
    for fname in d['fname']:
        extract_table_content(os.path.basename(fname))

@dag(
    schedule='*/10 * * * *',# every 10 minutes
    start_date=datetime.datetime(2024, 1, 1),
    max_active_runs=1,
    catchup=False,
    tags=["data","sec"],
)
def form14():
    print('ready')
    lf = edgar_download_task()
    # lf :XComArg(<Task(_PythonDecoratedOperator): edgar_download_task>)

    # Waits for a file or folder to land in a filesystem.
    tfname0 = os.path.join(*[g_edgar_folder, 'edgar_form_list.pkl'])
    t6 = FileSensor(task_id="wait_for_master_file", filepath=tfname0)

    # Then we need to do the processing of those files
    find_content_table(lf)

# if you do not instanciate it here it will not be recognized
form14()






with DAG(
    "test_form14",
    description="Simple tutorial DAG",
    schedule_interval='*/10 * * * *',
    start_date=datetime.datetime(2024,1,1),
    catchup=False,
) as dag:
    tfname0 = os.path.join(*[g_edgar_folder, 'edgar_form_list.pkl'])
    t6 = FileSensor(task_id="wait_for_master_file", filepath=tfname0)


def test_dag_starts_on_correct_date():
    mydag = form14()
    #import pdb;pdb.set_trace()
    assert mydag.start_date == pendulum.datetime(2022, 2, 8, tz="Asia/Singapore")


# ipython -i -m IdxSEC.airflow.dag_v2
if __name__=='__main__':
    test_dag_starts_on_correct_date()