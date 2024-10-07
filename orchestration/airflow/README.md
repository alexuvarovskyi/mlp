# Installing Airflow

```bash
export AIRFLOW_HOME=$PWD/airflow
AIRFLOW_VERSION=2.10.2

PYTHON_VERSION="$(python --version | cut -d " " -f 2 | cut -d "." -f 1-2)"
CONSTRAINT_URL="https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
pip install "apache-airflow==${AIRFLOW_VERSION}" --constraint "${CONSTRAINT_URL}"
airflow standalone

pip install -r requirements.txt
```

Open ui at:
http://0.0.0.0:8080


To trigger train pipeline
Find in iu `train_dag` and run it with green arrow button


To trigger inferene pipeline
Find in iu `inference_dag` and run it with green arrow button
