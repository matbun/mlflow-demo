#!/bin/bash

# Copy a local run under ./mlruns to the remote MLFlow server on EGI cloud

export MLFLOW_TRACKING_INSECURE_TLS='true'
export MLFLOW_TRACKING_USERNAME='matteo.bunino@cern.ch'
export MLFLOW_TRACKING_PASSWORD='YOUR_PWD'
export MLFLOW_TRACKING_URI='https://mlflow.intertwin.fedcloud.eu/'

copy-run --run-id $RUNID --experiment-name "MNIST_PyTorch_Experiment" --src-mlflow-uri mlruns --dst-mlflow-uri https://mlflow.intertwin.fedcloud.eu/
