#!/bin/bash

# Upload a specific run, which was previously exported under ./my-export, to the remote server.
# However, consider copying directly the run to the remote server using copy.sh instead of doing export.sh + import.sh.

export MLFLOW_TRACKING_INSECURE_TLS='true'
export MLFLOW_TRACKING_USERNAME='matteo.bunino@cern.ch'
export MLFLOW_TRACKING_PASSWORD='YOUR_PWD'
export MLFLOW_TRACKING_URI='https://mlflow.intertwin.fedcloud.eu/'

import-run --experiment-name 'MNIST_PyTorch_Experiment' --input-dir my-export/
