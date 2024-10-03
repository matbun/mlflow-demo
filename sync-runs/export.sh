#!/bin/bash

# Export a specific local run to a local folder in some intermediate "format".
# However, consider copying directly the run to the remote server using copy.sh instead.

export MLFLOW_TRACKING_URI="file://${PWD}/mlruns"
export-run --output-dir my-export --run-id $RUNID
