#!/bin/bash

# Export a specific run to a local folder
export MLFLOW_TRACKING_URI="file://${PWD}/mlruns"
export-run --output-dir my-export --run-id $RUNID
