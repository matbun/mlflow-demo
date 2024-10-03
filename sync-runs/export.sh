#!/bin/bash

# Export a specific local run under ./mlruns to a local folder called ./my-export using some intermediate "format".
# However, consider copying directly the run to the remote server using copy.sh instead of doing export.sh + import.sh.

export MLFLOW_TRACKING_URI="file://${PWD}/mlruns"
export-run --output-dir my-export --run-id $RUNID
