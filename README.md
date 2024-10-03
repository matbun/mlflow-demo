# MLFlow tracking server demo

The goal is to show how the MLFlow tracking server deployed on EGI resources can be used to store ML metadata and models, supporting both training and inference tasks.

To create a profile, add users to your experiment, or add users to your model, visit: https://mlflow.intertwin.fedcloud.eu/signup

General workflow:

1. Build the Python virtual environment and set your credentials for the MLFlow server in `train.py` and `inference.py` by setting `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`.
2. Run `train.py`: pushes ML training logs and best model to MLFLow tracking server. 
3. Register the trained model as `MNIST-classifier` on the MLFLow's models registry
4. Run `inference.py`: pulls best model (recently registered) and uses it to make predictions, saving them as a PNG image.

The `sync-runs` directory isprovides an example on how to generate the logs locally and *later* upload them to the remote MLFlow server.
This may be useful when you have not internet connection on compute nodes or for cherry-picking of runs.
