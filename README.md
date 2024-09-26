Some Copilot-generated scripts adapted to work with EGI's MLFlow instance.

Workflow:

1. Build the environment and set your credentials for the MLFlow server in `train.py` and `inference.py`
2. Run `train.py`: pushes ML training logs and best model to MLFLow tracking server. 
3. Register the trained model as `MNIST-classifier` on the MLFLow's models registry
4. Run `inference.py`: pulls best model (recently registered) and uses it to make predictions, saving them as a PNG image.
