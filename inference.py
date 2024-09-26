import torch
import mlflow
import mlflow.pyfunc
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'xxx'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'xxx'
TRACKING_URI = 'https://mlflow.intertwin.fedcloud.eu/'

# Step 0: Register the pre-trained model into the models registry and give it "production" tag

# Step 1: Load the model from MLflow's model registry using version or tags
def load_model_from_registry(model_name, version=None, tag=None):
    if version:
        # Load by specific version
        model_uri = f"models:/{model_name}/{version}"
    elif tag:
        # Load by tag (if the tag is set in the UI)
        model_uri = f"models:/{model_name}/{tag}"
    else:
        raise ValueError("Either version or tag must be specified for model loading.")

    model = mlflow.pytorch.load_model(model_uri)
    return model

# Step 2: Sample and preprocess the MNIST test dataset
def load_mnist_data(num_samples=5):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    samples = [test_dataset[i] for i in indices]
    images, labels = zip(*samples)
    return images, labels

# Step 3: Plot the MNIST images with predicted labels
def plot_images_with_predictions(images, labels, predictions):
    fig, axes = plt.subplots(1, len(images), figsize=(10, 2))
    for i, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(f"Pred: {prediction}, True: {label}")
        axes[i].axis("off")
    plt.savefig('inference_result.png')

# Step 4: Perform inference and plot results
def infer_and_plot(model_name, num_samples=5, version=None, tag=None):
    # Load the model from the model registry
    model = load_model_from_registry(model_name, version, tag)

    # Load sample MNIST images
    images, labels = load_mnist_data(num_samples)

    # Convert images to tensors
    image_tensors = torch.stack(images)

    # Perform inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensors)
        _, predicted = torch.max(outputs, 1)

    # Plot the images with predicted labels
    plot_images_with_predictions(images, labels, predicted)

if __name__ == "__main__":
    mlflow.set_tracking_uri(TRACKING_URI)
    # Infer and plot with the model registered under "MNIST-classifier" in the registry
    model_name = "MNIST-classifier"

    # Use either a version or tag to load the model
    # Option 1: Load by version number (e.g., version 1)
    model_version = 1

    # Option 2: Load by tag (if you have set tags like "production" in the UI)
    model_tag = "production"  # Adjust this based on your tag

    # Choose one of the options
    infer_and_plot(model_name, num_samples=5, version=model_version, tag=model_tag)
    # infer_and_plot(model_name, num_samples=5, tag=model_tag)
