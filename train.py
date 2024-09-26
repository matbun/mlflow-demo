import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import os

os.environ['MLFLOW_TRACKING_INSECURE_TLS'] = 'true'
os.environ['MLFLOW_TRACKING_USERNAME'] = 'xxx'
os.environ['MLFLOW_TRACKING_PASSWORD'] = 'xxx'
TRACKING_URI = 'https://mlflow.intertwin.fedcloud.eu/'

# Step 1: Set up MLflow
mlflow.set_tracking_uri(TRACKING_URI)  # Change this if needed
experiment_name = "MNIST_PyTorch_Experiment"
mlflow.set_experiment(experiment_name)

# Step 2: Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 3: Prepare the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Step 4: Train the model and log to MLflow
def train_model():
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 5
    mlflow.pytorch.autolog()  # Automatically log params, metrics, and model

    with mlflow.start_run() as run:
        for epoch in range(n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            avg_loss = running_loss / len(train_loader)

            print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

            # Log the loss and accuracy
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("accuracy", accuracy, step=epoch)

        # Save the trained model checkpoint
        model_checkpoint = "mnist_model.pth"
        torch.save(model.state_dict(), model_checkpoint)
        mlflow.log_artifact(model_checkpoint, "model_checkpoint")

        # Log the trained model to MLflow for deployment
        mlflow.pytorch.log_model(model, "MNIST-classifier")

if __name__ == "__main__":
    train_model()
