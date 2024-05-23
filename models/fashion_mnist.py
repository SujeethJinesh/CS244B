import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from filelock import FileLock

def get_data_loader():
    batch_size = 32
    # Transform to normalize the input images
    transform = transforms.Compose([ToTensor(), Normalize((0.5,), (0.5,))])

    with FileLock(os.path.expanduser("~/data.lock")):
        # Download training data from open datasets
        training_data = datasets.FashionMNIST(
            root="~/data",
            train=True,
            download=True,
            transform=transform,
        )

        # Download test data from open datasets
        test_data = datasets.FashionMNIST(
            root="~/data",
            train=False,
            download=True,
            transform=transform,
        )

    # Create data loaders
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    return train_dataloader, test_dataloader


def evaluate(model, test_loader):
    """Evaluates the accuracy of the model on a validation dataset."""
    model.eval()
    correct = 0
    total = 0
    test_loss, num_correct, num_total = 0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_loader):
            pred = model(X)
            loss = loss_fn(pred, y)

            test_loss += loss.item()
            num_total += y.shape[0]
            num_correct += (pred.argmax(1) == y).sum().item()

    test_loss /= len(test_loader)
    accuracy = num_correct / num_total
    return accuracy

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)
