import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize, ToTensor
from filelock import FileLock

IMAGE_SIZE=28

def fashion_mnist_get_data_loader():
    # batch_size = 64
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

class FashionMNISTConvNet(nn.Module):
    def __init__(self):
        super(FashionMNISTConvNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(IMAGE_SIZE * IMAGE_SIZE, 512),
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
