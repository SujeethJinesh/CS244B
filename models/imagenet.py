import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from filelock import FileLock

def get_data_loader():
    """Safely loads Tiny ImageNet data. Returns training/validation set dataloader."""
    tiny_imagenet_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.RandomCrop(64, padding=4) # Data augmentation
    ])

    data_dir="/Users/raycao/Documents/Stanford/cs244b/tiny-imagenet-200" # Replace the dir with your own path

    with FileLock(os.path.expanduser("~/imagenet_data.lock")):
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=tiny_imagenet_transforms)
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=tiny_imagenet_transforms)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True, num_workers=4
        ) 
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=False, num_workers=4
        )
    return train_loader, val_loader

def evaluate(model, val_loader):
    """Evaluates the accuracy and loss of the model on a validation dataset."""
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, target in val_loader:
            outputs = model(data)
            loss = loss_fn(outputs, target)  # Calculate loss
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)  # Get predictions
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100.0 * correct / total
    # avg_loss = total_loss / len(val_loader)
    return accuracy

class ImageConvNet(nn.Module):
    """Small ConvNet for Tiny ImageNet."""

    def __init__(self):
        super(ImageConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 200)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        return x

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

class AlexNet(nn.Module): 
    """AlexNet architecture (from scratch)."""

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # Conv Layer 1
            nn.ReLU(inplace=True),  # ReLU Activation 1
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), # Local Response Normalization
            nn.MaxPool2d(kernel_size=3, stride=2), # Max Pooling 1
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2), # Conv Layer 2
            nn.ReLU(inplace=True), # ReLU Activation 2
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2), # Local Response Normalization
            nn.MaxPool2d(kernel_size=3, stride=2), # Max Pooling 2

            nn.Conv2d(256, 384, kernel_size=3, padding=1), # Conv Layer 3
            nn.ReLU(inplace=True), # ReLU Activation 3

            nn.Conv2d(384, 384, kernel_size=3, padding=1), # Conv Layer 4
            nn.ReLU(inplace=True), # ReLU Activation 4

            nn.Conv2d(384, 256, kernel_size=3, padding=1), # Conv Layer 5
            nn.ReLU(inplace=True), # ReLU Activation 5
            nn.MaxPool2d(kernel_size=3, stride=2), # Max Pooling 3
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # Dropout 1
            nn.Linear(256 * 6 * 6, 4096), # Linear Layer 1 (Dense)
            nn.ReLU(inplace=True), # ReLU Activation 6
            nn.Dropout(p=0.5), # Dropout 2
            nn.Linear(4096, 4096), # Linear Layer 2 (Dense)
            nn.ReLU(inplace=True), # ReLU Activation 7
            nn.Linear(4096, num_classes), # Linear Layer 3 (Output)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def get_weights(self):
        """Gets the weights of the AlexNet model."""
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        """Sets the weights of the AlexNet model."""
        self.load_state_dict(weights, strict=False)

    def get_gradients(self):
        """Gets the gradients of the AlexNet model's parameters."""
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        """Sets the gradients of the AlexNet model's parameters."""
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)