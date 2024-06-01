import torch
import torch.nn as nn

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
            loss = loss_fn(pred, y) + 0.001 * torch.norm(model.linear_relu_stack[0].weight, p=2)

            test_loss += loss.item()
            num_total += y.shape[0]
            num_correct += (pred.argmax(1) == y).sum().item()

    test_loss /= len(test_loader)
    accuracy = num_correct / num_total
    return accuracy
