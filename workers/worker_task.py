import torch.nn.functional as F
from models.imagenet_model import ConvNet, get_data_loader
import ray

@ray.remote
def compute_gradients(weights):
    model = ConvNet()
    data_iterator = iter(get_data_loader()[0])

    model.set_weights(weights)
    try:
        data, target = next(data_iterator)
    except StopIteration:  # When the epoch ends, start a new epoch.
        data_iterator = iter(get_data_loader()[0])
        data, target = next(data_iterator)
    model.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    return model.get_gradients()
