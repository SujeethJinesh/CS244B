import torch
import numpy as np
from models.test_model import ConvNet
from workers.worker_task import compute_gradients
from models.test_model import ConvNet, get_data_loader, evaluate
import ray

iterations = 200
num_workers = 2

@ray.remote
class ParameterServer(object):
    def __init__(self, lr):
        self.model = ConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    # def apply_gradients(self, *gradients):
    #     print(f"zipped grads: {[z for z in zip(*gradients)]}")
    #     summed_gradients = [
    #         np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*gradients)
    #     ]
    #     self.optimizer.zero_grad()
    #     self.model.set_gradients(summed_gradients)
    #     self.optimizer.step()
    #     return self.model.get_weights()

    def apply_gradients(self, gradients):
        ready_gradients = ray.get(gradients)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*ready_gradients)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def get_weights(self):
        return self.model.get_weights()

    def run_synch_experiment(self):
      test_loader = get_data_loader()[1]

      print("Running synchronous parameter server training.")
      current_weights = self.get_weights()
      for i in range(iterations):
          gradients = [compute_gradients.remote(current_weights) for _ in range(num_workers)]
          # Calculate update after all gradients are available.
          current_weights = self.apply_gradients(gradients)

          if i % 10 == 0:
              # Evaluate the current model.
              self.model.set_weights(current_weights)
              accuracy = evaluate(self.model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

      print("Final accuracy is {:.1f}.".format(accuracy))

    def run_asynch_experiment(self):
      model = ConvNet()
      test_loader = get_data_loader()[1]

      print("Running Asynchronous Parameter Server Training.")
      current_weights = self.get_weights()
      gradients = []
      for _ in range(num_workers):
          gradients.append(compute_gradients.remote(current_weights))

      for i in range(iterations * num_workers):
          ready_gradient_list, _ = ray.wait(gradients)
          ready_gradient_id = ready_gradient_list[0]
          gradients.remove(ready_gradient_id)

          # Compute and apply gradients.
          current_weights = self.apply_gradients([ready_gradient_id])
          gradients.append(compute_gradients.remote(current_weights))

          if i % 10 == 0:
              # Evaluate the current model after every 10 updates.
              model.set_weights(current_weights)
              accuracy = evaluate(model, test_loader)
              print("Iter {}: \taccuracy is {:.1f}".format(i, accuracy))

      print("Final accuracy is {:.1f}.".format(accuracy))