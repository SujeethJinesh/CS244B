import torch
import numpy as np
import ray
import time
import os
from ray import train
from workers.worker_task import compute_gradients
# from models.test_model import ConvNet
# from models.test_model import ConvNet, get_data_loader, evaluate
from models.fashion_mnist import FashionMNISTConvNet, fashion_mnist_get_data_loader
from models.model_common import evaluate

iterations = 200
num_workers = 2

@ray.remote(max_restarts=-1, max_task_retries=-1)
class ParameterServerDiskCkpoint(object):
    def __init__(self, lr, checkpoint_dir):
        self.model = FashionMNISTConvNet()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.checkpoint_dir = checkpoint_dir

        # ====== Resume training state from the checkpoint. ======
        self.start_iteration = 0
        model_file = os.path.join(checkpoint_dir, "model.pt")
        if os.path.exists(model_file):
          model_state_dict = torch.load(model_file)
          self.model.load_state_dict(model_state_dict)
        extra_state_file = os.path.join(checkpoint_dir, "extra_state.pt")
        if os.path.exists(extra_state_file):
          self.start_iteration = (torch.load(extra_state_file)["iteration_count"] + 1)
        # ========================================================

    def apply_gradients(self, gradients):
        grad = ray.get(gradients)
        summed_gradients = [
            np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*grad)
        ]
        self.optimizer.zero_grad()
        self.model.set_gradients(summed_gradients)
        self.optimizer.step()
        return self.model.get_weights()

    def set_weights(self, weights, iteration_count):
        if weights:
          self.model.set_weights(weights)
          # === Make sure to save all state needed for resuming training ===
          torch.save(
              self.model.state_dict(),  # NOTE: Unwrap the model.
              os.path.join(self.checkpoint_dir, "model.pt"),
          )
          torch.save(
              {"iteration_count": iteration_count},
              os.path.join(self.checkpoint_dir, "extra_state.pt"),
          )
          # ================================================================


    def get_weights(self):
        return self.model.get_weights()

    def run_training(self, synchronous=True):
      if synchronous:
        self.run_synch_training()
      else:
        self.run_asynch_training()


    def run_synch_training(self):
      test_loader = fashion_mnist_get_data_loader()[1]

      print("Running synchronous parameter server training.")
      current_weights = self.get_weights()
      for i in range(self.start_iteration, iterations):
          gradients = [compute_gradients.remote(current_weights) for _ in range(num_workers)]
          # Calculate update after all gradients are available.
          current_weights = self.apply_gradients(gradients)

          if i % 10 == 0:
              # Evaluate the current model.
              self.set_weights(current_weights, i)
              accuracy, loss = evaluate(self.model, test_loader)
              print("Time {}: \taccuracy is {:.3f}\tloss is {:.3f}".format(int(time.time()), accuracy, loss))

      print("Final accuracy is {:.3f}.".format(accuracy))

    def run_asynch_training(self):
      test_loader = fashion_mnist_get_data_loader()[1]

      print("Running Asynchronous Parameter Server Training.")
      current_weights = self.get_weights()
      gradients = []
      for _ in range(num_workers):
          gradients.append(compute_gradients.remote(current_weights))

      for i in range(self.start_iteration, iterations * num_workers):
          ready_gradient_list, _ = ray.wait(gradients)
          ready_gradient_id = ready_gradient_list[0]
          gradients.remove(ready_gradient_id)

          # Compute and apply gradients.
          current_weights = self.apply_gradients([ready_gradient_id],)
          gradients.append(compute_gradients.remote(current_weights))

          if i % 10 == 0:
              # Evaluate the current model after every 10 updates.
              self.set_weights(current_weights, i)
              accuracy, loss = evaluate(self.model, test_loader)
              print("Time {}: \taccuracy is {:.3f}\tloss is {:.3f}".format(int(time.time()), accuracy, loss))

      print("Final accuracy is {:.3f}.".format(accuracy))

    def exit(self, sleep_sec):
        print("in exit method")
        time.sleep(sleep_sec)
        os._exit(0)
