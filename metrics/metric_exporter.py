import ray
from ray.util.metrics import Gauge

@ray.remote
class MetricExporter:
    def __init__(self, experiment_name):
        """
        To instrument custom metrics, add Counter, Gauge, or Histogram field 
        into this global actor and add a corresponding set method. Then pass it
        to Ray experiment actors/tasks. Corresponding Prometheus metrics will be
        collected.
        
        Parameters:
        experiment_name (str): The name of the experiment for tagging the 
        metrics.
        """
        self.accuracy_gauge = Gauge(
            "training_accuracy",
            description="Accuracy of the current training run.",
            tag_keys=("Experiment",)
        )
        self.accuracy_gauge.set_default_tags({"Experiment": experiment_name})

        self.loss_gauge = Gauge(
            "training_loss",
            description="Loss of the current training run.",
            tag_keys=("Experiment",)
        )
        self.loss_gauge.set_default_tags({"Experiment": experiment_name})

        self.gradients_gauge = Gauge(
            "processed_gradients",
            description="Cumulative number of gradients processed in the weight update.",
            tag_keys=("Experiment",)
        )
        self.gradients_gauge.set_default_tags({"Experiment": experiment_name})

        self.cumulative_gradients = 0

    def set_accuracy(self, accuracy):
        self.accuracy_gauge.set(accuracy)

    def set_loss(self, loss):
        self.loss_gauge.set(loss)

    def set_gradients_processed(self, count):
        self.cumulative_gradients += count
        self.gradients_gauge.set(self.cumulative_gradients)
