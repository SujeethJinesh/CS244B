import ray
from ray.util.metrics import Gauge

runtime_env = {"pip": ["kazoo"]}
ray.init(ignore_reinit_error=True, _metrics_export_port=8081, runtime_env=runtime_env)

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

        self.zookeeper_reads_gauge = Gauge(
            "zookeeper_reads",
            description="Number of read hits to Zookeeper.",
            tag_keys=("Experiment",)
        )
        self.zookeeper_reads_gauge.set_default_tags({"Experiment": experiment_name})

        self.zookeeper_writes_gauge = Gauge(
            "zookeeper_writes",
            description="Number of write hits to Zookeeper.",
            tag_keys=("Experiment",)
        )
        self.zookeeper_writes_gauge.set_default_tags({"Experiment": experiment_name})

        self.cumulative_gradients = 0
        self.zookeeper_reads = 0
        self.zookeeper_writes = 0

    def set_accuracy(self, accuracy):
        self.accuracy_gauge.set(accuracy)

    def set_loss(self, loss):
        self.loss_gauge.set(loss)

    def set_gradients_processed(self, count):
        self.cumulative_gradients += count
        self.gradients_gauge.set(self.cumulative_gradients)

    def set_zookeeper_reads(self, count):
        self.zookeeper_reads += count
        self.zookeeper_reads_gauge.set(self.zookeeper_reads)

    def set_zookeeper_writes(self, count):
        self.zookeeper_writes += count
        self.zookeeper_writes_gauge.set(self.zookeeper_writes)
