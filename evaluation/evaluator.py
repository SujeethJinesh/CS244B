import time
from evaluation.evaluator_state import evaluator_state
from kazoo.client import KazooClient
import ray

def _get_weights(zk):
  retrieved_data = zk.get("/base/weights")
  unpickled_w_string = ray.cloudpickle.loads(retrieved_data[0])
  return ray.get(unpickled_w_string)

def async_eval(timer_runs, model, test_loader, metric_exporter, evaluate, use_zk=False):
  zk = KazooClient(hosts='127.0.0.1:2181')
  zk.start()

  while timer_runs.is_set():
    if use_zk:
      evaluator_state.weights_lock.acquire()
      evaluator_state.CURRENT_WEIGHTS = _get_weights(zk)
      model.set_weights(evaluator_state.CURRENT_WEIGHTS)
      evaluator_state.weights_lock.release()
    elif evaluator_state.CURRENT_WEIGHTS:
      evaluator_state.weights_lock.acquire()
      model.set_weights(evaluator_state.CURRENT_WEIGHTS)
      evaluator_state.weights_lock.release()
      
      accuracy = evaluate(model, test_loader)
      print("accuracy is {:.1f}".format(accuracy))
      metric_exporter.set_accuracy.remote(accuracy)

    time.sleep(2)
