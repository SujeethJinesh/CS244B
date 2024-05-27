import time
from evaluation.evaluator_state import evaluator_state
from shared import MODEL_MAP

def async_eval(timer_runs, model_name, test_loader, metric_exporter, evaluate):
  model = MODEL_MAP[model_name]()
  while timer_runs.is_set():
    if evaluator_state.CURRENT_WEIGHTS:
      evaluator_state.weights_lock.acquire()
      model.set_weights(evaluator_state.CURRENT_WEIGHTS)
      evaluator_state.weights_lock.release()
      
      accuracy = evaluate(model, test_loader)
      print("accuracy is {:.1f}".format(accuracy))
      metric_exporter.set_accuracy.remote(accuracy)

    time.sleep(10)
