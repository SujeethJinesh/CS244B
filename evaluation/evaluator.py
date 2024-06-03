import time
from evaluation.evaluator_state import evaluator_state

def async_eval(timer_runs, model, test_loader, metric_exporter, evaluate):
  while timer_runs.is_set():
    if evaluator_state.CURRENT_WEIGHTS:
      evaluator_state.weights_lock.acquire()
      model.set_weights(evaluator_state.CURRENT_WEIGHTS)
      evaluator_state.weights_lock.release()
      
      accuracy, loss = evaluate(model, test_loader)
      print("Time {}: \taccuracy is {:.3f}\tloss is {:.3f}".format(int(time.time()), accuracy, loss))
      metric_exporter.set_accuracy.remote(accuracy)

    time.sleep(2)
