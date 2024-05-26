import threading

class EvaluatorStateSingleton:
  _instance = None
  _lock = threading.Lock()

  def __new__(cls, *args, **kwargs):
    if not cls._instance:
      with cls._lock:
        if not cls._instance:
          cls._instance = super().__new__(cls, *args, **kwargs)

          # Note: We make CURRENT_WEIGHTS a list because those are mutable.
          cls._instance.CURRENT_WEIGHTS = []
          cls._instance.weights_lock = threading.Lock()
    return cls._instance

evaluator_state = EvaluatorStateSingleton()
