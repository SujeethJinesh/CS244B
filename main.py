import ray
import time
import asyncio
import threading

from experiments.synch import run_synch_experiment
from experiments.asynch import run_asynch_experiment
from parameter_servers.server_actor import ParameterServer

@ray.remote
def kill_server(actor_handle_list, timeout_sec=10):
    asyncio.run(kill(actor_handle_list, timeout_sec))

async def kill(actor_handle_list, timeout_sec=10):
    await asyncio.sleep(timeout_sec)
    print("disrupt thread wakes up")
    for actor_handle in actor_handle_list:
    	ray.kill(actor_handle)
    print("killed server successfully")


async def main():
  # Run asynchronous param server experiment
  ray.init(ignore_reinit_error=True)
  ps1 = ParameterServer.remote(1e-2, 1)
  ps2 = ParameterServer.remote(1e-2, 2)
  ps3 = ParameterServer.remote(1e-2, 3)
  try:
    # ray.get([kill_server.remote([ps1], 10), kill_server.remote([ps2], 25)])
    await asyncio.gather(*[kill_server.remote([ps1], 10), ps2.run_wait_synch_experiment.remote(),ps3.run_wait_synch_experiment.remote()])
    ps1.run_synch_experiment.remote()
  except Exception as e:
    print("Catching exception", e)

if __name__ == "__main__":
  asyncio.run(main())