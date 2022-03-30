import robosuite as suite
import yaml

import time
import numpy as np
import matplotlib.pyplot as plt

from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config
from robosuite.environments.base import register_env

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


from typing import Callable

from src.environments import Lift_4_objects

from src.wrapper import GymWrapper_multiinput
from tqdm.auto import tqdm

class ProgressBarCallback(BaseCallback):
    """
    :param pbar: (tqdm.pbar) Progress bar object
    """
    def __init__(self, pbar):
        super(ProgressBarCallback, self).__init__()
        self._pbar = pbar

    def _on_step(self):
        # Update the progress bar:
        self._pbar.n = self.num_timesteps
        self._pbar.update(0)

# this callback uses the 'with' block, allowing for correct initialisation and destruction
class ProgressBarManager(object):
    def __init__(self, total_timesteps): # init object with total timesteps
        self.pbar = None
        self.total_timesteps = total_timesteps
        
    def __enter__(self): # create the progress bar and callback, return the callback
        self.pbar = tqdm(total=self.total_timesteps)
            
        return ProgressBarCallback(self.pbar)

    def __exit__(self, exc_type, exc_val, exc_tb): # close the callback
        self.pbar.n = self.total_timesteps
        self.pbar.update(0)
        self.pbar.close()



def make_robosuite_env(env_id, options, observations, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = GymWrapper_multiinput(suite.make(env_id, **options), observations)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':

    # The different number of processes that will be used
    n_procs = 1
    PROCESSES_TO_TEST = 1
    NUM_EXPERIMENTS = 3 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
    TRAIN_STEPS = 200
    # Number of episodes for evaluation
    EVAL_EPS = 20
    ALGO = PPO
    
    register_env(Lift_4_objects)

    with open("multiprocess.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    env_options = config["robosuite"]
    env_id = env_options.pop("env_id")

    obs_config = config["observations"]
    obs_image = [obs_config["rgb"]] #lager en liste av det

    # We will create one environment to evaluate the agent on
    eval_env = Monitor(GymWrapper(suite.make(env_id, **env_options),obs_image)) # Denne lager da standard environment

    reward_averages = []
    reward_std = []
    training_times = []
    total_procs = 0

    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        train_env = DummyVecEnv([lambda: Monitor(GymWrapper(suite.make(env_id, **env_options),obs_image))])
    else:
        # Here we use the "fork" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
        print("lager flere envs")
        train_env = SubprocVecEnv([make_robosuite_env(env_id, env_options, obs_image, i, seed=3) for i in range(n_procs)])
    
    
    """Go to Worb bro!!!!!!!!"""
    env = VecNormalize(train_env)

    obs = train_env.reset()
    print(obs)



    """Goooood work bro!!!"""
    rewards = []

    train_env.close()

