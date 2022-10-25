import os
import shutil
import argparse
import numpy as np

from stable_baselines.td3.policies import FeedForwardPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.callbacks import EvalCallback

from expert_traj.expert_algos.td3 import TD3
from utils.load_confs import load_parameters, load_paths
from utils.stable_baselines_helpers import _get_latest_run_id

params = load_parameters()
paths = load_paths()


def argparser():
    parser = argparse.ArgumentParser(
        "Training on Mujoco using TD3 from stable_baselines")
    # for all
    parser.add_argument(
        '--task', type=str, choices=['train', 'sample_best', 'sample_all'], default='train')
    parser.add_argument('--env_id', help='environment ID')
    parser.add_argument('--seed', type=int, default=872)
    return parser.parse_args()

# Custom MLP policy of 2 layers of size 256 each with relu activation
class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                              layers=[256, 256],
                                              layer_norm=False,
                                              feature_extraction="mlp")

def train(env_id, env, eval_env, task_name, 
          save_tb_logs=False, save_checkpoints=[],
          run_id=None, 
          results_dir=paths['rl_demo']['results_dir']):
    """
    Save checkpoints should be a list that is a sublist of ["best", "all", "last"]
    "best": save best checkpoint only 
    "all" : save all checkpoints 
    "last": save last checkpoint only
    """
    assert set(save_checkpoints).issubset(set(["best", "all", "last"]))
    params_algo = params["td3"][env_id.split("-")[0].lower()]
    params_train = params["training"][env_id.split("-")[0].lower()]

    tb_log_name = f"td3_{env_id}"
    tb_log_path = f"{results_dir}/{task_name}/log/"
    save_name = f"td3_{env_id}"
    save_path = f"{results_dir}/{task_name}/checkpoint/"

    if run_id is None:
        latest_run_id = _get_latest_run_id(tb_log_path, tb_log_name)
    else:
        latest_run_id = run_id

    tb_log_path = os.path.join(tb_log_path, f"{tb_log_name}_{latest_run_id+1}")
    save_path = os.path.join(save_path, f"{save_name}_{latest_run_id+1}")

    # wipe existing dir
    if os.path.exists(tb_log_path):
        shutil.rmtree(tb_log_path, ignore_errors=True)
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(CustomPolicy, env, 
                action_noise=action_noise,
                verbose=1,
                tensorboard_log=tb_log_path if save_tb_logs else None,
                gamma=params_algo["gamma"],
                learning_rate=params_algo["learning_rate"],
                buffer_size=int(params_algo["buffer_size"]),
                batch_size=params_algo["batch_size"],
                gradient_steps=params_algo["gradient_steps"], 
                tau=params_algo["tau"],
                target_policy_noise=0.2, target_noise_clip=0.5,
                policy_delay=params_algo["policy_delay"],
                learning_starts=params_algo["learning_starts"], 
                seed=params["training"]["seeds"][latest_run_id]
                )
    eval_callback = EvalCallback(eval_env,
                                 n_eval_episodes=params["eval"]["n_eval_episodes"],
                                 best_model_save_path=save_path if "best" in save_checkpoints else None,
                                 log_path=tb_log_path,
                                 # eval agent every eval_freq of callback; callback is called for every training step of model
                                 eval_freq=params["eval"]["eval_freq"][env_id.split(
                                     "-")[0].lower()],
                                 deterministic=True, render=False)

    model.learn(total_timesteps=int(params_train["max_timesteps"]),
                log_interval=1000,
                tb_log_name=tb_log_name,
                save_checkpoints=True if "all" in save_checkpoints else False,
                save_at_end=True if "last" in save_checkpoints else False,
                save_interval=params_train["save_per_iter"],
                save_path=save_path,
                save_name=save_name,
                callback=eval_callback)
