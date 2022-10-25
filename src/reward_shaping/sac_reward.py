#!/scratch/cluster/clw4542/rlzoo/bin/python
import os
import argparse
import copy

import numpy as np
import gym

from envs.demo_wrapper import DemoWrapper
from envs.time_feature_wrapper import TimeFeatureWrapper
from envs.reward_wrapper import RewardWrapper
from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool, get_demo

params = load_parameters()
paths = load_paths()


def argparser():
    parser = argparse.ArgumentParser("Training SAC with demonstration")
    parser.add_argument('--env_id', type=str, choices=[
                        'Reacher-v2', 'Swimmer-v2', 'HalfCheetah-v2', 'Ant-v2', 'Hopper-v2', 'Walker2d-v2'], default=None)
    parser.add_argument('--task', type=str, default="huber+env_time_feat_rl")
    parser.add_argument('--displace_t', type=int, default=1)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--demo_algo', type=str, default=None)
    parser.add_argument('--demo_level', type=str, default=None)
    parser.add_argument('--params_file', type=str, default=None) # location w.r.t. confs dir 

    return parser.parse_args()


def make_custom_env(args):
    args.task_log_name = args.task

    env = gym.make(args.env_id)

    if args.task == "huber+env_time_feat_rl":
        expert_demo = get_demo(
            args.env_id, demo_algo=args.demo_algo, raw=True, shuffle=False)
        env = TimeFeatureWrapper(env)
        env = RewardWrapper(env, args.env_id, expert_demo,
                            reward_type="huber", 
                            params_filename=args.params_file,
                            displace_t=args.displace_t, 
                            raw=True)

    elif args.task == "huber2+env_time_feat_rl":
        expert_demo = get_demo(
            args.env_id, demo_algo=args.demo_algo, raw=True, shuffle=False)
        env = TimeFeatureWrapper(env)
        env = RewardWrapper(env, args.env_id, expert_demo,
                            reward_type="huber2", 
                            params_filename=args.params_file,
                            displace_t=args.displace_t, 
                            raw=True)

    elif args.task == "sbs_time_feat_rl":
        expert_demo = get_demo(
            args.env_id, demo_algo=args.demo_algo, raw=False, shuffle=False)
        env = TimeFeatureWrapper(env)
        env = RewardWrapper(env, args.env_id, expert_demo,
                            reward_type="sbs_potential", params_filename=args.params_file,
                            displace_t=args.displace_t, 
                            raw=False, time_feat=True)

    else:
        print("task is ", args.task)
        raise Exception("No matching tasks found.")

    env.param_description = "_".join([f"{k}={v}" for k, v in env.reward_params.items()]) # this line is necessary for hyperparameter search 
    args.task_log_name = f"{args.task}_displace-t={args.displace_t}_demo={args.demo_level}_{env.param_description}"

    return env


def train_sac(args):
    from expert_traj.train_sample_sac import train

    env = make_custom_env(args)
    eval_env = make_custom_env(args)
    params_algo = copy.deepcopy(params["sac"])

    # train for only 20% of full time
    for env_id in ["swimmer", "ant", "halfcheetah", "walker2d", "hopper"]:
        params_algo[env_id]["max_timesteps"] = params_algo[env_id]["max_timesteps"] * 0.2

    train(args.env_id, env, eval_env,
          task_name=args.task_log_name,
          params_algo=params_algo,
          save_tb_logs=False,
          save_checkpoints=[],
          run_id=args.run_id,
          results_dir=paths['rl_demo']['hyperparam_results_dir'])
    env.close()
    eval_env.close()


if __name__ == '__main__':
    args = argparser()
    train_sac(args)
