#!/scratch/cluster/clw4542/rlzoo/bin/python
import argparse
import gym

from envs.demo_wrapper import DemoWrapper
from envs.time_feature_wrapper import TimeFeatureWrapper
from envs.reward_wrapper import RewardWrapper
from envs.sparsifier_wrapper import SparsifyWrapper
from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool, get_demo

params = load_parameters()
paths = load_paths()


def argparser():
    parser = argparse.ArgumentParser(
        "Training RL alone or with demonstration")
    parser.add_argument('--algo', type=str, choices=['sac', 'td3'], default=None)
    parser.add_argument('--env_id', type=str, choices=[
                        'Reacher-v2', 'Swimmer-v2', 'HalfCheetah-v2', 'Ant-v2', 'Hopper-v2', 'Walker2d-v2'], default=None)
    parser.add_argument('--task', type=str)
    parser.add_argument('--displace_t', type=int, default=1)
    parser.add_argument(
        '--run_id', type=int, help="if specified, save file will have id=run_id+1", default=None)
    parser.add_argument('--demo_algo', type=str, default=None)
    parser.add_argument('--demo_level', type=str, default=None)
    parser.add_argument('--sparse_rew', type=str2bool,
                        help="sparsify mujoco task reward by accumulating for rew_delay timesteps", default=False)
    return parser.parse_args()


def make_custom_env(args, eval_mode):
    results_dir = paths['rl_demo']['results_dir']
    args.task_log_name = f"{args.task}_{args.algo}"
    rew_delay = params["training"][args.env_id.replace("-v2", "").lower()]["rew_delay"]
    sin_cos_repr = params["training"][args.env_id.replace("-v2", "").lower()]["sin_cos_repr"]

    env = gym.make(args.env_id)

    if args.task == 'baseline_rl':
        if args.sparse_rew:
            env = SparsifyWrapper(env, rew_delay=rew_delay)
        # write a custom wrapper for the sin-cos repr???
        args.task_log_name = f"{args.task}_hidden={params['baseline_rl']['hidden_size']}_sparse-rew={args.sparse_rew}"

    elif args.task == 'time_feature_rl':
        gamma = params["sac"][args.env_id.replace("-v2", "").lower()]["gamma"]
        env = TimeFeatureWrapper(env)
        if args.sparse_rew:
            env = SparsifyWrapper(env, rew_delay=rew_delay)
        args.task_log_name = f"{args.task}_sparse-rew={args.sparse_rew}"

    elif args.task == "raw_demo_time_feat_rl":
        expert_demo, _ = get_demo(
            args.env_id, demo_algo=args.demo_algo, raw=True, sin_cos_repr=sin_cos_repr, shuffle=False)
        env = TimeFeatureWrapper(env)
        if args.sparse_rew:
            env = SparsifyWrapper(env, rew_delay=rew_delay)
        env = DemoWrapper(env, expert_demo, displace_t=args.displace_t)
        args.task_log_name = f"{args.task}_demo={args.demo_level}_sparse-rew={args.sparse_rew}"

    elif args.task == "potential_dense_time_feat_rl":
        expert_demo, _ = get_demo(
            args.env_id, demo_algo=args.demo_algo, raw=True, sin_cos_repr=sin_cos_repr, shuffle=False)
        env = TimeFeatureWrapper(env)
        env = RewardWrapper(env, args.env_id, expert_demo,
                            reward_type="potential" if eval_mode is False else "env", 
                            displace_t=args.displace_t, raw=True)
        if args.sparse_rew:
            env = SparsifyWrapper(env, rew_delay=rew_delay)
        args.task_log_name = f"{args.task}_demo={args.demo_level}_sparse-rew={args.sparse_rew}"

    elif args.task == "huber+env_time_feat_rl":
        expert_demo, _ = get_demo(
            args.env_id, demo_algo=args.demo_algo, raw=True, sin_cos_repr=sin_cos_repr, shuffle=False)
        env = TimeFeatureWrapper(env)
        if args.sparse_rew:
            env = SparsifyWrapper(env, rew_delay=rew_delay)
        env = RewardWrapper(env, args.env_id, expert_demo,
                            reward_type="huber" if eval_mode is False else "env", 
                            displace_t=args.displace_t, raw=True)
        args.task_log_name = f"{args.task}_demo={args.demo_level}_sparse-rew={args.sparse_rew}"

    elif args.task == "sbs_time_feat_rl":
        expert_demo, _ = get_demo(
            args.env_id, demo_algo=args.demo_algo, raw=False, sin_cos_repr=sin_cos_repr, shuffle=False)
        env = TimeFeatureWrapper(env)
        if args.sparse_rew:
            env = SparsifyWrapper(env, rew_delay=rew_delay)
        env = RewardWrapper(env, args.env_id, expert_demo,
                            reward_type="sbs_potential" if eval_mode is False else "env", 
                            displace_t=args.displace_t, raw=False, time_feat=True)
        args.task_log_name = f"{args.task}_demo={args.demo_level}_sparse-rew={args.sparse_rew}"

    else:
        raise Exception("No matching tasks found.")

    return env


def train_sac(args):
    from expert_traj.train_sample_sac import train

    env = make_custom_env(args, eval_mode=False)
    eval_env = make_custom_env(args, eval_mode=True)

    train(args.env_id, env, eval_env,
          task_name=args.task_log_name,
          save_tb_logs=params["eval"]["save_tb_logs"],
          save_checkpoints=params["eval"]["ckpt_options"],
          run_id=args.run_id)
    env.close()
    eval_env.close()


def train_td3(args):
    from expert_traj.train_sample_td3 import train

    env = make_custom_env(args, eval_mode=False)
    eval_env = make_custom_env(args, eval_mode=True)

    train(args.env_id, env, eval_env,
          task_name=args.task_log_name,
          save_tb_logs=params["eval"]["save_tb_logs"],
          save_checkpoints=params["eval"]["ckpt_options"],
          run_id=args.run_id)
    env.close()
    eval_env.close()


if __name__ == '__main__':
    args = argparser()
    if args.algo=="sac":
        train_sac(args)
    elif args.algo=="td3":
        train_td3(args)