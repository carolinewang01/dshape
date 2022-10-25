#!/scratch/cluster/clw4542/rlzoo/bin/python
import os
import shutil
import argparse
import numpy as np
import gym

from envs.goal_demo_wrapper import GoalDemoWrapper
from envs.time_feature_wrapper import TimeFeatureWrapper
from envs.sparsifier_wrapper import SparsifyWrapper
from utils.load_confs import load_parameters, load_paths
from utils.helpers import get_demo, str2bool
from utils.her_utils import CustomHERGoalEnvWrapper, GoalSelectionStrategy, KEY_TO_GOAL_STRATEGY

paths = load_paths()
params = load_parameters()
EXPT_NAME = params["her_demo"]["expt_name"]
RESULTS_DIR = paths['rl_demo']['results_dir']


def argparser():
    parser = argparse.ArgumentParser("Training DShape with state-only demos")
    parser.add_argument('--algo', type=str, choices=['sac', 'td3'], default=None)
    parser.add_argument('--env_id', type=str, choices=[
                        'Reacher-v2', 'Swimmer-v2', 'HalfCheetah-v2', 'Ant-v2', 'Hopper-v2', 'Walker2d-v2'], default=None)
    parser.add_argument('--raw', type=str2bool, default=True)
    parser.add_argument('--gss', type=str, choices=['episode', 'future', 'final', 'random', 'next_state',
                                                    'next_state_and_ep', 'episode_nearest', 
                                                    'episode_nearest_future'], default='episode')
    parser.add_argument('--rew', type=str, choices=[
                        'sparse', 'dense', 'huber+env', 'env', 'potential_dense'], default='potential_dense')
    parser.add_argument('--demo_algo', type=str, default=None)
    parser.add_argument('--demo_level', type=str, default=None)
    parser.add_argument('--n_sampled_goal', type=int, default=10)
    parser.add_argument('--time_feat', type=str2bool, default=True)
    parser.add_argument('--run_id', type=int, default=None)
    parser.add_argument('--displace_t', type=int, default=1)
    parser.add_argument('--random_exploration', help="% time random action is taken", type=float, default=0.0)
    parser.add_argument('--sin_cos_repr', type=str2bool, help="representation of joint angles", default=True)
    parser.add_argument('--sparse_rew', type=str2bool, help="sparsify mujoco task reward by accumulating for rew_delay timesteps", default=False)
    return parser.parse_args()


def make_custom_env(args, expert_demo, algo_name):
    # terminal_phi0 = params['pbrs']['terminal_phi0']
    rew_delay = params["training"][args.env_id.replace("-v2", "").lower()]["rew_delay"]
    learning_starts = params[args.algo][args.env_id.replace("-v2", "").lower()]["learning_starts"]

    # gamma = params[algo_name][args.env_id.replace("-v2", "").lower()]["gamma"]
    # buffer = params[algo_name][args.env_id.replace("-v2", "").lower()]["buffer_size"]

    args.task_log_name = f"{EXPT_NAME}_algo={args.algo}_rew={args.rew}_raw={args.raw}_gss={args.gss}_n-goal={args.n_sampled_goal}_demo={args.demo_level}_sparse-rew={args.sparse_rew}_sin-cos-repr={args.sin_cos_repr}_learning-starts={learning_starts}"

    envs_dict = {"train": None, "eval": None}
    for env_type in ["train", "eval"]:
        # use list because order matters
        envs_dict[env_type] = gym.make(args.env_id)
        wrappers = [(TimeFeatureWrapper, {}) if args.time_feat else (None, None), 
                    (SparsifyWrapper, {"rew_delay": rew_delay}) if args.sparse_rew else (None, None),
                    (GoalDemoWrapper, {"env_id": args.env_id, 
                                       "expert_demo": expert_demo, 
                                       "raw": args.raw, 
                                       "displace_t": args.displace_t,
                                       # goal achieving reward
                                       "reward_type": args.rew if env_type=='train' else 'env', 
                                       "distance_threshold": 1e-3, 
                                       "time_feat": args.time_feat, 
                                       "sin_cos_repr": args.sin_cos_repr}), 
                    (CustomHERGoalEnvWrapper, {})
                    ]

        for wrapper, wrapper_args in wrappers:
            if wrapper is None: continue
            envs_dict[env_type] = wrapper(env=envs_dict[env_type], **wrapper_args)

    return envs_dict["train"], envs_dict["eval"]


def train_dshape_sac(args, expert_demo_obs):
    from expert_traj.train_sample_sac import CustomPolicy
    from utils.stable_baselines_helpers import _get_latest_run_id, EvalCallback
    from her_demo.sac_her_custom import SACCustom as SAC
    from her_demo.her_custom import CustomHER as HER

    # logging/checkpoint settings
    save_tb_logs = params["eval"]["save_tb_logs"]
    save_checkpoints = params["eval"]["ckpt_options"]

    env, eval_env = make_custom_env(args, expert_demo_obs, "sac")
    # setup
    params_algo = params["sac"][args.env_id.split("-")[0].lower()]
    params_train = params["training"][args.env_id.split("-")[0].lower()]

    tb_log_name = f"sac_{args.env_id}"
    tb_log_path = f"{RESULTS_DIR}/{args.task_log_name}/log/"

    save_name = f"sac_{args.env_id}"
    save_path = f"{RESULTS_DIR}/{args.task_log_name}/checkpoint/"

    if args.run_id is None:
        latest_run_id = _get_latest_run_id(tb_log_path, tb_log_name)
    else:
        latest_run_id = args.run_id

    tb_log_path = os.path.join(tb_log_path, f"{tb_log_name}_{latest_run_id+1}")
    save_path = os.path.join(save_path, f"{save_name}_{latest_run_id+1}")

    # wipe existing dir
    if os.path.exists(tb_log_path):
        shutil.rmtree(tb_log_path, ignore_errors=True)
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)

    model = HER(policy=CustomPolicy, env=env, model_class=SAC,
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy=KEY_TO_GOAL_STRATEGY[args.gss],
                total_timesteps=int(params_train["max_timesteps"]),
                expert_demo=expert_demo_obs,
                time_feat=args.time_feat,
                sin_cos_repr=args.sin_cos_repr,
                raw=args.raw, env_id=args.env_id,
                verbose=1,
                tensorboard_log=tb_log_path if save_tb_logs else None,
                # sac params
                batch_size=params_algo["batch_size"], 
                buffer_size=int(params_algo["buffer_size"]),
                ent_coef=params_algo["ent_coef"], gamma=params_algo["gamma"],
                learning_starts=params_algo["learning_starts"], learning_rate=params_algo["learning_rate"],
                random_exploration=args.random_exploration, seed=params["training"]["seeds"][latest_run_id]
                )

    eval_callback = EvalCallback(eval_env,
                                 n_eval_episodes=params["eval"]["n_eval_episodes"],
                                 best_model_save_path=save_path if "best" in save_checkpoints else None,
                                 log_path=tb_log_path,
                                 eval_freq=params["eval"]["eval_freq"][args.env_id.split(
                                     "-")[0].lower()],
                                 deterministic=True, render=False)

    model.learn(log_interval=1000,
                tb_log_name=tb_log_name,
                save_checkpoints=True if "all" in save_checkpoints else False,
                save_at_end=True if "last" in save_checkpoints else False,
                save_interval=params_train["save_per_iter"],
                save_name=save_name,
                save_path=save_path,
                callback=eval_callback)

    env.close()
    eval_env.close()

def train_dshape_td3(args, expert_demo_obs):
    from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    from expert_traj.train_sample_td3 import CustomPolicy
    from utils.stable_baselines_helpers import _get_latest_run_id, EvalCallback
    from her_demo.td3_her_custom import TD3Custom as TD3
    from her_demo.her_custom import CustomHER as HER

    # logging/checkpoint settings
    save_tb_logs = params["eval"]["save_tb_logs"]
    save_checkpoints = params["eval"]["ckpt_options"]

    env, eval_env = make_custom_env(args, expert_demo_obs, "td3")
    # setup
    params_algo = params["td3"][args.env_id.split("-")[0].lower()]
    params_train = params["training"][args.env_id.split("-")[0].lower()]

    tb_log_name = f"td3_{args.env_id}"
    tb_log_path = f"{RESULTS_DIR}/{args.task_log_name}/log/"

    save_name = f"td3_{args.env_id}"
    save_path = f"{RESULTS_DIR}/{args.task_log_name}/checkpoint/"

    if args.run_id is None:
        latest_run_id = _get_latest_run_id(tb_log_path, tb_log_name)
    else:
        latest_run_id = args.run_id

    tb_log_path = os.path.join(tb_log_path, f"{tb_log_name}_{latest_run_id+1}")
    save_path = os.path.join(save_path, f"{save_name}_{latest_run_id+1}")

    # wipe existing dir
    if os.path.exists(tb_log_path):
        shutil.rmtree(tb_log_path, ignore_errors=True)
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = HER(policy=CustomPolicy, env=env, model_class=TD3,
                n_sampled_goal=args.n_sampled_goal,
                goal_selection_strategy=KEY_TO_GOAL_STRATEGY[args.gss],
                total_timesteps=int(params_train["max_timesteps"]),
                expert_demo=expert_demo_obs,
                time_feat=args.time_feat,
                sin_cos_repr=args.sin_cos_repr,
                raw=args.raw, env_id=args.env_id,
                verbose=1,
                tensorboard_log=tb_log_path if save_tb_logs else None,
                # td3 params
                action_noise=action_noise,
                batch_size=params_algo["batch_size"], 
                buffer_size=int(params_algo["buffer_size"]),
                tau=params_algo["tau"], gamma=params_algo["gamma"],
                learning_starts=params_algo["learning_starts"], learning_rate=params_algo["learning_rate"],
                gradient_steps=params_algo["gradient_steps"],
                target_policy_noise=0.2, target_noise_clip=0.5,
                policy_delay=params_algo["policy_delay"],
                seed=params["training"]["seeds"][latest_run_id]
                )

    eval_callback = EvalCallback(eval_env,
                                 n_eval_episodes=params["eval"]["n_eval_episodes"],
                                 best_model_save_path=save_path if "best" in save_checkpoints else None,
                                 log_path=tb_log_path,
                                 eval_freq=params["eval"]["eval_freq"][args.env_id.split(
                                     "-")[0].lower()],
                                 deterministic=True, render=False)

    model.learn(log_interval=1000,
                tb_log_name=tb_log_name,
                save_checkpoints=True if "all" in save_checkpoints else False,
                save_at_end=True if "last" in save_checkpoints else False,
                save_interval=params_train["save_per_iter"],
                save_name=save_name,
                save_path=save_path,
                callback=eval_callback)

    env.close()
    eval_env.close()


if __name__ == '__main__':
    args = argparser()
    # TODO: rename expert_demo_raw_obs for GOAL obs- may not always be raw. 
    expert_demo_raw_obs, _ = get_demo(args.env_id, demo_algo=args.demo_algo,
                                      raw=args.raw, shuffle=False, 
                                      time_feat=args.time_feat, sin_cos_repr=args.sin_cos_repr)

    if args.algo=='sac':
        train_dshape_sac(args, expert_demo_raw_obs)
    elif args.algo=="td3":
        train_dshape_td3(args, expert_demo_raw_obs)
