import os
import numpy as np
import shutil
from gym.wrappers.time_limit import TimeLimit

from envs.reward_wrapper import RewardWrapper
from envs.time_feature_wrapper import TimeFeatureWrapper
from envs.goal_demo_wrapper import GoalDemoWrapper
from envs.demo_wrapper import DemoWrapper
from envs.sparsifier_wrapper import SparsifyWrapper

from poc.utils import extend_demo
from poc.pointworld_env import PointWorld

from utils.load_confs import load_parameters, load_paths
from utils.her_utils import CustomHERGoalEnvWrapper, GoalSelectionStrategy, KEY_TO_GOAL_STRATEGY

params = load_parameters()
paths = load_paths()
RESULTS_DIR = paths['rl_demo']['results_dir']


def generate_demo(env, max_episode_steps):
    base_demo, demo_return = env.unwrapped.generate_solution()
    demo = np.array(extend_demo(base_demo, max_episode_steps,
                                extension_type="demo",
                                time_feat=True
                                ))
    return demo, demo_return


def make_env(task_vars: list, env_size, goal, max_episode_steps, 
             sparse_rew=True,
             rew_base_value=-1,
             early_term=True, 
             rew_delay=1, 
             potential_coef=1, 
             eval_mode=False):
    # TODO: implement SEEDING in POINTWORLD ENV
    env = TimeLimit(PointWorld(size=env_size, goal=goal, early_term=early_term, 
                               sparse_rew=sparse_rew,
                               rew_base_value=rew_base_value),
                    max_episode_steps=max_episode_steps)
    displace_t = 1
    demo, demo_return = generate_demo(env, max_episode_steps)
    # print("DEMO RETURN IS ", demo_return)
    if "time_feat" in task_vars:
        env = TimeFeatureWrapper(env)

    if "sparse" in task_vars:
        env = SparsifyWrapper(env, rew_delay=rew_delay)

    if "potential" in task_vars:
        env = RewardWrapper(env, env_id="pointworld",
                            expert_demo=demo,
                            # TODO: check direction of sign
                            reward_type="potential" if eval_mode is False else "env",
                            displace_t=displace_t,
                            raw=False)

    if "state_aug" in task_vars:
        env = DemoWrapper(env, demo, displace_t=displace_t)

    if "her" in task_vars:
        env = GoalDemoWrapper(env,
                              env_id="pointworld",
                              expert_demo=demo,
                              displace_t=displace_t,
                              raw=False,
                              reward_type="potential_dense" if eval_mode is False else "env",
                              distance_threshold=0.01,  # only relevant for sparse reward setting
                              time_feat=True,
                              sin_cos_repr=False,
                              potential_coef=potential_coef
                              )
        env = CustomHERGoalEnvWrapper(env)

    return env, demo, demo_return


def train_sac(task_vars, task_log_name, 
              env_size, goal, max_episode_steps, 
              gradient_steps=1,
              ent_coef=0.2,
              deterministic_policy=False,
              sparse_rew=True,
              rew_base_value=-1,
              early_term=True,
              rew_delay=1, run_id=0):
    from expert_traj.train_sample_sac import train

    env, _, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                         max_episode_steps=max_episode_steps, 
                         sparse_rew=True,
                         rew_base_value=rew_base_value,
                         early_term=early_term, rew_delay=rew_delay, 
                         eval_mode=False)
    eval_env, _, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                              max_episode_steps=max_episode_steps, 
                              sparse_rew=True,
                              rew_base_value=rew_base_value,
                              early_term=early_term, rew_delay=rew_delay, 
                              eval_mode=True)

    train("pointworld", env, eval_env,
          task_name=task_log_name,
          save_tb_logs=params["eval"]["save_tb_logs"],
          save_checkpoints=params["eval"]["ckpt_options"],
          ent_coef=ent_coef,
          deterministic_policy=deterministic_policy,
          gradient_steps=gradient_steps,
          run_id=run_id)

    env.close()
    eval_env.close()


def train_td3(task_vars, task_log_name, 
              env_size, goal, max_episode_steps, 
              gradient_steps=1,
              policy_delay=2,
              sparse_rew=True,
              rew_base_value=-1,
              early_term=True,
              rew_delay=1, run_id=0):
    from expert_traj.train_sample_td3 import train

    env, _, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                         max_episode_steps=max_episode_steps, 
                         sparse_rew=True,
                         rew_base_value=rew_base_value,
                         early_term=early_term, rew_delay=rew_delay, 
                         eval_mode=False)
    eval_env, _, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                              max_episode_steps=max_episode_steps, 
                              sparse_rew=True,
                              rew_base_value=rew_base_value,
                              early_term=early_term, rew_delay=rew_delay, 
                              eval_mode=True)

    train("pointworld", env, eval_env,
          task_name=task_log_name,
          save_tb_logs=params["eval"]["save_tb_logs"],
          save_checkpoints=params["eval"]["ckpt_options"],
          gradient_steps=gradient_steps,
          policy_delay=policy_delay,
          run_id=run_id)

    env.close()
    eval_env.close()


def train_her_sac(task_vars, task_log_name,
                  env_size, goal, max_episode_steps,
                  sparse_rew=True,
                  rew_base_value=-1,
                  early_term=True,
                  rew_delay=1, 
                  ent_coef=0.2,
                  deterministic_policy=False,
                  gradient_steps=1,
                  n_sampled_goal=3, 
                  goal_sampling_strategy="episode",
                  potential_coef=1, 
                  run_id=0,
                  ):
    # from stable_baselines.common.callbacks import EvalCallback
    from expert_traj.train_sample_sac import CustomSACPolicy
    from utils.stable_baselines_helpers import _get_latest_run_id, EvalCallback
    from her_demo.sac_her_custom import SACCustom as SAC
    from her_demo.her_custom import CustomHER as HER

    env_id = "pointworld"
    print("REW BASE VALUE IS ", rew_base_value)

    # logging/checkpoint settings
    save_tb_logs = params["eval"]["save_tb_logs"]
    save_checkpoints = params["eval"]["ckpt_options"]

    env, demo, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                            max_episode_steps=max_episode_steps, 
                            sparse_rew=True, 
                            rew_base_value=rew_base_value,
                            early_term=True,
                            rew_delay=rew_delay, potential_coef=potential_coef, 
                            eval_mode=False)
    eval_env, _, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                              max_episode_steps=max_episode_steps, 
                              sparse_rew=True, 
                              rew_base_value=rew_base_value,
                              early_term=True,
                              rew_delay=rew_delay, potential_coef=potential_coef, 
                              eval_mode=True)

    params_algo = params["sac"][env_id.split("-")[0].lower()]

    tb_log_name = f"sac_{env_id}"
    tb_log_path = f"{RESULTS_DIR}/{task_log_name}/log/"

    save_name = f"sac_{env_id}"
    save_path = f"{RESULTS_DIR}/{task_log_name}/checkpoint/"

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

    model = HER(policy=CustomSACPolicy, env=env, model_class=SAC,
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=KEY_TO_GOAL_STRATEGY[goal_sampling_strategy],
                total_timesteps=int(params_algo["max_timesteps"]),
                expert_demo=demo,
                time_feat=True,
                sin_cos_repr=False,
                raw=False, env_id=env_id,
                verbose=1,
                tensorboard_log=tb_log_path if save_tb_logs else None,
                # sac params
                ent_coef=ent_coef,
                # ent_coef=params_algo["ent_coef"], 
                deterministic_policy=deterministic_policy,
                gradient_steps=gradient_steps,
                batch_size=params_algo["batch_size"], 
                buffer_size=int(
                    params_algo["buffer_size"]),
                gamma=params_algo["gamma"],
                learning_starts=params_algo["learning_starts"], learning_rate=params_algo["learning_rate"],
                random_exploration=0.0, seed=params["training"]["seeds"][latest_run_id]
                )

    eval_callback = EvalCallback(eval_env,
                                 n_eval_episodes=params["eval"]["n_eval_episodes"],
                                 best_model_save_path=save_path if "best" in save_checkpoints else None,
                                 log_path=tb_log_path,
                                 eval_freq=params["eval"]["eval_freq"][env_id.split(
                                     "-")[0].lower()],
                                 deterministic=True, render=False)

    model.learn(log_interval=1000,
                tb_log_name=tb_log_name,
                save_checkpoints=True if "all" in save_checkpoints else False,
                save_at_end=True if "last" in save_checkpoints else False,
                save_interval=params_algo["save_per_iter"],
                save_path=save_path,
                save_name=save_name,
                callback=eval_callback)

    env.close()
    eval_env.close()


def train_her_td3(task_vars, task_log_name,
                  env_size, goal, max_episode_steps,
                  sparse_rew=True,
                  rew_base_value=-1,
                  early_term=True,
                  rew_delay=1, 
                  deterministic_policy=False,
                  gradient_steps=1,
                  n_sampled_goal=3, 
                  goal_sampling_strategy="episode",
                  potential_coef=1, 
                  run_id=0,
                  ):
    # from stable_baselines.common.callbacks import EvalCallback
    from expert_traj.train_sample_td3 import CustomPolicy
    from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

    from utils.stable_baselines_helpers import _get_latest_run_id, EvalCallback
    from her_demo.td3_her_custom import TD3Custom as TD3
    from her_demo.her_custom import CustomHER as HER

    env_id = "pointworld"
    # logging/checkpoint settings
    save_tb_logs = params["eval"]["save_tb_logs"]
    save_checkpoints = params["eval"]["ckpt_options"]

    env, demo, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                            max_episode_steps=max_episode_steps, 
                            sparse_rew=True, 
                            rew_base_value=rew_base_value,
                            early_term=True,
                            rew_delay=rew_delay, potential_coef=potential_coef, 
                            eval_mode=False)
    eval_env, _, _ = make_env(task_vars=task_vars, env_size=env_size, goal=goal,
                              max_episode_steps=max_episode_steps, 
                              sparse_rew=True, 
                              rew_base_value=rew_base_value,
                              early_term=True,
                              rew_delay=rew_delay, potential_coef=potential_coef, 
                              eval_mode=True)

    # setup
    # params_algo = params["td3"][env_id.split("-")[0].lower()]

    tb_log_name = f"td3_{env_id}"
    tb_log_path = f"{RESULTS_DIR}/{task_log_name}/log/"

    save_name = f"td3_{env_id}"
    save_path = f"{RESULTS_DIR}/{task_log_name}/checkpoint/"

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

    model = HER(policy=MlpPolicy, env=env, model_class=TD3,
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy=KEY_TO_GOAL_STRATEGY[goal_sampling_strategy],
                total_timesteps=50000, # int(params_algo["max_timesteps"]),
                expert_demo=demo,
                time_feat=True,
                sin_cos_repr=False,
                raw=False, env_id=env_id,
                verbose=1,
                tensorboard_log=tb_log_path if save_tb_logs else None,
                # td3 params
                action_noise=action_noise,
                gamma=0.99, # params_algo["gamma"],
                tau=0.005, # params_algo["tau"],
                batch_size=128, # params_algo["batch_size"], 
                buffer_size=2000, # int(params_algo["buffer_size"]),
                learning_starts=100, # params_algo["learning_starts"], 
                learning_rate=0.0003, # params_algo["learning_rate"],
                gradient_steps=gradient_steps,
                target_policy_noise=0.2,
                target_noise_clip=0.5,
                policy_delay=2, # params_algo["policy_delay"],
                seed=params["training"]["seeds"][latest_run_id]
                )


    eval_callback = EvalCallback(eval_env,
                                 n_eval_episodes=params["eval"]["n_eval_episodes"],
                                 best_model_save_path=save_path if "best" in save_checkpoints else None,
                                 log_path=tb_log_path,
                                 eval_freq=params["eval"]["eval_freq"][env_id.split(
                                     "-")[0].lower()],
                                 deterministic=True, render=False)

    model.learn(log_interval=1000,
                tb_log_name=tb_log_name,
                save_checkpoints=True if "all" in save_checkpoints else False,
                save_at_end=True if "last" in save_checkpoints else False,
                save_interval=1, # params_algo["save_per_iter"],
                save_name=save_name,
                save_path=save_path,
                callback=eval_callback)

    env.close()
    eval_env.close()


if __name__ == '__main__':
    env_size = 1
    goal = np.array([-0.4,  0.4])
    max_episode_steps = 5
    rew_delay = 1
    early_term = True

    # task_vars = ["time_feat", "potential"]
    # task_log_name = "pointworld_pbrs"
    # train_sac(task_vars, task_log_name, env_size, goal, max_episode_steps, early_term, rew_delay)

    # task_vars = ["time_feat", "potential"]
    # task_log_name = "pbrs_test_td3"
    # train_td3(task_vars, task_log_name, 
    #           env_size=env_size, goal=goal, 
    #           max_episode_steps=max_episode_steps, 
    #           gradient_steps=1,
    #           sparse_rew=True,
    #           rew_base_value=0,
    #           early_term=early_term,
    #           run_id=0)

    task_log_name = "dshape_test"
    task_vars = ["time_feat", "her"]
    train_her_td3(task_vars=task_vars, 
                  task_log_name=task_log_name, 
                  env_size=env_size, 
                  goal=goal, 
                  max_episode_steps=max_episode_steps, 
                  early_term=early_term, 
                  goal_sampling_strategy="random",
                  n_sampled_goal=1,
                  rew_base_value=0
                  )