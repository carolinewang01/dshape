import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import gym

from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import EvalCallback

from expert_traj.expert_algos.sac import SAC
from envs.time_feature_wrapper import TimeFeatureWrapper
from utils.load_confs import load_parameters, load_paths
from utils.helpers import get_joint_angles
from utils.stable_baselines_helpers import _get_latest_run_id

params = load_parameters()
paths = load_paths()


def argparser():
    parser = argparse.ArgumentParser(
        "Training on Mujoco using SAC from stable_baselines")
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
          params_algo=params["sac"],
          gradient_steps=1,
          ent_coef=0.2,
          deterministic_policy=False,
          save_tb_logs=False, save_checkpoints=[],
          run_id=None, results_dir=paths['rl_demo']['results_dir']):
    """
    Save checkpoints should be a list that is a sublist of ["best", "all", "last"]
    "best": save best checkpoint only 
    "all" : save all checkpoints 
    "last": save last checkpoint only
    """
    assert set(save_checkpoints).issubset(set(["best", "all", "last"]))
    params_algo = params_algo[env_id.split("-")[0].lower()]
    params_train = params["training"][env_id.split("-")[0].lower()]
    # consider adding the time feature wrapper?
    # if params_algo['env_wrapper'] is not None:
    #     env = env_wrapper(env) # what is time limit wrapper??
    tb_log_name = f"sac_{env_id}"
    tb_log_path = f"{results_dir}/{task_name}/log/"
    save_name = f"sac_{env_id}"
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

    model = SAC(CustomPolicy, env, verbose=1,
                tensorboard_log=tb_log_path if save_tb_logs else None,
                batch_size=params_algo["batch_size"], 
                buffer_size=int(
                    params_algo["buffer_size"]),
                gradient_steps=gradient_steps,
                ent_coef=ent_coef, #params_algo["ent_coef"],
                deterministic_policy=deterministic_policy, 
                gamma=params_algo["gamma"],
                learning_starts=params_algo["learning_starts"], learning_rate=params_algo["learning_rate"],
                random_exploration=0.0, seed=params["training"]["seeds"][latest_run_id]
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
                save_name=save_name,
                save_path=save_path,
                callback=eval_callback)


def generate_data(env, env_id, num_trajs, manual, visual,
                  model=None,
                  save_path=False, save_max_len_only=True, deterministic=True,
                  remove_time_feat=True, random=False):
    ep = 0
    obs = []
    acs = []
    rews = []
    ep_lens = []
    ep_rets = []
    tot_rew = []
    while True:  # iterate over num_traj
        if visual:
            env.render("human")
        ob = env.reset()
        if visual:
            env.render()

        ep_obs = []
        ep_acs = []
        ep_rews = []
        score = 0
        done, state = False, None

        ep_obs.append(ob)
        while not done:  # one episode
            if visual:
                # time.sleep(1/60)
                time.sleep(1 / 10)
                env.render("human")
            if random:
                ac = env.action_space.sample()
            else:
                ac, state = model.predict(
                    ob, state=state, deterministic=deterministic)
            ob, r, done, _ = env.step(ac)

            ep_obs.append(ob)
            ep_acs.append(ac)
            ep_rews.append(r)
            score += r  # score is return
        print("score:", score)
        tot_rew.append(score)
        if manual:  # manually specify episodes to save
            var = input("Are you inclined to save this trajectory? (y/n)")
            if var == "y":
                print("episode:", ep)
                ep += 1
                obs.append(np.vstack(ep_obs))
                acs.append(np.vstack(ep_acs))
                rews.append(ep_rews)
                ep_rets.append(score)
        elif save_max_len_only:  # automatically record episode if the episode hit max length
            if len(ep_obs) >= env._max_episode_steps:
                print("episode:", ep)
                ep += 1
                obs.append(np.vstack(ep_obs))
                acs.append(np.vstack(ep_acs))
                rews.append(ep_rews)
                ep_rets.append(score)
        else:  # save all episodes
            print("episode:", ep)
            ep += 1
            obs.append(np.vstack(ep_obs))
            acs.append(np.vstack(ep_acs))
            rews.append(ep_rews)
            ep_rets.append(score)

        print("Length of eps obs:", len(ep_obs))
        ep_lens.append(len(ep_obs))
        if ep >= num_trajs:
            break

    r = np.array(tot_rew)
    print('mean {} std {}'.format(np.mean(r), np.std(r)))

    if remove_time_feat:
        obs = np.array(obs)[:, :, :-1]  # time feat is last idx
    if save_path:
        np.savez(save_path, obs=np.array(obs), acs=np.array(acs),
                 rews=np.array(rews), ep_rets=np.array(ep_rets))
    print("Warning: only returning max length episodes")
    return obs, acs, ep_rets, ep_lens


def sample_subopt_demos(env_id, num_trajs, task_name):
    max_train_ts = int(params["training"][env_id.replace(
        "-v2", "").lower()]["max_timesteps"])
    manual_ts = [5000, 20000]
    ts_list = [f"ts={i}_seed=982" for i in manual_ts]
    # ts_list = [f"ts={i}_seed=982" for i in range(max_train_ts // 5 - 1000, max_train_ts, max_train_ts // 5)]

    for model_name in [*ts_list]:  # "best_model" # add None to list to sample randomly
        saved_traj_path = sample(
            env_id, task_name=task_name, num_trajs=num_trajs, model_name=model_name)
        print(
            f"Sampled trajectories for {model_name} can be found at {saved_traj_path}")
        print(
            f"Converting sampled trajectories for {model_name} to raw state only")
        get_joint_angles(env_id=env_id,
                         expert_demo_path=saved_traj_path,
                         paths=paths,
                         mode="best" if model_name == "best_model" else "median"
                         )


def sample(env_id, task_name, num_trajs=100, model_name="best_model", results_dir=paths['rl_demo']['results_dir']):
    '''Note that sampled trajectories from SAC uses deterministic actions'''
    run_id = 1
    if model_name == "best_model":
        model = SAC.load(
            f"{results_dir}/{task_name}/checkpoint/sac_{env_id}_{run_id}/{model_name}.zip")
    elif model_name is not None:
        model = SAC.load(
            f"{results_dir}/{task_name}/checkpoint/sac_{env_id}_{run_id}/sac_{env_id}/{model_name}.zip")
    else:
        model = None

    env = TimeFeatureWrapper(gym.make(env_id))

    save_dir = os.path.join(
        paths["train_sample_paths"]["sample_save_dir"])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if model_name is None:
        save_path = os.path.join(save_dir, f'random.{env_id}.seed=None')
    else:
        save_path = os.path.join(
            save_dir, f'sac-{model_name.split("_")[0]}.{env_id}.seed=None')

    generate_data(env, env_id, num_trajs, model=model,
                  manual=False, visual=False, save_path=save_path,
                  remove_time_feat=True,
                  random=True if model is None else False)
    env.close()
    return save_path


if __name__ == '__main__':
    args = argparser()
    set_global_seeds(args.seed)

    task_name = "time_feature_rl"

    if args.task == 'train':
        env = TimeFeatureWrapper(gym.make(args.env_id))
        eval_env = TimeFeatureWrapper(gym.make(args.env_id))
        train(args.env_id, env, eval_env, task_name=task_name,
              save_checkpoints=["best", "all"])

    elif args.task == 'sample_best':
        saved_traj_path = sample(args.env_id, task_name, num_trajs=20, model_name="best_model"
                                 )
        print(
            f"Sampled trajectories for model can be found at {saved_traj_path}")
        print("Converting sampled trajectories to raw state only")
        # get_joint_angles(env_id=args.env_id,
        #                  expert_demo_path=saved_traj_path,
        #                  paths=paths,
        #                  mode="median"
        #                  )
    elif args.task == 'sample_all':
        sample_subopt_demos(args.env_id, num_trajs=100, task_name=task_name)
