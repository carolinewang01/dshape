import os
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

from poc.gridworld_env import GridWorld
from poc.env_wrapper import DemoWrappedGridworld, TimeFeatureWrapper
from poc.agent import Q_Agent, SarsaAgent
from poc.poc_utils import extend_demo
from utils.helpers import str2bool
from utils.load_confs import load_parameters, load_paths

paths = load_paths()
params = load_parameters()


def argparser():
    parser = argparse.ArgumentParser("Running gridworld experiments")
    parser.add_argument('--n_trials', type=int, default=5)
    parser.add_argument('--trial_idx', type=int, default=0,
                        help="only used if n-trials is 1")
    parser.add_argument('--total_train_ts', type=int, default=250000)
    parser.add_argument('--max_steps_per_episode', type=int, default=500)
    # env settings
    parser.add_argument('--gridworld_size', type=int, default=20)
    parser.add_argument('--reward_base_value', type=int, default=-1)
    parser.add_argument('--gridworld_goal', type=int, nargs="+", default=None)

    # algo params
    parser.add_argument('--reward_type', type=str, default=None)
    parser.add_argument('--termphi0', type=str2bool, default=True)
    # load from saved
    parser.add_argument('--load_saved_agent', type=str, default=None)
    parser.add_argument('--load_agent_path', type=str, default=None)

    # q-learning params
    parser.add_argument('--init_value', type=int, default=0,
                        help="q table initial value")
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--use_buffer', type=str2bool, default=True)
    parser.add_argument('--buffer_size', type=int, default=5000)
    parser.add_argument('--updates_per_step', type=int, default=20)
    parser.add_argument('--time_feat', type=str2bool, default=True)
    # demo details
    parser.add_argument('--demo_style', type=str,
                        choices=["lower", "middle", "upper"], default="lower")
    parser.add_argument('--demo_extend_type', type=str, default="extend_last")
    parser.add_argument('--demo_goal', type=int, nargs="+", default=None)

    # imitation algo params
    parser.add_argument('--state_aug', type=str2bool, default=False)
    parser.add_argument('--relabel', type=str2bool, default=False)
    parser.add_argument('--n_sampled_goal', type=int, default=3)
    parser.add_argument('--rew_coef', type=float, default=1.0) 
    parser.add_argument('--dist_scale', type=float, default=1.0)
    # logging stuff
    parser.add_argument('--eval_interval', type=int, default=2500)
    parser.add_argument('--vis_perf', type=str2bool, default=False)
    parser.add_argument('--show_progress', type=str2bool,
                        default=False, help="tqdm progress bar")
    parser.add_argument('--save_log', type=str2bool,
                        default=False, help="save trial information + rewards")
    parser.add_argument('--save_policy', type=str2bool,
                        default=False, help="save q-table")
    parser.add_argument('--save_steps', type=int, nargs="+",
                        default=(), help="save q-table at episodes specified in list")
    parser.add_argument('--savedir_name', type=str, default=None)

    return parser.parse_args()


def make_env(gridworld_size, gridworld_goal=None, reward_base_value=-1,
             reward_type=None, rew_coef=1, dist_scale=1,
             demo_style=None, demo_goal=None,
             demo_extend_type=None, demo_extend_num=None,
             negate_potential=False, termphi0=True,
             state_aug=False, time_feat=True,
             max_steps_per_episode=None):
    """Create env and add all necessary wrappers"""
    env = GridWorld(size=gridworld_size, goal=gridworld_goal,
                    reward_base_value=reward_base_value,
                    max_steps=max_steps_per_episode)
    _, opt_reward = env.generate_solution(goal=env.goal, style="lower")

    if time_feat:
        env = TimeFeatureWrapper(env, max_steps_per_episode)
    if reward_type in ["pbrs_demo", "sbs", "manhattan"] or state_aug:
        base_demo, _ = env.generate_solution(goal=demo_goal,
                                             style=demo_style)
        # TODO: make default extension behavior repeating the last state
        demo = extend_demo(base_demo, max_steps_per_episode,
                           demo_extend_type, demo_extend_num)

        env = DemoWrappedGridworld(env, demo=demo,
                                   reward_type=reward_type,
                                   rew_coef=rew_coef, dist_scale=dist_scale,
                                   negate_potential=negate_potential,
                                   state_aug=state_aug,
                                   termphi0=termphi0)
    return env, opt_reward


def get_state_aug_value(value_table, demo, max_steps_per_episode):
    '''
    Compute 2d value function from 2d state-augmented demo
    '''
    values = []
    for t in range(max_steps_per_episode):
        values.append(value_table[:, :, t, demo[t][0], demo[t][1]])
    values = np.array(values).mean(0)
    return values


# from memory_profiler import profile
# @profile
def run_agent(gridworld_size, gridworld_goal=None,
              reward_base_value=-1,
              total_train_ts=250000, max_steps_per_episode=500,
              epsilon=0.05, alpha=0.1,
              init_value=0,
              reward_type=None, rew_coef=1, dist_scale=1,
              demo_style=None, demo_goal=None,
              demo_extend_type="extend_last", demo_extend_num=None,
              negate_potential=False, termphi0=True,
              state_aug=False, time_feat=True,
              relabel=False, n_sampled_goal=2,
              use_buffer=True, buffer_size=1000,
              updates_per_step=10,
              eval_interval=5, n_trials=5, trial_idx=0,
              load_saved_agent=None, load_agent_path=None,
              vis_perf=False, show_progress=True,
              save_policy=True, save_log=True, savedir_name=None, save_steps=[]
              ):
    """
    demo_style: specify whether to 
    plot_perf: plots reward curves averaged over n_trials
    vis_value_fcn: plots value function average across all agents trained
    """
    expt_args = locals()
    seeds = params["training"]["seeds"]
    env, opt_rew = make_env(gridworld_size, gridworld_goal=gridworld_goal, 
                            reward_base_value=reward_base_value,
                            reward_type=reward_type, rew_coef=rew_coef, dist_scale=dist_scale,
                            demo_style=demo_style, demo_goal=tuple(
                                demo_goal) if demo_goal is not None else None,
                            demo_extend_type=demo_extend_type, demo_extend_num=demo_extend_num,
                            negate_potential=negate_potential, termphi0=termphi0,
                            state_aug=state_aug, time_feat=time_feat,
                            max_steps_per_episode=max_steps_per_episode)

    trial_rewards = {"env": {"ret": []},
                     "shaped": {"ret": []},
                     }

    savedir_base = os.path.join(
        paths["rl_demo"]["gridworld_results_dir"], savedir_name)

    agents = []
    for i in range(n_trials):
        if n_trials > 1:
            trial_idx = i

        savedir = os.path.join(savedir_base, f"trial={trial_idx}")
        if (save_policy or save_log):
            assert savedir_name is not None
            if not os.path.exists(savedir):
                os.makedirs(savedir)

        # set trial seed
        np.random.seed(seeds[i])

        env.reset()
        agent = Q_Agent(env, epsilon=epsilon, alpha=alpha,
                        init_value=init_value,
                        total_train_ts=total_train_ts,
                        max_steps_per_episode=max_steps_per_episode,
                        eval_interval=eval_interval,
                        reward_modified=False if reward_type is None else True,
                        use_buffer=use_buffer, buffer_size=buffer_size,
                        # can only be true if state_aug and potential_shaping is also true
                        relabel=relabel, n_sampled_goal=n_sampled_goal,
                        updates_per_step=updates_per_step, show_progress=show_progress,
                        save_policy=save_policy, 
                        save_steps=save_steps
                        )
        if load_saved_agent:
            agent.load_agent(load_path=load_agent_path)
        env_reward_per_episode, shaped_reward_per_episode, eval_ts = agent.train(
            savedir)  # these are same if no shaping present

        if save_log:
            np.savez_compressed(os.path.join(savedir, "logs.npz"),
                                expt_args=expt_args, opt_rew=opt_rew,
                                env_ret=env_reward_per_episode,
                                shaped_ret=shaped_reward_per_episode,
                                eval_ts=eval_ts
                                )
        
        # TODO: UPDATE CODE BASED ON TRIAL REWARDS TO PLOT BASED OFF OF EVAL TS
        trial_rewards["env"]["ret"].append(env_reward_per_episode)
        trial_rewards["shaped"]["ret"].append(shaped_reward_per_episode)
        agents.append(agent)


    for rew_name, rew_stats_dict in trial_rewards.items():
        rew_stats_dict["mean_ret"] = np.mean(
            np.array(rew_stats_dict["ret"]), axis=0)
        rew_stats_dict["std_ret"] = np.std(
            np.array(rew_stats_dict["ret"]), axis=0)

    # plot simple learning curve
    if vis_perf:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].axhline(opt_rew, label="optimal reward", linestyle="--")

        # simple learning curves
        eval_ep_idx = [
            i * eval_interval for i in range(trial_rewards["env"]["mean_ret"].shape[0])]

        for rew_name, rew_stats_dict in trial_rewards.items():
            mean, std = rew_stats_dict["mean_ret"], rew_stats_dict["std_ret"]
            ax[0].plot(eval_ep_idx, mean, label=rew_name)
            ax[0].fill_between(eval_ep_idx, mean - std, mean + std,
                               alpha=0.5  # , edgecolor=color, facecolor=color
                               )
        ax[0].set_title("learning curve")
        ax[0].legend()

        # visualize average trained agent's value function
        values_all = []
        for agent in agents:
            value_fcn = np.max(agent.q_table, axis=-1)
            if time_feat:
                if state_aug:
                    value_fcn = get_state_aug_value(
                        value_fcn, env.demo, max_steps_per_episode)
                else:
                    # average across all timesteps
                    value_fcn = np.mean(value_fcn, axis=-1)
            values_all.append(value_fcn)

        values_all = np.mean(values_all, axis=0)
        # mask values that have not been changed from 0
        masked_values = np.ma.masked_where(
            values_all == init_value, values_all)
        # use "Greens_r" for reversed colormap
        cmap = copy.copy(matplotlib.cm.get_cmap('Greens'))
        cmap.set_bad(color='yellow')
        ax[1].imshow(masked_values, cmap=cmap, interpolation='none')
        ax[1].set_title("value fcn")

        # vis state visitation of trained agents
        states_all = []
        for agent in agents:
            env.reset()
            for i in range(20):  # run 20 episodes test
                _, _, _, states = agent.run_episode(test_mode=True)
                states_all += states
        state_visitation = env.unwrapped.traj_on_map(states_all)
        # mask values that have not been changed from 0
        masked_visitation = np.ma.masked_where(
            state_visitation == 0., state_visitation)
        cmap = copy.copy(matplotlib.cm.get_cmap('Blues'))
        cmap.set_bad(color='yellow')
        ax[2].imshow(masked_visitation, cmap=cmap, interpolation='none')
        ax[2].set_title("state visitation")

        fig.suptitle(f"{gridworld_size}x{gridworld_size} GW, Shaping={reward_type}/style={demo_style}/goal={demo_goal},\nState Aug={state_aug}, Relabel={relabel}, n_sampled_goal={n_sampled_goal}, epsilon={epsilon}")
        fig.tight_layout()
        plt.show()

    del agents
    del env
    # return learning curve info
    return trial_rewards, opt_rew


def plot_expts(learning_info: dict, eval_interval: int, opt_rew: float):
    # plot learning curves
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # simple learning curve
    for exp_name, exp_reward_dict in learning_info.items():
        trial_rewards_mean, trial_rewards_std = exp_reward_dict[
            "env"]["mean_ret"], exp_reward_dict["env"]["std_ret"]
        eval_ep_idx = [
            i * eval_interval for i in range(trial_rewards_mean.shape[0])]
        ax.plot(eval_ep_idx, trial_rewards_mean, label=exp_name)
        ax.fill_between(eval_ep_idx, trial_rewards_mean - trial_rewards_std, trial_rewards_mean + trial_rewards_std,
                        alpha=0.5  # , edgecolor=color, facecolor=color
                        )
    ax.axhline(opt_rew, label="optimal reward", linestyle="--")
    ax.set_title("learning curve")
    ax.set_ylabel("return")
    ax.set_xlabel("timesteps")

    plt.legend()
    plt.show()


if __name__ == '__main__':
    ######## DEBUG #############
    # default_params = {
    #     # global exp settings
    #     "n_trials": 1, # n_trials per call to run_agent
    #     "eval_interval": 1000,
    #     "max_steps_per_episode": 500,
    #     "time_feat": True,
    #     # gridworld settings
    #     "reward_base_value": 0,
    #     # algo params
    #     "epsilon": 0.2,
    #     "alpha": 0.1,
    #     "init_value": 0,
    #     "use_buffer": True,
    #     "buffer_size": 5000,
    #     "updates_per_step": 20,
    #     "n_sampled_goal": 3,
    #     # save settings
    #     "save_log": False,
    #     "save_policy": True,
    #     "vis_perf": False,
    #     "show_progress": True,
    # }
    # exp_params = {
    #     "dshape": {
    #         "reward_type": "pbrs_demo",
    #         "demo_style": "lower",
    #         "state_aug": True,
    #         "relabel": True,
    #         "save_steps": [50, 70]
    #                   },
    # }

    # world_dependent_params = {10: {"total_train_ts": 2000000},
    #               20: {"total_train_ts": 15000},
    #               30: {"total_train_ts": 30000}}
    # gridworld_size = 10
    # savedir_name = "dshape_debug"
    # all_params = {
    #               **default_params,
    #               **exp_params["dshape"],
    #               "gridworld_size": gridworld_size,
    #               "total_train_ts": world_dependent_params[gridworld_size]["total_train_ts"],
    #               "savedir_name": savedir_name,
    #               "trial_idx": 0
    #               }
    # run_agent(**all_params)
    ######### RUN ON CLUSTER #############
    args = argparser()
    run_agent(**vars(args))
