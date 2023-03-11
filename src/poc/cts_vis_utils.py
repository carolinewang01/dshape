import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from poc.run_cts_agent import generate_demo, make_env
from her_demo.her_custom import CustomHER as HER
from expert_traj.expert_algos.sac import SAC
from expert_traj.expert_algos.td3 import TD3

from utils.stable_baselines_helpers import evaluate_policy
from utils.vis_utils import combine_eval_logs


# generate list of colors
def get_color_list(cmap_name:str ="tab10"):
    cmap = matplotlib.cm.get_cmap(cmap_name)
    color_list = []
    for i in range(cmap.N):
        rgba = cmap(i)
        # rgb2hex accepts rgb or rgba
        color_list.append(matplotlib.colors.rgb2hex(rgba))
    return color_list

def vis_state_traj(state_trajs:dict, goal, env_size, 
                    ax,
                    ckpt_name,
                    color_dict,
                    expert_demo=None,
                    show_rect=False, 
                   ):
    if show_rect:
        ax.add_patch(Rectangle((-env_size/2, -env_size/2), # lower left corner
                     env_size, env_size, # side lengths
                     edgecolor = 'lightgray',
                     facecolor = 'none',
                     fill=False,
                     lw=1)
                    )
    for label, traj in state_trajs.items():
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:,1], 
                marker="o", markersize=4,
                color=color_dict[label],
                label=label)
    # plot exp demo
    if expert_demo is not None:
        ax.plot(expert_demo[:, 0], expert_demo[:, 1], 
                marker="s", markersize=4,
                linestyle="dashed", 
                label="demo")

    # plot goal
    goal_box = 0.01
    goal_pt = ax.plot(goal[0], goal[1], marker="*", label="goal")
    ax.add_patch(Rectangle((goal[0]-goal_box, goal[1]-goal_box), # lower left corner
#                  0.05, 0.05, # side lengths
                 goal_box*2, goal_box*2, # side lengths
                 edgecolor = goal_pt[0].get_color(),
                 fill=True,
                 alpha=0.4,
                 lw=0.5)
                )
    # ax.set_xlim()
    ax.set_title(f"{env_size}x{env_size} Pointworld, Ckpt={ckpt_name}")
    ax.legend()

def visualize(base_path, expt_dict, algo_name="sac",
              ckpt_name="best_model", 
              run_ids=[1,2,3,4,5], 
              show=True, return_fig=False):
    
    # map colors to expt names
    color_list = get_color_list()
    color_dict = {expt_name: color_list.pop(0) for expt_name in expt_dict.keys()}
    
    fig, ax = plt.subplots(1,3, figsize=(12,3))
    ### plot visitation trajectories
    state_trajs = {}

    for expt_name, expt_settings in expt_dict.items():
        expt_params = expt_settings["expt_params"]
        task_vars = expt_settings["task_vars"]
        vis_id = 1
        if len(run_ids)==1:
            vis_id = run_ids[0]
        if ckpt_name == "best_model":
            model_path = os.path.join(base_path, expt_settings["task_log_name"], "checkpoint", f"{algo_name}_pointworld_{vis_id}", f"best_model.zip")
        else:
            model_path = os.path.join(base_path, expt_settings["task_log_name"], "checkpoint", f"{algo_name}_pointworld_{vis_id}", f"{algo_name}_pointworld", f"{ckpt_name}.zip")

        eval_env, exp_demo, exp_ret = make_env(task_vars=task_vars, 
                               env_size=expt_params["env_size"],
                               goal=expt_params["goal"], 
                               max_episode_steps=expt_params["max_episode_steps"], 
                               rew_base_value=expt_params["rew_base_value"] if "rew_base_value" in expt_params else -1,
                               eval_mode=True)

        if  "her" in task_vars:
            model = HER.load(model_path, env=None)
            her_policy = True
        elif algo_name == "sac":
            model = SAC.load(model_path, env=None)
            her_policy = False
        elif algo_name == "td3":
            model = TD3.load(model_path, env=None)
            her_policy = False
        # get trained agent trajectory
        policy_state_traj = evaluate_policy(model, eval_env, 
                                        n_eval_episodes=1, 
                                        deterministic=True, 
                                        return_episode_rewards=False,
                                        return_episode_obses=True,
                                        her_policy=her_policy
                                       )
        state_trajs[expt_name] = policy_state_traj[0]

    eval_env.close()
    vis_state_traj(state_trajs, 
                    goal=expt_params["goal"], 
                    env_size=expt_params["env_size"],
                    ax=ax[2], 
                    ckpt_name=ckpt_name,
                    color_dict=color_dict,
                    expert_demo=exp_demo, 
                    show_rect=True)
    
    ### plot learning curves and goal dists
    for expt_name, expt_settings in expt_dict.items():
        log_base = os.path.join(base_path, expt_settings["task_log_name"], "log", f"{algo_name}_pointworld")
        ts, mean_res, std_res, _, _, mean_goal_dists, std_goal_dists = combine_eval_logs(log_base, 
                                                              normalize=False, 
                                                              run_ids=run_ids, 
                                                              max_ts=3e+6)
        ax[0].plot(ts, mean_res, label=expt_name, color=color_dict[expt_name])
        ax[0].fill_between(ts, mean_res-std_res, mean_res+std_res, alpha=0.5, color=color_dict[expt_name] )
        if np.size(mean_goal_dists) != 0:
            ax[1].plot(ts, mean_goal_dists, label=expt_name, color=color_dict[expt_name])
            ax[1].fill_between(ts, mean_goal_dists-std_goal_dists, 
                               mean_goal_dists+std_goal_dists, alpha=0.5, color=color_dict[expt_name] )
        
    ax[0].axhline(exp_ret, label="opt return", linestyle="dashed")
    ax[0].set_title("Learning Curves")
#     ax[0].legend()
    ax[1].set_title("Cumulative Goal Distances")
#     ax[1].legend()

    ### show plot
    if show:
        plt.legend(bbox_to_anchor=(1.0, 0.8))
        plt.show()
    if return_fig:
        return fig
    return