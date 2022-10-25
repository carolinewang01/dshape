from collections import deque 
import copy
import os

import pandas as pd 
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import uniform_filter1d

from JSAnimation.IPython_display import display_animation
from IPython.display import display

import pytablewriter

import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.ticker as ticker

####### code used to visualize her_demo results ##########
def combine_eval_logs(log_base, 
                      normalize=True, 
                      env_id=None, expert_perf=None, random_perf=None, demo_level=None, # needed for normalization
                      run_ids=[1,2,3,4,5], max_ts=3e+6
                     ):
    results_all, ep_lens_all, goal_dists_all = [], [], []
    for run_id in run_ids:
        log_path = f"{log_base}_{run_id}/"
        # TODO: even if path doesn't exist, just print warning
        ts, results, ep_lens, goal_dists = read_eval_data(log_path, env_id=env_id,
                                                          normalize=normalize, 
                                                          expert_perf=expert_perf, random_perf=random_perf,
                                                          demo_level=demo_level,
                                                          max_ts=max_ts)
        results_all.append(results)  # shape (ts, n_evals, 1 )
        ep_lens_all.append(ep_lens)
        goal_dists_all.append(goal_dists)
        
    combi_mean = lambda x: (np.concatenate(x, axis=1)
                              .squeeze()
                              .mean(axis=1))
    combi_std = lambda x: (np.concatenate(x, axis=1)
                              .squeeze()
                              .std(axis=1))
    
    mean_results, std_results = combi_mean(results_all), combi_std(results_all)
    mean_ep_lens, std_ep_lens = combi_mean(ep_lens_all), combi_std(ep_lens_all)
    if np.count_nonzero(goal_dists_all) == 0:
        mean_goal_dists, std_goal_dists = [], []
    else:
        mean_goal_dists, std_goal_dists = combi_mean(goal_dists_all), combi_std(goal_dists_all)

    return (ts, mean_results, std_results,
            mean_ep_lens, std_ep_lens,
            mean_goal_dists, std_goal_dists)
                

def normalizer(x:np.ndarray, max_value:float, min_value:float):
    return (x - min_value) / (max_value - min_value)


def read_eval_data(log_path:str,
                   normalize=True,
                   env_id=None,
                   expert_perf:dict=None,
                   random_perf:dict=None,
                   demo_level=None,
                   max_ts=3e+6
                  ):
    eval_data = np.load(os.path.join(log_path, "evaluations.npz"))
    ts = eval_data["timesteps"]
    results = eval_data["results"]
    ep_lens = eval_data["ep_lengths"]
        
    truncated_ts = [i for i in ts if i < max_ts]
    max_idx = len(truncated_ts)
    
    goal_dist = np.array([])
    if "goal_dists" in eval_data.keys():
        goal_dist = eval_data["goal_dists"]

    if normalize:
        results = normalizer(results, 
                             max_value=expert_perf[env_id]["optimal"], 
                             min_value=random_perf[env_id]["ret"]
                           )
        goal_dist = normalizer(goal_dist,
                              max_value=random_perf[env_id][f"{demo_level}_goal-dist"],
                              min_value=0)
        
    return ts[:max_idx], results[:max_idx], \
           ep_lens[:max_idx], goal_dist[:max_idx]
    
        
def plot_algo_with_demo(logs_dict, 
                        axis,
                        env_id,
                        expert_perf:dict,
                        random_perf:dict,
                        normalize=True,
                        demo_level=None, 
                        algo=None,
                        eval_mode=False,
                        show_demo_labels=True
                       ):
    '''the result should be a tuple of 2 lists, representing the x and y coords to plot'''
    for label, result in logs_dict.items():
        time_idx, mean, std, mean_lens, std_lens, mean_goal_dist, std_goal_dist = result            
        color = algo_colors[label]
        p = axis.plot(time_idx, mean, label=label, color=color)
        std = np.array(std)
        axis.fill_between(time_idx, mean-std, mean+std,
                        alpha=0.5, edgecolor=color, facecolor=color 
                       )
    
    demo_perf_copy = copy.deepcopy(expert_perf[env_id])
    demo_perf_copy["random"] = random_perf[env_id]['ret']
    if normalize:
        normalize_return = lambda x: normalizer(x, max_value=expert_perf[env_id]["optimal"], min_value=random_perf[env_id]["ret"])
        demo_perf_copy["optimal"] = normalize_return(demo_perf_copy["optimal"])
        demo_perf_copy["medium"] = normalize_return(demo_perf_copy["medium"])
        demo_perf_copy["worst"] = normalize_return(demo_perf_copy["worst"])
        demo_perf_copy["random"] = normalize_return(demo_perf_copy["random"])

    if demo_level in ["worst", "medium", "optimal"]:
        label = f"{demo_level} demo" if demo_level != "optimal" else "best demo"
        axis.axhline(y=demo_perf_copy[demo_level], color=algo_colors[label], linestyle='--', 
                     label="demo" if show_demo_labels else None)
    elif demo_level == "all":
        axis.axhline(y=demo_perf_copy["optimal"], color=algo_colors['best demo'], linestyle='--', 
                     label='best demo' if show_demo_labels else None)
        axis.axhline(y=demo_perf_copy["medium"], color=algo_colors['medium demo'], linestyle='--', 
                     label='medium demo' if show_demo_labels else None)
        axis.axhline(y=demo_perf_copy["worst"], color=algo_colors['worst demo'], linestyle='--', 
                     label='worst demo' if show_demo_labels else None)
        axis.axhline(y=demo_perf_copy['random'], color=algo_colors['random demo'], linestyle='--', 
                     label='random demo' if show_demo_labels else None)
        
    axis.set_title(f"{env_id.replace('-v2','').lower()}", fontsize=15)
    axis.xaxis.set_major_formatter(ticker.EngFormatter())
    if not normalize:
        axis.yaxis.set_major_formatter(ticker.EngFormatter())

    
def plot_goal_dist(logs_dict, axis, env_id, algo=None):
    for label, result in logs_dict.items():
        time_idx, mean, std, mean_lens, std_lens, mean_goal_dist, std_goal_dist = result
        if len(mean_goal_dist) == 0:
            continue
        color = algo_colors[label]
        p = axis.plot(time_idx, mean_goal_dist, label=label, color=color)
#         color = p[0].get_color()
        std_goal_dist = np.array(std_goal_dist)
        axis.fill_between(time_idx, 
                          mean_goal_dist-std_goal_dist, 
                          mean_goal_dist+std_goal_dist,
                          alpha=0.5, edgecolor=color, facecolor=color 
                         )
    axis.set_title(f"{env_id.replace('-v2', '').lower()}", fontsize=15)
    axis.xaxis.set_major_formatter(ticker.EngFormatter())

    
def plot_ep_lens(logs_dict, algo, env):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    for label, result in logs_dict.items():
        time_idx, mean, std, mean_lens, std_lens, mean_goal_dist, std_goal_dist = result
        p = ax.plot(time_idx, mean_lens, label=label)
        color = p[0].get_color()
        std_lens = np.array(std_lens)
        ax.fill_between(time_idx, mean_lens-std_lens/2, mean_lens+std_lens/2,
                        alpha=0.5, edgecolor=color, facecolor=color 
                       )

    ax.legend(fontsize=14, bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.set_title(f"Ep Len of {algo.upper()} on {env}", fontsize=20)
    ax.set_xlabel("Timesteps", fontsize=16)
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.show()
    plt.close()


def newest(path):
    '''Returns full path of newest file in given path'''
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    return max(paths, key=os.path.getctime)
#################

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    display(display_animation(anim, default_mode='loop'))


def render_trajectory(env, actions):
    """Render the given sequence of actions for an environment in a Jupyter notebook
    # this currently renders the random policy 
    """
    from mujoco_py import GlfwContext
    GlfwContext(offscreen=True)  # Create a window to init GLFW.

    for t in range(100):
        # Render into buffer.     
        frames.append(env.render(mode = 'rgb_array'))
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
    env.close()
    display_frames_as_gif(frames)


def df_to_latex(df: pd.DataFrame):
    writer = pytablewriter.LatexTableWriter()
    writer.from_dataframe(df)
    writer.write_table()

    return