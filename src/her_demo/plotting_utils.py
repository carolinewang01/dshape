import os
import copy
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.load_confs import load_parameters, load_paths


paths = load_paths()
params = load_parameters()


RANDOM_PERF = {
               'Reacher-v2': {"ret": -43.1199222,
                              'optimal_goal-dist': 102.94778817288184,
                              'medium_goal-dist': 114.21570126810595,
                              'worst_goal-dist': 142.1727936522922,
                              'random_goal-dist': 139.0812864536828
                             }, 

               'HalfCheetah-v2': {"ret": -282.7423069,
                                  'optimal_goal-dist': 1316.2468678170692,
                                  'medium_goal-dist': 1321.765717231416,
                                  'worst_goal-dist': 1225.9308478261776,
                                  'random_goal-dist': 958.7524994584503
                                 }, 
               'Ant-v2': {"ret": -54.35667815,
                          'optimal_goal-dist': 1482.1148244903445,
                          'medium_goal-dist': 1512.326710452443,
                          'worst_goal-dist': 1436.6197684456367,
                          'random_goal-dist': 1378.0341266287257
                         }, 
               'Walker2d-v2': {"ret": 0.55296615,
                               'optimal_goal-dist': 35.409822850297,
                               'medium_goal-dist': 26.131589529193658,
                               'worst_goal-dist': 27.378735829954508,
                               'random_goal-dist': 26.367460484293424
                              }, 
               'Swimmer-v2': {"ret": 1.4335161,
                              'optimal_goal-dist': 2250.6821455130303,
                              'medium_goal-dist': 2366.3265878390534,
                              'worst_goal-dist': 2499.692321340913,
                              'random_goal-dist': 2076.9467436289383
                             }, 
               'Hopper-v2': {"ret":18.14677599,
                             'optimal_goal-dist': 2002.7879088181642,
                             'medium_goal-dist': 2166.11858980727,
                             'worst_goal-dist': 2126.7812898846073,
                             'random_goal-dist': 1458.8889784643468
                            }
}


def get_expert_perfs():
    demo_dict = params["her_demo"]["demo_algo"]
    env_ids=["Reacher-v2", "Swimmer-v2", "Ant-v2", "Hopper-v2", "Walker2d-v2", "HalfCheetah-v2"]

    expert_perf = {}
    expert_average_perf = {}
    
    for env_id in env_ids:
        expert_perf[env_id] = {}
        expert_average_perf[env_id] = {}
        for demo_level, demo_algo in demo_dict[env_id.replace("-v2", "").lower()].items():    
            data = np.load(f"../data/expert_joint_angle_traj/joint_angles_{demo_algo}.{env_id}.seed=None.npz")
            rew = data['rew']
            expert_perf[env_id][demo_level] = rew

            data = np.load(f"../data/expert_full_traj/{demo_algo}.{env_id}.seed=None.npz")
            expert_average_perf[env_id][demo_level] = np.mean(data['ep_rets'])
    return expert_perf, expert_average_perf

EXPERT_PERF, EXPERT_AVERAGE_PERF  = get_expert_perfs()

'''
'''

def combine_eval_logs(log_base, normalize=True, env_id=None, demo_level=None,
                      run_ids=[1,2,3,4,5], max_ts=3e+6
                     ):
    results_all, ep_lens_all, goal_dists_all = [], [], []
    for run_id in run_ids:
        log_path = f"{log_base}_{run_id}/"
        # TODO: even if path doesn't exist, just print warning
        ts, results, ep_lens, goal_dists = read_eval_data(log_path, max_ts=max_ts, 
                                                          normalize=normalize, 
                                                          env_id=env_id, demo_level=demo_level)
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
                             max_value=EXPERT_PERF[env_id]["optimal"], 
                             min_value=RANDOM_PERF[env_id]["ret"]
                           )
        goal_dist = normalizer(goal_dist,
                              max_value=RANDOM_PERF[env_id][f"{demo_level}_goal-dist"],
                              min_value=0)
        
    return ts[:max_idx], results[:max_idx], \
           ep_lens[:max_idx], goal_dist[:max_idx]
    
        
def plot_algo_with_demo(logs_dict, 
                        axis,
                        env_id,
                        algo_colors,
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
    
    demo_perf_copy = copy.deepcopy(EXPERT_PERF[env_id])
    demo_perf_copy["random"] = RANDOM_PERF[env_id]['ret']
    if normalize:
        normalize_return = lambda x: normalizer(x, max_value=EXPERT_PERF[env_id]["optimal"], min_value=RANDOM_PERF[env_id]["ret"])
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