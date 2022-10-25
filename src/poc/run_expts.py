import time
import os
import glob
import argparse

from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool
from utils.cluster_helpers import submit_to_slurm, submit_to_condor
from poc.run_agent import run_agent

paths = load_paths()

def check_single_dir(savefile_dir):
    if not os.path.exists(savefile_dir):  # directory doesn't even exist
        return False
    else:
        if not os.listdir(savefile_dir):  # directory exists but is empty
            return False
    return True

def count_nonempty_dirs(savefile_dirlist):
    '''
    Checks number of nonempty and/or existing directories in the list of directories
    '''
    done = []
    for savefile_dir in savefile_dirlist:
        done.append(check_single_dir(savefile_dir))
    return sum(done)

def run_gridworld_expts(trials=30, overwrite=False):
    default_params = {
        # global exp settings
        "n_trials": 1, # n_trials per call to run_agent
        "max_steps_per_episode": 500,
        "time_feat": True,
        # gridworld settings
        "reward_base_value": -1,
        # algo params
        "epsilon": 0.2,
        "alpha": 0.1,
        "init_value": 0,
        "use_buffer": True,
        "buffer_size": 5000,
        "updates_per_step": 20,
        "n_sampled_goal": 3,
        # save settings
        "save_log": True,
        "save_policy": True,
        "vis_perf": False,
        "show_progress": True, 
    }

    exp_settings = {
        "q-learning_termphi0=False": {
            "termphi0": False
        },
        "dshape_termphi0=False": {
            "reward_type": "pbrs_demo",
            "demo_style": "lower",
            "state_aug": True, 
            "relabel": True,
            "termphi0": False 
                      },
        "pbrs_alone_termphi0=False": {
            "reward_type": "pbrs_demo",
            "demo_style": "lower",
            "termphi0": False
        },
        "pbrs+state-aug_termphi0=False":{
            "reward_type": "pbrs_demo", 
            "demo_style": "lower",
            "state_aug": True,
            "termphi0": False
        },
        "ridm-state-aug-alone_termphi0=False": {
            "demo_style": "lower",
            "state_aug": True,
            "termphi0": False
        },
        "sbs_termphi0=False": {
            "reward_type": "sbs",
            "demo_style": "lower",
            "state_aug": False,
            "rew_coef": 1,
            "dist_scale": 10,
            "termphi0": False
        },
        "manhattan_termphi0=False": {
            "reward_type": "manhattan",
            "demo_style": "lower",
            "state_aug": False,
            "rew_coef": 1.0,
            "termphi0": False
        },

        # 20 x 20 GRIDWORLD ONLY EXPTS
        # # coeff study
        # "dshape": {
        #     "reward_type": "pbrs_demo",
        #     "demo_style": "lower",
        #     "state_aug": True, 
        #     "relabel": True ,
        #     "termphi0": False,
        # },

        # "manhattan": {
        #     "reward_type": "manhattan",
        #     "demo_style": "lower",
        #     "state_aug": False,
        #     "termphi0": False,
        # },

        # "pbrs_alone": {
        #     "reward_type": "pbrs_demo",
        #     "demo_style": "lower",
        #     # "termphi0": False
        # },

        # EXPTS TO GEN STATE VISITATION PLOTS
        # "dshape_demo-goal=19_grid-goal=19": {
        #     "gridworld_goal": (0, 19),
        #     "demo_goal": (0, 19),
        #     "reward_type": "pbrs_demo",
        #     "demo_style": "lower",
        #     "state_aug": True, 
        #     "relabel": True 
        # },
        # "pbrs+state-aug_demo-goal=19_grid-goal=19":{
        #     "gridworld_goal": (0, 19),
        #     "demo_goal": (0, 19),
        #     "reward_type": "pbrs_demo", 
        #     "demo_style": "lower",
        #     "state_aug": True,
        # },
    }

    world_dependent_params = {
                  10: {"total_train_ts": 2000000,
                       "eval_interval": 5000, # evaluation frequency in ts

                                    },  # 2000000 timesteps should be okay
                  20: {"total_train_ts": 10000000,
                      "eval_interval": 10000, # evaluation frequency in ts
                      # "save_steps": (3, 7, 20), # n_evals to save q-table at
                  }, # 5000000 timesteps should be okay
                  30: {"total_train_ts": 10000000,
                       "eval_interval": 20000, # evaluation frequency in ts
                  }
              } # 10000000 timesteps should be okay

    subopt_demo_dict = {
                10: {"subopt_step": 2},
                20: {"subopt_step": 2},
                30: {"subopt_step": 4}
    }
    results_dir = paths['rl_demo']['gridworld_results_dir']

    for gridworld_size in [
                           # 10, 
                           20, 
                           # 30
                           ]:
        for exp_name, exp_params in exp_settings.items():
            # for i in range(1, 4):

            # for i in range(4, 10):
            #     demo_goal = gridworld_size-1-subopt_demo_dict[gridworld_size]["subopt_step"]*i
            #     exp_params["demo_goal"] = (0, demo_goal)
            demo_goal = "15.0"
            exp_params["demo_goal"] = (15, 0)
            # savedir_name = f"{exp_name}_world=basic_size={gridworld_size}"
            savedir_name = f"{exp_name}_demo-goal={demo_goal}_world=basic_size={gridworld_size}"

            for trial_idx in range(trials):
                all_params = {
                              **default_params,
                              **exp_params,
                              **world_dependent_params[gridworld_size],
                              "gridworld_size": gridworld_size,
                              "savedir_name": savedir_name,
                              "trial_idx": trial_idx
                              }
                if not overwrite: # check whether log file already exists
                    save_path = os.path.join(results_dir, savedir_name, f"trial={trial_idx}")
                    logs_present = check_single_dir(save_path)
                    if logs_present: continue

                run_dir = os.path.dirname(os.path.abspath(__file__))
                submit_to_condor(env_id=f"{gridworld_size} gridworld", 
                                 exec_cmd=os.path.join(run_dir, "run_agent.py"),
                                 results_dir=results_dir, 
                                 job_name=savedir_name,
                                 expt_params=all_params, 
                                 num_trials=1,
                                 req_gpu=False)


if __name__ == '__main__':
    run_gridworld_expts(trials=30, overwrite=False)
    