import sys
import os
import shutil
import argparse

from reward_shaping.sac_param_searcher import SACParamSearcher as ParamSearcher
from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool

params = load_parameters()
paths = load_paths()

rew_param_grid = {
                  # huber params
                  'imit_rew_coef': [0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1],
                  'alpha': [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1],
                  'gamma': [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 1]
                  # sbs params
                  # 'dist_scale': [0.01, 0.2, 0.4, 0.6, 0.8, 0.99],
                  # 'tau': [0.01, 0.2, 0.4, 0.6, 0.8, 0.99]
                 }

def argparser():
    parser = argparse.ArgumentParser("Running hyperparameter search over SAC w/demonstration+reward shaping")
    parser.add_argument('--env_id', type=str, choices=["Reacher-v2", "Swimmer-v2", "Ant-v2", "Walker2d-v2", "Hopper-v2", "HalfCheetah-v2"], 
                        default=None)
    parser.add_argument('--job_name', type=str, default='hyperparam_huber')
    parser.add_argument('--task', type=str, help="tasks offered in sac_reward.py", default="huber+env_time_feat_rl")
    parser.add_argument('--email_on_error', type=str2bool, help="if true send emails to me", default=False)
    return parser.parse_args()


def wipe_and_recreate_path(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == '__main__':
    args = argparser()
    results_dir = paths['rl_demo']['hyperparam_results_dir']
    conf_dir = "../confs/reward_shaping/"
    condor_log_dir = os.path.join(results_dir, f"condor_logs_{args.job_name}_{args.env_id}")

    # wipe_and_recreate_path(conf_dir)
    wipe_and_recreate_path(condor_log_dir)

    condor_extra_contents = ""
    if args.email_on_error:
        condor_extra_contents += "Notify_user = caroline.l.wang@utexas.edu\n"
        condor_extra_contents += "Notification = Error\n"


    exec_kwargs = {"env_id": f"{args.env_id}", # eventually put this into a for loop
                   "task": args.task,
                   "displace_t": 1,
                   "demo_algo": params["her_demo"]["demo_algo"][args.env_id.replace("-v2", "").lower()]["optimal"],
                   "demo_level": "optimal",
                   # params_file is filled in by the ParamSearcher operating over the rew_param_grid
                   } 

    param_searcher = ParamSearcher(cmd="/u/clw4542/research/rl_ifo_mujoco/src/reward_shaping/sac_reward.py", 
                                   conf_dir=conf_dir, run_local=False, param_dict=rew_param_grid, 
                                   job_name=f"{args.job_name}_{args.env_id}", 
                                   results_dir=results_dir,
                                   algo_name="sac",
                                   random_sampling=True, search_budget=50,
                                   # random_sampling=False,
                                   trials_per_param=1,
                                   condor_log_dir=condor_log_dir, 
                                   condor_extra_contents=condor_extra_contents, 
                                   require_gpu=True)
    param_searcher.run(exec_kwargs)
