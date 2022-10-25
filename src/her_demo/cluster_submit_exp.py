import time
import os
import glob
import argparse

from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool
from utils.cluster_helpers import submit_to_slurm, submit_to_condor

algo = "td3"
params = load_parameters()
paths = load_paths()

mem_dict = {'Reacher-v2': 20,
            'Swimmer-v2': 30,
            'Ant-v2': 64,
            'Hopper-v2': 30,  # 64, #20
            'HalfCheetah-v2': 48,
            'Walker2d-v2': 64
            }

def argparser():
    parser = argparse.ArgumentParser("Submitting experiments to cluster")
    parser.add_argument('--expt', type=str,
                        choices=['dshape', 'baselines'])
    parser.add_argument('--cluster', type=str,
                        choices=['slurm', 'condor', None], default="slurm")
    parser.add_argument('--overwrite', type=str2bool, default=True)

    return parser.parse_args()


def get_savefile_dirs(results_dir, env_id, task_log_name, run_id=None):
    '''Returns a list of savefile directories for a given env_id, expt, and run_id
    '''
    savefile_name = f"{algo}_{env_id}"
    if run_id is not None:
        savefile_name += f"_{run_id+1}"
    savefile_base = os.path.join(
        results_dir, task_log_name, "log", f"{savefile_name}*")
    savefile_dirs = glob.glob(savefile_base, recursive=True)
    return savefile_dirs


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


def run_her_expts(results_dir, cluster, trials_per_param=5, overwrite=False):
    # memory requirements discovered thru trial and error
    for env_id in [
        'Reacher-v2',
        'Swimmer-v2',
        'Ant-v2',
        'Hopper-v2',
        'HalfCheetah-v2',
        # 'Walker2d-v2'
    ]:
        for gss in [
            # 'episode',
            'random',
            # 'future',
            # 'next_state',
            # 'next_state_and_ep',
            # 'episode_nearest',
            # 'episode_nearest_future'
        ]:
            for rew in [
                # 'env',
                'potential_dense',
                # 'huber+env'
            ]:
                n_sampled_goal = 10
                if rew == 'env':
                    n_sampled_goal = 0

                rew_delay = params["training"][env_id.replace("-v2", "").lower()]["rew_delay"]
                sin_cos_repr = params["training"][env_id.replace("-v2", "").lower()]["sin_cos_repr"]
                learning_starts = params[algo][env_id.replace("-v2", "").lower()]["learning_starts"]


                demo_dict = params["her_demo"]["demo_algo"][env_id.replace("-v2", "").lower()]
                for demo_level in [
                                   "optimal", 
                                   # "medium", 
                                   # "worst", 
                                   # "random"
                                   ]:
                    demo_algo = demo_dict[demo_level]
                    expt_params = {'algo': algo,
                                   'env_id': env_id,
                                   'raw': True,
                                   'rew': rew,
                                   'n_sampled_goal': n_sampled_goal,
                                   'time_feat': True,
                                   'gss': gss,
                                   'displace_t': 1,
                                   'demo_algo': demo_algo,
                                   'demo_level': demo_level,
                                   'sparse_rew': False,
                                   'sin_cos_repr': sin_cos_repr
                                   }

                    task_log_name = f"{params['her_demo']['expt_name']}_algo={algo}_rew={expt_params['rew']}_raw={expt_params['raw']}_gss={expt_params['gss']}_n-goal={expt_params['n_sampled_goal']}_demo={expt_params['demo_level']}_sparse-rew={expt_params['sparse_rew']}_sin-cos-repr={expt_params['sin_cos_repr']}_learning-starts={learning_starts}"

                    if cluster == "condor":
                        submit_to_condor(env_id, exec_cmd="/u/clw4542/research/rl_ifo_mujoco/src/her_demo/run_dshape.py",
                                         results_dir=results_dir, job_name=f'{env_id}_{task_log_name}',
                                         expt_params=expt_params, num_trials=trials_per_param)
                    elif cluster == "slurm":
                        submit_to_slurm(env_id, exec_cmd="python -m her_demo.run_dshape",
                                        results_dir=results_dir, job_name=f'{env_id}_{task_log_name}',
                                        expt_params=expt_params, num_trials=trials_per_param,
                                        partition="dgx", # if overwrite else "dgx",
                                        mem=mem_dict[env_id]
                                        )
                    else: # test mode
                        print("No jobs submitted.")


def run_baseline_expts(results_dir, cluster, trials_per_param=5, overwrite=False):
    # TODO: add sin cos repr support
    displace_t = 1
    for env_id in [
        'Reacher-v2',
        'Swimmer-v2',
        'Ant-v2',
        'Hopper-v2',
        'HalfCheetah-v2',
        # 'Walker2d-v2'
    ]:
        demo_dict = params["her_demo"]["demo_algo"][env_id.replace("-v2", "").lower()]

        task_dict = {
            'time_feature_rl': f'time_feature_rl',
            # 'raw_demo_time_feat_rl': f'raw_demo_time_feat_rl',
            'potential_dense_time_feat_rl': f'potential_dense_time_feat_rl',
            # 'huber+env_time_feat_rl': f'huber+env_time_feat_rl',
            # 'sbs_time_feat_rl': f'sbs_time_feat_rl'
        }
        
        def send_expts(trials_per_param, expt_params, task_log_name):
            """Send baseline experiments to cluster
            """
            if cluster == "condor":
                submit_to_condor(env_id, exec_cmd="/u/clw4542/research/rl_ifo_mujoco/src/rl_demo/run_rl_demo.py",
                                 results_dir=results_dir, job_name=f'{env_id}_{task_log_name}',
                                 expt_params=expt_params, num_trials=trials_per_param
                                 )
            elif cluster == "slurm":
                submit_to_slurm(env_id, exec_cmd="python -m rl_demo.run_rl_demo",
                                results_dir=results_dir, job_name=f'{env_id}_{task_log_name}',
                                expt_params=expt_params, num_trials=trials_per_param,
                                partition="dgx",
                                mem=mem_dict[env_id])
            else: # test mode
                print("No jobs submitted.")

        for task, task_log_name in task_dict.items():
            expt_params = {'algo': algo,
                           'env_id': env_id,
                           'task': task,
                           'displace_t': displace_t,
                           'sparse_rew': False
                           }

            if task == 'time_feature_rl':
                task_log_name = f"{task_log_name}_sparse-rew={expt_params['sparse_rew']}"
                send_expts(trials_per_param, expt_params, task_log_name)
            else:
                for demo_level in ["optimal", 
                                   # "medium", 
                                   # "worst", 
                                   # "random"
                                   ]:
                    demo_algo = demo_dict[demo_level]
                    expt_params = {**expt_params,                        
                                   'demo_algo': demo_algo,
                                   'demo_level': demo_level
                                   }
                    task_log_name = f"{task_log_name}_demo={expt_params['demo_level']}_sparse-rew={expt_params['sparse_rew']}"
                    send_expts(trials_per_param, expt_params, task_log_name)


if __name__ == '__main__':
    args = argparser()
    results_dir = paths['rl_demo']['results_dir']

    if args.expt == "dshape":
        run_her_expts(results_dir, 
                      cluster=args.cluster,
                      trials_per_param=5,
                      overwrite=args.overwrite)
    elif args.expt == "baselines":
        run_baseline_expts(results_dir, 
                           cluster=args.cluster, 
                           trials_per_param=5,
                           overwrite=args.overwrite)