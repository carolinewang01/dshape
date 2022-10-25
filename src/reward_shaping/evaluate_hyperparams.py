import os 
import re
import glob
import argparse
import gym 
from sklearn.model_selection import ParameterGrid

from reward_shaping.hyperparam_search import rew_param_grid
from utils.ppo2_evaluator import PPO2Evaluator
from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool

params = load_parameters()
paths = load_paths()


def argparser():
    parser = argparse.ArgumentParser("Evaluating PPO with demonstration")
    parser.add_argument('--env_id', type=str, choices=['Reacher-v2', 'Swimmer-v2', 'HalfCheetah-v2', 'Ant-v2', 'Hopper-v2', 'Walker2d-v2'], default=None)
    parser.add_argument('--expt', type=str, choices=[ 'full_demo_huber_reward_displace-t=0_demo=sac','raw_demo_huber_reward_displace-t=0_demo=sac'], default=None)
    parser.add_argument('--model_idx', type=int, default=None)
    parser.add_argument('--imit_rew_coef', type=float, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    return parser.parse_args()


def return_expt_params(env_id):
    from envs.reward_wrapper import RewardWrapper
    from utils.helpers import get_demo

    ppo2_demo_expt = {
        'raw_demo_huber_reward_displace-t=0_demo=sac': {'env_wrapper': RewardWrapper, 
                                                        'env_args': {'expert_demo': get_demo(env_id, raw=True, shuffle=False), 
                                                                      'displace_t': 0,
                                                                      'reward_type': None, # for eval only
                                                                      'env_id': env_id,
                                                                      'raw': True
                                                                      }},
        'full_demo_huber_reward_displace-t=0_demo=sac': {'env_wrapper': RewardWrapper, 
                                                         'env_args': {'expert_demo': get_demo(env_id, raw=False, shuffle=False), 
                                                                      'displace_t': 0,
                                                                      'reward_type': None, # for eval only
                                                                      'env_id': env_id,
                                                                      'raw': False
                                                                      }},
    }
    return ppo2_demo_expt


def evaluate_single_experiment(env_id, n_eval_seeds, results_dir, expt_params, expt, model_idx, imit_rew_coef, gamma, alpha):
    checkpoint_dir = os.path.join(results_dir, expt, "checkpoint", f"ppo2_{env_id}_alpha={alpha}_gamma={gamma}_imit_rew_coef={imit_rew_coef}_{model_idx}") 
    seed = params['ppo']['seeds'][int(model_idx)-1]
    job_name = f"ppo2_{env_id}_alpha={alpha}_gamma={gamma}_{imit_rew_coef}_{model_idx}_seed={seed}"

    env_wrapper = expt_params[expt]["env_wrapper"]
    env_args = expt_params[expt]["env_args"]
    env_args['reward_params'] = {"imit_rew_coef": imit_rew_coef, 
                                 "gamma": gamma, 
                                 "alpha": alpha}

    if env_wrapper: 
        env = env_wrapper(env=gym.make(env_id), **env_args,)
    else:
        env = gym.make(env_id)

    ppo2_evaluator = PPO2Evaluator(env=env, env_id=env_id, job_name=job_name, 
                                   checkpoint_dir=checkpoint_dir, 
                                   n_eval_seeds=n_eval_seeds, n_eval_rollouts=len(n_eval_seeds), 
                                   verbose=False
                                   )
    eval_data = ppo2_evaluator.evaluate(read_from_saved=True, result_prefix="model_last") # overwrite??
    return eval_data
    

def evaluate_experiments(env_id, n_eval_seeds, results_dir, expt_params):
    results = {}
    for expt in expt_params.keys():
        results[expt] = {}
        print(f"Evaluating expt {expt} for {env_id}")
        # loop through hyperparams combox 
        # for reward_params in param_grid:
        # for os.path.join(results_dir, expt, "checkpoint", f"ppo2_{env_id}_" )
            # expt_params[expt]["env_args"]["reward_params"] = reward_params 
        ckpt_dir_prefix = os.path.join(results_dir, expt, "checkpoint", f"ppo2_{env_id}_") 
        for ckpt_dir_path in glob.glob(ckpt_dir_prefix + "*", recursive=True): 
            ckpt_dirname = os.path.basename(ckpt_dir_path)
            print("CKPT DIRNAME IS ", ckpt_dirname)
            alpha = re.search( "alpha=\\d{1,3}(\\.\\d{1,3})?", ckpt_dirname).group().replace("alpha=", "")
            gamma = re.search("gamma=\\d{1,3}(\\.\\d{1,3})?", ckpt_dirname).group().replace("gamma=", "")
            imit_rew_coef = re.search("imit_rew_coef=\\d{1,3}(\\.\\d{1,3})?", ckpt_dirname).group().replace("imit_rew_coef=", "")

            if os.listdir(ckpt_dir_path): # not empty
                model_idx = ckpt_dirname.split("_")[-1] 
                eval_data = evaluate_single_experiment(env_id=env_id, n_eval_seeds=n_eval_seeds, results_dir=results_dir, 
                                                       expt_params=expt_params, expt=expt, model_idx=model_idx, 
                                                       alpha=alpha, gamma=gamma, imit_rew_coef=imit_rew_coef)   
                results[expt][ckpt_dirname] = eval_data

    return results
            

if __name__ == '__main__':
    args = argparser()
    results_dir = paths['rl_demo']['results_dir']
    n_eval_seeds = params['validation']['seeds']

    expt_params = return_expt_params(args.env_id)


    if args.expt and args.model_idx and args.imit_rew_coef and args.gamma and args.alpha: 
        # always run locally
        evaluate_single_experiment(args.env_id, n_eval_seeds, results_dir, expt_params, args.expt, args.model_idx,
                                   imit_rew_coef=args.imit_rew_coef, gamma=args.gamma, alpha=args.alpha)
    else: 
        # param_grid = ParameterGrid(rew_param_grid)
        # option of running on cluster
        evaluate_experiments(args.env_id, n_eval_seeds, results_dir, expt_params)
