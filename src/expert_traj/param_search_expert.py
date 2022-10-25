import sys
import os
import time
import glob
import shutil
import subprocess
import argparse

import numpy as np 
import pandas as pd 
import gym  
import tensorflow as tf

from sklearn.model_selection import ParameterGrid
from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool

params = load_parameters()['train_sample_params']
paths = load_paths()['train_sample_paths']
temp_dir = "expert_traj/temp"
save_dir = "expert_traj/results/"

if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.mkdir(temp_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

def argparser():
    parser = argparse.ArgumentParser("Running hyperparameter search over PPO")
    parser.add_argument('--env_id', help='environment ID', default='Reacher-v2')
    parser.add_argument('--random_init', type=int, choices=[-1, 0, 1, 2], help='random initial positions for agent', default=-1) 
    parser.add_argument('--run_local', type=str2bool, help="if true run locally else on condor")
    return parser.parse_args()
args = argparser()

ppo_param_grid = {'adam_epsilon': [3e-4, 1e-5],
                'clip_param': [0.2, 0.3],
                'entcoeff': [0, 0.01],
                'gamma': [0.99, .9],
                'lam':  [0.9, 0.95]
                 }

ppo_param_grid = list(ParameterGrid(ppo_param_grid))

if not args.run_local: 
    condor_log_enabled = True
    ids = []
#################################
for i, param_dict in enumerate(ppo_param_grid): 
    # train model
    adam_epsilon = param_dict['adam_epsilon']
    clip_param = param_dict['clip_param']
    entcoeff = param_dict['entcoeff']
    gamma = param_dict['gamma']
    lam = param_dict['lam']

    if args.run_local: 
        cmd = f"python -m expert_traj.train_sample_experts --env_id {args.env_id} --task train --random_init {args.random_init} \
 --adam_epsilon {adam_epsilon} --clip_param {clip_param} --entcoeff {entcoeff} --gamma {gamma} --lam {lam} \
 --task_name_save_path {temp_dir}/{args.env_id}_{i}.txt"
        os.system(cmd)
#        proc = subprocess.Popen(cmd)
 #       proc.wait()

    # run on condor    
    else: 
        condor_contents = \
f"""Executable = expert_traj/train_sample_experts.py
Universe = vanilla
Getenv = true
Requirements = (ARCH == "X86_64") && (Angrist)

+Group = "GRAD" 
+Project = "AI_ROBOTICS"
+ProjectDescription = "Hyperparam Search on PPO"

Input = /dev/null
""" 

        if condor_log_enabled:
            condor_log_dir = f'expert_traj/log/condor/{args.env_id}'
            # if os.path.exists(condor_log_dir):
            #     shutil.rmtree(condor_log_dir)

            if not os.path.exists(condor_log_dir):
                os.mkdir(condor_log_dir)

            condor_contents += f'Error = {condor_log_dir}/error-param-{i}.err\n'   
            condor_contents += f'Output = {condor_log_dir}/out-param-{i}.out\n'   
            condor_contents += f'Log = {condor_log_dir}/log-param-{i}.log\n' 
        else:
            condor_contents += 'Error = /dev/null\n'
            condor_contents += 'Output = /dev/null\n'
            condor_contents += 'Log = /dev/null\n'
        
        condor_contents += 'arguments = '
        condor_contents += f"--env_id {args.env_id} --task train --random_init {args.random_init} \
                             --adam_epsilon {adam_epsilon} --clip_param {clip_param} --entcoeff {entcoeff} --gamma {gamma} --lam {lam} \
                             --task_name_save_path {temp_dir}/{args.env_id}_{i}.txt"

        condor_contents += '\nQueue 1'
        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write(condor_contents.encode())
        proc.stdin.close()
        proc.wait()

        for line in proc.stdout:
            if not isinstance(line, str):
                line = line.decode('utf8')
            if 'cluster' in line:
                ids.append(line.split()[-1][:-1])
        time.sleep(0.05)

if not args.run_local:
    # continue checking when jobs are over by checking for desired output
    print('Submitted %d jobs' % len(ppo_param_grid))
    print(f"Job ids: {ids}")

    remaining_jobs = len(ids)
    while remaining_jobs>0:
        # recompute remaining_jobs
        remaining_jobs = 0
        files = list(glob.iglob(f"{temp_dir}/{args.env_id}_*.txt"))
        for i in range(len(ppo_param_grid)):
            file = f"{temp_dir}/{args.env_id}_{i}.txt"
            if file not in files: 
                remaining_jobs +=1
        print(f"{remaining_jobs} jobs remaining. Sleeping for 5 seconds.")
        time.sleep(5)


#################################
# process information from logs 
res_dicts = []
for i, param_dict in enumerate(ppo_param_grid):
    with open(f"{temp_dir}/{args.env_id}_{i}.txt") as f: 
        task_name = f.readline()
    log_path = f"{paths['log_dir']}{task_name}/tb"
    log_file = os.listdir(log_path)[0]

    iteration_rets = []
    for summary in tf.train.summary_iterator(f"{log_path}/{log_file}"):
        for v in summary.summary.value:
            if v.tag == 'Episode_Reward_Mean':
                mean_ep_ret = v.simple_value
                iteration_rets.append(mean_ep_ret)
    # best performance through all iterations
    param_best_perf = np.max(np.array(iteration_rets))

    param_dict['best_mean_ep_ret'] = param_best_perf

    res_dicts.append(param_dict)

df = pd.DataFrame(res_dicts)
df.to_csv(f"{save_dir}hyperparam_search_ppo_{args.env_id}_random_init={args.random_init}.csv")
best_row_ind = df['best_mean_ep_ret'].idxmax()
print("Best Parameters:\n", df.iloc[best_row_ind])

# cleanup 
# shutil.rmtree(temp_dir)
