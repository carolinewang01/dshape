import os 
import subprocess
import argparse

from utils.load_confs import load_parameters, load_paths
from utils.helpers import str2bool

params = load_parameters()
paths = load_paths()


def evaluate_experiments(results_dir):

    for env_id in [
                   'Reacher-v2', 
                   'Swimmer-v2', 'Ant-v2', 
                   "Hopper-v2", 
                   "HalfCheetah-v2", 
                   "Walker2d-v2"
                   ]:
        print(f"Evaluating {env_id}")
          
        submit_to_slurm(env_id, checkpoint_dir=results_dir, job_name=f"rew_shaping_{env_id}_hyperparam")
                
    return 
            

def submit_to_slurm(env_id, checkpoint_dir, job_name 
                    # model_idx
                    ):
    '''purpose of this function is to submit a script to slurm that creates an Evaluator object, and runs the
    evaluation locally on slurm
    '''    
    slurm_contents = \
f"""#!/bin/bash
#SBATCH --job-name {job_name}_eval                                    # Job name
### Logging
#SBATCH --output={checkpoint_dir}/{job_name}_eval_%A_%a.out           # Name of stdout output file (%j expands to jobId)
#SBATCH --error={checkpoint_dir}/{job_name}_eval_%A_%a.err            # Name of stderr output file (%j expands to jobId) %A should be job id, %a sub-job
### Node info
#SBATCH --partition dgx                                                    # titans or dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 48:00:00                                                      # Run time (hh:mm:ss)
#SBATCH --gres=gpu:1                                                         # Number of gpus needed
#SBATCH --mem=2G                                                            # Memory requirements
#SBATCH --cpus-per-task=2                                                    # Number of cpus needed per task
python -m reward_shaping.evaluate_hyperparams --env_id {env_id} 
""" 
    # submit to slurm
    proc = subprocess.Popen('sbatch', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(slurm_contents.encode())
    proc.stdin.close()
    # print("SLURM SUB SCRIPT IS ", slurm_contents)
    print(f"Submitted evaluation job for {checkpoint_dir}, {env_id} to slurm")


if __name__ == '__main__':
    results_dir = paths['rl_demo']['results_dir']
    evaluate_experiments(results_dir)
