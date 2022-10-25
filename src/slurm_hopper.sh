#!/bin/bash
#SBATCH --job-name ppo2_Hopper                                    # Job name
### Logging
#SBATCH --output=/scratch/cluster/clw4542/rl_demo_results_final/baseline_rl_hidden=64/ppo2_Hopper_%A_%a.out           # Name of stdout output file (%j expands to jobId)
#SBATCH --error=/scratch/cluster/clw4542/rl_demo_results_final/baseline_rl_hidden=64/ppo2_Hopper_%A_%a.err            # Name of stderr output file (%j expands to jobId) %A should be job id, %a sub-job
### Node info
#SBATCH --partition dgx                                                    # titans or dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 120:00:00                                                      # Run time (hh:mm:ss)
#SBATCH --gres=gpu:2                                                         # Number of gpus needed
#SBATCH --mem=9G                                                            # Memory requirements
#SBATCH --cpus-per-task=8                                                    # Number of cpus needed per task
sleep $((SLURM_ARRAY_TASK_ID * 2))
python -m rl_demo.ppo_demo --task baseline_rl --env_id Hopper-v2 --algo ppo2 
