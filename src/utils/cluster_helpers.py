import os
import subprocess


def submit_to_slurm(env_id, exec_cmd, results_dir, job_name, expt_params, num_trials, partition="dgx", mem=8):
    '''purpose of this function is to submit a script to condor that runs num_trials instances of the reward shaping expt
    '''    
    if num_trials == 0: 
        # print(f"0 jobs submitted to slurm for {results_dir + job_name}, {env_id} to slurm {partition}")
        return 
    slurm_log_dir = os.path.join(results_dir, 'slurm_logs')
    if not os.path.exists(slurm_log_dir):
        os.makedirs(slurm_log_dir)

    slurm_contents = \
f"""#!/bin/bash
#SBATCH --job-name {job_name}                                   # Job name
### Logging
#SBATCH --output={slurm_log_dir}/{job_name}_%A_%a.out           # Name of stdout output file (%j expands to jobId)
#SBATCH --error={slurm_log_dir}/{job_name}_%A_%a.err            # Name of stderr output file (%j expands to jobId) %A should be job id, %a sub-job
### Node info
#SBATCH --partition {partition}                                                    # titans or dgx
#SBATCH --nodes=1                                                            # Always set to 1 when using the cluster
#SBATCH --ntasks-per-node=1                                                  # Number of tasks per node (Set to the number of gpus requested)
#SBATCH --time 72:00:00                                                      # Run time (hh:mm:ss)
#SBATCH --gres=gpu:1                                                         # Number of gpus needed
#SBATCH --mem={mem}G                                                            # Memory requirements
#SBATCH --cpus-per-task=2                                                    # Number of cpus needed per task
sleep $(($SLURM_ARRAY_TASK_ID * 5))
""" 
    # create multiple trials
    for trial_idx in range(num_trials):
        # TODO: remove the run id from expt_params
        expt_params["run_id"] = trial_idx
        slurm_contents += exec_cmd
        for k, v in expt_params.items():
            slurm_contents += f" --{k} {v}" 
        slurm_contents += " &\n"
    slurm_contents += "wait\n"
    slurm_contents += "echo \"Done\"\n"
    slurm_contents += "exit 0\n"

    # submit to slurm
    # proc = subprocess.Popen(['sbatch', f'--array=1-{num_trials}'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc = subprocess.Popen(['sbatch', f'--array=1-1'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(slurm_contents.encode())
    proc.stdin.close()

    # print("SLURM SUB SCRIPT IS \n", slurm_contents)
    print(f"Submitted {num_trials} jobs for {results_dir + job_name}, {env_id} to slurm {partition}")


def submit_to_condor(env_id, exec_cmd, results_dir, job_name, expt_params, num_trials, req_gpu=True):
    '''purpose of this function is to submit a script to condor that runs num_trials instances of the reward shaping expt
    '''    
    if num_trials == 0: 
        # print(f"0 jobs submitted to condor for {results_dir + job_name}, {env_id} to slurm")
        return 

    condor_log_dir = os.path.join(results_dir, 'condor_logs')
    if not os.path.exists(condor_log_dir):
        os.makedirs(condor_log_dir)

    condor_contents = \
f"""Executable = {exec_cmd} 
Universe = vanilla
Getenv = true
"""
    if req_gpu:
        condor_contents += \
f"""
+GPUJob = true
Requirements = (TARGET.GPUSlot)
"""
    else:
        condor_contents += \
f"""
Requirements = InMastodon
"""
    condor_contents += \
f"""
+Group = "GRAD" 
+Project = "AI_ROBOTICS"
+ProjectDescription = "{job_name} {env_id}"

Input = /dev/null
Error = {condor_log_dir}/{job_name}_$(CLUSTER).err
Output = {condor_log_dir}/{job_name}_$(CLUSTER).out
Log = {condor_log_dir}/{job_name}_$(CLUSTER).log

Notification = Never

arguments = \
""" 
    for k, v in expt_params.items():
        if type(v) is tuple:
            v = [str(i) for i in v]
            v = " ".join(v)
            condor_contents += f" --{k} {v}" 
        else:
            condor_contents += f" --{k} {v}" 
    condor_contents += f"\nQueue {num_trials}"

    # submit to condor
    proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    proc.stdin.write(condor_contents.encode())
    proc.stdin.close()
    # print("CONDOR SUB SCRIPT IS \n", condor_contents)
    print(f"Submitted {num_trials} jobs for {results_dir + job_name}, {env_id} to condor")
