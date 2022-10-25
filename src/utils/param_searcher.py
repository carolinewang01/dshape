import os
import shutil
import subprocess

import numpy as np
import yaml

from sklearn.model_selection import ParameterGrid


class ParamSearcher():
    def __init__(self, cmd, conf_dir, run_local, param_dict, job_name, results_dir, trials_per_param=1, 
                 random_sampling=False, search_budget=None,
                 condor_log_dir="", condor_extra_contents="", require_gpu=False):
        '''
        The ParamSearcher does a grid search over the provided parameters and submits jobs locally or to Condor. 
        Because it is meant to be a general hyperparameter search tool, and the analysis process varies immensely, 
        we do not provide results analysis.  

        cmd: command line command, no arguments 
            if running locally, include prefix "python <-m> filedir.mypythonfile"
            if running on condor, write full filepath filedir/mypythonfile.py and make python file executable 
        conf_dir: directory to save config yaml to s.t. the cmd cOohan later read it back out 
        results_dir: directory to save results to 
        run_local: boolean for whether or not to run hyperparameter search locally. If false, run on Condor
        param_dict: a dictionary in format {"param_name": [list of values]}
        random_sampling: if True, search parameter grid uniformly at random
        search_budget: number of combinations to sample 
        '''
        self.cmd = cmd 
        self.conf_dir = conf_dir 
        self.run_local = run_local
        self.param_grid = list(ParameterGrid(param_dict))
        self.require_gpu = require_gpu
        self.trials_per_param = trials_per_param
        self.job_name = job_name
        self.results_dir = results_dir
        self.condor_log_dir = condor_log_dir
        self.condor_extra_contents = condor_extra_contents

        self.random_sampling = random_sampling 
        if self.random_sampling: 
            self.param_grid = np.random.choice(self.param_grid, size=search_budget, replace=False)

        # if not self.run_local: # wipe and recreate condor log dirs
        #     if os.path.exists(self.condor_log_dir):
        #         shutil.rmtree(self.condor_log_dir)
        #     if not os.path.exists(self.condor_log_dir):
        #         os.mkdir(condor_log_dir)


    def run_on_local(self, exec_kwargs:dict, param_idx:int, num_trials:int):
        '''
        Assumption is that the cmd takes command line keyword args in the format --kwarg value 
        '''
        cmd = self.cmd
        for key, value in exec_kwargs.items():
            cmd += f" --{key} {value}"

        for i in range(num_trials):
            os.system(cmd)

    def run_on_condor(self, exec_kwargs:dict, param_idx:int, num_trials:int):
        '''
        Assumption is that the cmd takes command line keyword args in the format --kwarg value, and 
        the cmd file is already executable 
        '''
        if self.require_gpu: 
            pass 
        else: 
            pass 
        condor_contents = \
f"""Executable = {self.cmd}
Universe = vanilla
Getenv = true
+Group = "GRAD" 
+Project = "AI_ROBOTICS"
+ProjectDescription = "{self.job_name}"

Input = /dev/null
Error = {self.condor_log_dir}/{self.job_name}-param-{param_idx}.err
Output = {self.condor_log_dir}/{self.job_name}-param-{param_idx}.out
Log = {self.condor_log_dir}/{self.job_name}-param-{param_idx}.log
""" 
        condor_contents += self.condor_extra_contents
        if self.require_gpu: 
            condor_contents += "+GPUJob = true\n"
            condor_contents += "Requirements = (TARGET.GPUSlot)\n"
        condor_contents += 'arguments = '
        for key, value in exec_kwargs.items():
            condor_contents += f"--{key} {value} "
        condor_contents += f'\nQueue {num_trials}'

        # submit to condor        
        proc = subprocess.Popen('condor_submit', stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        proc.stdin.write(condor_contents.encode())
        proc.stdin.close()
        print(f"Submitted param set {param_idx} to condor with {num_trials} trials")

    def write_config_yaml(self, param_idx, param_dict):
        '''writes current set of configs to the conf_dir'''
        name_of_yaml = os.path.join(self.conf_dir, f'{self.job_name}_{param_idx}.yml')
        with open(name_of_yaml, 'w') as outfile: 
            yaml.dump(param_dict, outfile, default_flow_style=False)
        return name_of_yaml

    def get_savefile_dirs(self, exec_kwargs):
        '''
        Should return a list of savefile directories for a given parameter set
        THIS MUST BE IMPLEMENTED IN INHERITED CLASS
        TODO: clean this up to be an ABS
        '''
        return []

    def check_single_dir(self, savefile_dir):
        if not os.path.exists(savefile_dir): # directory doesn't even exist
            return False
        else: 
            if not os.listdir(savefile_dir): # directory exists but is empty
                return False
        return True

    def count_nonempty_dirs(self, savefile_dirlist):
        '''
        Checks number of nonempty and/or existing directories in the list of directories
        '''
        done = []
        for savefile_dir in savefile_dirlist:
            done.append(self.check_single_dir(savefile_dir))
        return sum(done)

    def run(self, exec_kwargs):
        '''
        This is the main function to call
        Cmd must have an commandline kwarg corresponding to the location of the params_file
        '''
        for i, single_param_dict in enumerate(self.param_grid): 
            name_of_yaml = self.write_config_yaml(i, single_param_dict)
            exec_kwargs['params_file'] = name_of_yaml

            savefile_dirs = self.get_savefile_dirs(exec_kwargs) # get all directories corresponding to paramset
            num_trials_done = self.count_nonempty_dirs(savefile_dirs)
            num_trials_remaining = max(self.trials_per_param - num_trials_done, 0)

            # train model
            if num_trials_remaining > 0:
                if self.run_local:
                    self.run_on_local(exec_kwargs, param_idx=i, num_trials=num_trials_remaining)
                else: 
                    self.run_on_condor(exec_kwargs, param_idx=i, num_trials=num_trials_remaining)
