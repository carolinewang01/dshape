import os
import glob
from types import SimpleNamespace
from reward_shaping.sac_reward import make_custom_env
from utils.param_searcher import ParamSearcher


class SACParamSearcher(ParamSearcher):
    def __init__(self, cmd, conf_dir, run_local, param_dict, job_name, results_dir, algo_name, 
                 trials_per_param=1, random_sampling=False, search_budget=None,
                 condor_log_dir="", condor_extra_contents="", 
                 require_gpu=False):

        super(SACParamSearcher, self).__init__(cmd, conf_dir, run_local, param_dict, job_name, results_dir, 
              trials_per_param=trials_per_param, 
              random_sampling=random_sampling, search_budget=search_budget,
              condor_log_dir=condor_log_dir, condor_extra_contents=condor_extra_contents, require_gpu=require_gpu)

        self.algo_name = algo_name


    def get_savefile_dirs(self, exec_kwargs):
        '''
        Returns a list of savefile directories for a given parameter set
        '''
        exec_kwargs = SimpleNamespace(**exec_kwargs) # enable dot syntax access for dicts

        assert exec_kwargs.params_file is not None
        dummy_env = make_custom_env(exec_kwargs) # TODO: clean this up later; this is only here to get the task log name
        dummy_env.close()

        savefile_name = f"{self.algo_name}_{exec_kwargs.env_id}"
        savefile_base = os.path.join(self.results_dir, exec_kwargs.task_log_name, "checkpoint", f"{savefile_name}*") # distinguish between run ids?
        savefile_dirs = glob.glob(savefile_base, recursive=True)
        return savefile_dirs
