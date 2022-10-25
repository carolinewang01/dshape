from abc import ABC, abstractmethod
import os
import glob
import json


class Evaluator(ABC):
    '''The purpose of this class is to act as a base class to evaluate N model checkpoints in a folder. 
    Evaluation results are saved to the checkpoint dir as a csv
    '''
    def __init__(self, job_name, checkpoint_dir, n_eval_seeds:list, n_eval_rollouts:int):
        self.job_name = job_name
        self.checkpoint_dir = checkpoint_dir
        self.n_eval_rollouts = n_eval_rollouts      
        self.n_eval_seeds = n_eval_seeds
        assert len(self.n_eval_seeds) == n_eval_rollouts # a seed per rollout

    def evaluate(self, read_from_saved=True, result_format='json', result_prefix="ts="):
        '''
        If saved result is detected or read_from_saved is True, load results from a saved file and return. 
        Else, iterate through all checkpoints in the checkpoint_dir, and do n_eval_rollouts for each checkpoint
        '''
        # iterate (glob) through all checkpoints in file 
        if result_format == 'json':
            result_path = f"{self.checkpoint_dir}/{self.job_name}_evaluation.json"
        elif result_format == 'csv':
            result_path = f"{self.checkpoint_dir}/{self.job_name}_evaluation.csv"


        if not os.path.exists(result_path) or not read_from_saved:
            print(f"Evaluation {result_format} not found for {self.checkpoint_dir}. \nCreating one now...")
            data = self.create_eval_data(result_prefix=result_prefix)

            self.save_result(data, result_path, result_format)
        else:
            print(f"Found {result_format} for {self.checkpoint_dir}. \nLoading from saved...")
            data = self.read_result(result_path, result_format)
        return data 


    def create_eval_data(self, result_prefix):
        data = {}
        for ckpt_file_path in glob.glob(os.path.join(self.checkpoint_dir, f"{result_prefix}*"), recursive=True): 
            ckpt_filename = os.path.basename(ckpt_file_path)
            ts, name = self.get_ts_and_name(ckpt_filename, result_prefix) # get jobname and timestep info if present
            if name not in data.keys(): 
                data[name] = {}
            data[name][ts] = self.eval_checkpoint(ckpt_file_path)
        return data 


    @abstractmethod
    def eval_checkpoint(self, ckpt_file_path:str):
        '''purpose of this function is to load checkpointed model from path, evaluate for n_eval_rollouts
        using n_eval_seeds, and return the mean performance. 
        Returns param: mean_return:float 
        '''
        pass


    @abstractmethod
    def get_ts_and_name(self, ckpt_filename:str):
        ''' purpose of this function is to process the name of a checkpoint and return the timestep
        and model name 
        Return param: (ts, ckpt_name)
        '''
        pass 

    def save_result(self, data:dict, result_path:str, result_format='json'):
        # save to the checkpoint dir 
        if result_format == 'json':
            with open(result_path, 'w') as f: 
                json.dump(data, f, sort_keys=True, indent=4)
        elif result_format == 'csv': 
            pass 

    def read_result(self, result_path:str, result_format='json'):
        '''Read from ckpt saved result file '''
        print(f"Reading saved values from {self.checkpoint_dir}")

        if result_format == 'json':
            with open(result_path, 'r') as f:
                print("f is ", f)
                contents = f.read()
                # print("ORIG CONTENTS IS ", contents)

                while contents.count("{") != contents.count("}"):
                    lcount_bracket, rcount_bracket  = contents.count("{"),  contents.count("}")
                    if lcount_bracket > rcount_bracket: 
                        k = contents.lfind("{")
                        contents = contents[:k] + contents[k+1:]
                    if rcount_bracket > lcount_bracket: 
                        k = contents.rfind("}")
                        contents = contents[:k] + contents[k+1:]

                # print("CONTENTS IS ", contents)
                data = json.loads(contents,  object_hook=lambda d: {int(k) if k.lstrip('-').isdigit() else k: v for k, v in d.items()})
            return data

        elif result_format == 'csv': 
            pass 

