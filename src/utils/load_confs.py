import yaml
import os
import re

project_dir = re.split(r"\\|/",
                    os.path.dirname(os.path.realpath(__file__))) + ['..', '..']


def collapse_dict_hierarchy(nested_dict: dict):
    collapsed_dict = {}
    for name, subdict in nested_dict.items():
        collapsed_dict = {**collapsed_dict, **subdict}
    return collapsed_dict


def load_parameters():
    filepath = os.sep.join(project_dir + 'confs/parameters.yml'.split('/'))
    with open(filepath) as f:
        params = yaml.safe_load(f)
    return params


def load_paths():
    filepath = os.sep.join(project_dir + 'confs/paths.yml'.split('/'))

    with open(filepath) as f:
        paths = yaml.safe_load(f)
    for path_collection_name in paths: 
        paths_dict = paths[path_collection_name]
        for k in paths_dict.keys():
            paths_dict[k] = os.sep.join(paths_dict[k].split('/'))

    return paths


def load_reward_params(params_name):
    filepath = os.sep.join(project_dir + ['confs'] + params_name.split('/'))
    with open(filepath) as f:
        params = yaml.safe_load(f)
    return params

