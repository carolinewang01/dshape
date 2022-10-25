import numpy as np 
import argparse
from contextlib import contextmanager
import sys, os
from utils.load_confs import load_parameters, load_paths

params = load_parameters()
paths = load_paths()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_demo(env_id, demo_algo=None, raw=True, shuffle=False, time_feat=False, sin_cos_repr=False):
    expert_raw_demo_path = f"{paths['train_sample_paths']['joint_angles_save_dir']}/joint_angles_{demo_algo}.{env_id}.seed=None.npz"
    expert_raw_demo_files = np.load(expert_raw_demo_path)
    if raw: 
        # load raw demo
        expert_demo_obs = expert_raw_demo_files['obs']
        expert_demo_acs = expert_raw_demo_files['acs']
        if sin_cos_repr:
            # TODO: do this for full demonstrations as well
            expert_demo_obs = np.concatenate([np.cos(expert_demo_obs), np.sin(expert_demo_obs)], axis=1)
    else: 
        # load full demo corresp to raw demo
        demo_idx = expert_raw_demo_files['traj_ind'][0]
        expert_demo_path = f"{paths['train_sample_paths']['sample_save_dir']}/{demo_algo}.{env_id}.seed=None.npz"
        expert_demo_obs = np.load(expert_demo_path)['obs'][demo_idx]
        expert_demo_acs = np.load(expert_demo_path)['acs'][demo_idx]
    if shuffle: # shuffle along the time dimension
        np.random.shuffle(expert_demo_obs) # should only shuffle along first dimension
    if time_feat:
        exp_timesteps = expert_demo_obs.shape[0]
        max_timesteps = get_env_timesteps(env_id)
        time_feature = np.expand_dims(1 - (np.arange(exp_timesteps) / max_timesteps), axis=1)
        expert_demo_obs = np.concatenate((expert_demo_obs, time_feature), axis=1)
    return expert_demo_obs, expert_demo_acs


def get_env_timesteps(env_id):
    if env_id == "Reacher-v2": 
        return 50
    elif env_id in ["Swimmer-v2", "Ant-v2", "HalfCheetah-v2", "Walker2d-v2", "Hopper-v2"]:
        return 1000
    else: 
        env = gym.make(env_id)
        return env._max_episode_steps


def extract_observable_state(full_state, env_id, time_feat=False, precision=None, sin_cos_repr=False):
    '''Given the full observation from a MuJoCo environment, extract the a subset of variables
    (e.g. joint angles for Reacher). If time_feat, then preserve the time feature (which should be the last idx of the observation)
    '''
    if env_id == 'Reacher-v2': # joint angles (radians) 
        # TODO: add the target position!?
        if precision:
            if sin_cos_repr:
                raw_state = np.array(full_state[:4], dtype=precision)
            else:
                raw_state = np.array([np.arctan2(full_state[2], full_state[0], dtype=precision), 
                                      np.arctan2(full_state[3], full_state[1])], dtype=precision)
        else: 
            if sin_cos_repr:
                raw_state = np.array(full_state[:4], dtype=precision)
            else:
                raw_state = np.array([np.arctan2(full_state[2], full_state[0]), 
                          np.arctan2(full_state[3], full_state[1])])

    elif env_id == 'Swimmer-v2': # joint angles (radians)
        raw_state = full_state[1:3]
        if sin_cos_repr:
            raw_state = np.concatenate([np.cos(raw_state), np.sin(raw_state)])
    elif env_id == 'Ant-v2': # joint angles (radians)
        raw_state = full_state[5:13]
        if sin_cos_repr:
            raw_state = np.concatenate([np.cos(raw_state), np.sin(raw_state)])
    elif env_id == 'Hopper-v2': # joint angles (radians)
        raw_state = full_state[2:5]
        if sin_cos_repr:
            raw_state = np.concatenate([np.cos(raw_state), np.sin(raw_state)])
    elif env_id == 'HalfCheetah-v2': # joint angles (radians)
        raw_state = full_state[2:8]
        if sin_cos_repr:
            raw_state = np.concatenate([np.cos(raw_state), np.sin(raw_state)])
    elif env_id == 'Walker2d-v2': # joint angles (radians)
        raw_state = full_state[2:8]
        if sin_cos_repr:
            raw_state = np.concatenate([np.cos(raw_state), np.sin(raw_state)])
    if time_feat:
        raw_state = np.concatenate((raw_state, full_state[-1:]))
    return raw_state


def get_raw_state_size(env_id, sin_cos_repr=False):
    if env_id == 'Reacher-v2':
        raw_state_size = 2
    elif env_id == 'Ant-v2':
        raw_state_size = 8
    elif env_id == 'Hopper-v2':
        raw_state_size = 3
    elif env_id == 'HalfCheetah-v2':
        raw_state_size = 6
    elif env_id == 'Walker2d-v2':
        raw_state_size = 6
    elif env_id == 'Swimmer-v2':
        raw_state_size = 2
    if sin_cos_repr:
        raw_state_size *= 2
    return raw_state_size


def argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


def get_joint_angles(env_id, expert_demo_path, paths, mode=None):
    '''Converts demonstrations from TRPO/PPO/SAC to a joint angle demo (observable states)
    Returns single trajectory only 
    '''
    demo = np.load(f"{expert_demo_path}.npz")

    if mode == "best":
        traj_ind = np.argmax(demo['ep_rets'])
    elif mode == "median":
        traj_ind = argmedian(demo['ep_rets'])

    obs = demo['obs'][traj_ind]
    acs = demo['acs'][traj_ind]
    ret = demo['ep_rets'][traj_ind]
    print(f"{mode} sampled trajectory achieved ep ret of {ret}")

    new_obs = []
    for i in obs:
        i = extract_observable_state(i, env_id) 
        new_obs.append(i)
    new_obs = np.array(new_obs)
    assert len(new_obs) == len(obs) 

    # construct save path names
    save_dir = os.path.join(paths['train_sample_paths']['joint_angles_save_dir'])
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if expert_demo_path is not None:
        save_path = os.path.join(save_dir, 'joint_angles_' + expert_demo_path.split(os.path.sep)[-1])
        np.savez(save_path, obs = new_obs, acs = acs, traj_ind = np.array([traj_ind]), rew=ret)


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

