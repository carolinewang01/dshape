#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import os.path as osp
import os
import sys, time, argparse
from time import localtime, strftime

import numpy as np
import gym  
from mpi4py import MPI
from baselines.common.misc_util import boolean_flag
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.cmd_util import make_mujoco_env

from env_interact_other import traj_segment_generator
from utils.helpers import str2bool, extract_observable_state, return_random_reset, return_generalized_reset, return_fixed_reset, get_joint_angles
from utils.load_confs import load_parameters, load_paths
from expert_traj.expert_algos import pposgd_simple, trpo_mpi
from expert_traj.expert_algos.mlp_policy import MlpPolicy

params = load_parameters()['train_sample_params']
paths = load_paths()['train_sample_paths']


def argparser():

    parser = argparse.ArgumentParser("Training and running mujoco")
    # for all
    parser.add_argument('--task', type=str, choices=['train', 'sample', 'exec_random_policy'], default='train')
    parser.add_argument('--env_id', help='environment ID', default='Reacher-v2')
    parser.add_argument('--random_init', type=int, choices=[-1, 0, 1, 2], help='random initial positions for agent', default=0) 
    parser.add_argument('--traj_seed', help='seed for a random initial positions for agent', type=int, default=None)
    # for training 
    parser.add_argument('--adam_epsilon', type=float, default=1e-5)
    parser.add_argument('--clip_param', type=float, default=0.3)
    parser.add_argument('--entcoeff', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=.99)
    parser.add_argument('--lam', type=float, default=.95)
    parser.add_argument('--task_name_save_path', type=str, help='FULL path to write the task name to',)
    # for sampling
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)

    return parser.parse_args()


def get_task_name(env, algo, seed, traj_seed):
    task_name = algo + "."
    task_name += env.split("-")[0]
    task_name += f".seed_{seed}.traj_seed_{traj_seed}." + strftime("%m_%d %H_%M_%S", localtime())
    # task_name += f".seed_reduced_state_{seed}.traj_seed_{traj_seed}." + strftime("%Y_%m_%d %H_%M_%S", localtime())
    return task_name


def policy_fn(name, ob_space, ac_space):
    '''Define the policy architecture for TRPO and PPO
    '''
    return MlpPolicy(name=name, ob_space=ob_space, 
                    ac_space=ac_space, hid_size=64, 
                    num_hid_layers=2)


def train(env_id, algo, seed, ckpt_dir, log_dir, 
          random_init, traj_seed=None, **params):
    '''  
    '''
    task_name = get_task_name(env_id, algo, seed, traj_seed)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()

    env = make_mujoco_env(env_id, workerseed)
    if random_init == -1: 
        # arm initialized in original fashion
        init_type = "original_init"

    elif random_init == 0:
        # arm initialized deterministically according to traj_seed 
        env.unwrapped.reset = return_random_reset(env_id, env, traj_seed)
        init_type = "random_init"

    elif random_init == 1:
        # env.reset() will reset arm to random initial position
        env.unwrapped.reset = return_generalized_reset(env_id, env)
        init_type = "generalized_init"

    elif random_init == 2:
        # env.reset() will reset deterministically to initial position determined by traj_seed 
        # init velocity and target position also fixed 
        env.unwrapped.reset = return_fixed_reset(env_id, env, traj_seed)
        init_type = "fixed_init"

    if ckpt_dir is not None: # None means don't save models to checkpoint
        ckpt_dir = osp.join(ckpt_dir, init_type, task_name)
        if not osp.isdir(ckpt_dir):
            os.makedirs(ckpt_dir)
    sess = U.make_session(num_cpu=1).__enter__()

    if algo == 'trpo':
        trpo_mpi.learn(env, policy_fn, task_name=task_name, 
                       ckpt_dir=ckpt_dir, log_dir=log_dir, 
                       timesteps_per_batch=512, max_kl=0.01, cg_iters=10, 
                       cg_damping=0.1, max_iters=1e6, gamma=0.99, 
                       lam=0.98, vf_iters=5, vf_stepsize=1e-3)
        # params for swimmer 270 reward
        #trpo_mpi.learn(env, policy_fn, task_name=task_name, ckpt_dir=ckpt_dir, log_dir=log_dir, timesteps_per_batch=5000, max_kl=0.01, cg_iters=20, cg_damping=0.1, max_iters=max_iters, gamma=0.995, lam=0.97, vf_iters=5, vf_stepsize=1e-3)
        # hopper from deep rl that matters paper
        #trpo_mpi.learn(env, policy_fn, task_name=task_name, ckpt_dir=ckpt_dir, log_dir=log_dir, timesteps_per_batch=5000, max_kl=0.01, cg_iters=20, cg_damping=0.1, max_iters=max_iters, gamma=0.995, lam=0.97, vf_iters=5, vf_stepsize=1e-3)
        # cheetah 
        #trpo_mpi.learn(env, policy_fn, task_name=task_name, ckpt_dir=ckpt_dir, log_dir=log_dir, timesteps_per_batch=512, max_kl=0.01, cg_iters=10, cg_damping=0.1, max_iters=max_iters, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)

        print ('WARNING!!! - breaking at fixed iteration!!!!!')

    elif algo == 'ppo': 
        pposgd_simple.learn(env, policy_fn,
                            task_name=task_name,
                            ckpt_dir=ckpt_dir,
                            log_dir=log_dir,
                            log_per_iter=5,
                            max_timesteps= 2e6, # 5e4, #10 iter
                            timesteps_per_actorbatch=5000,
                            clip_param=params['clip_param'], entcoeff=params['entcoeff'],
                            optim_epochs=10, optim_stepsize=2e-4, optim_batchsize=64,
                            gamma=params['adam_epsilon'], lam=params['lam'], schedule='linear',
                            adam_epsilon=params['adam_epsilon']
            )

    env.close()
    return task_name


def sample(env_id, algo, num_trajs,
           load_dir, data_dir, 
           manual, visual, 
           seed, random_init=False, traj_seed=None):
    '''
    Args
       manual: decide whether to keep traj manually or not
       visual: visualize or not
       traj_seed: if random_init is true, this is the seed that will be used to generate random trajectories
    Returns
       data_dir: save_path to the sampled trajectory
    '''
    print ('xxxxxxxx load dir {}'.format(load_dir))
    # task_name = get_task_name(env_id, algo, seed)
    # workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    # env = make_mujoco_env(env_id, workerseed)

    env = gym.make(env_id)
    if random_init == -1: 
        save_dir = osp.join(data_dir, "original_init")

    elif random_init == 0:
        #arm initialized according to traj_seed with random noise
        # save to correct folder and include random seed in name
        env.unwrapped.reset = return_random_reset(env_id, env, traj_seed)
        save_dir = osp.join(data_dir, "random_init")

    elif random_init == 1:
        # env.reset() will reset arm to random initial position
        env.unwrapped.reset = return_generalized_reset(env_id, env)
        save_dir = osp.join(data_dir, "generalized_init")

    elif random_init == 2:
        # env.reset() will reset deterministically to initial position determined by traj_seed 
        env.unwrapped.reset = return_fixed_reset(env_id, env, traj_seed)
        save_dir = osp.join(data_dir, "fixed_init")

    if not osp.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, f'{algo}.{env_id}.seed={traj_seed}')

    sess = U.make_session(num_cpu=1).__enter__()
    if algo =='trpo':
        trpo_mpi.sample(env, env_id, policy_fn=policy_fn, random_init=random_init, 
                        load_dir=load_dir, data_dir=save_path, 
                        manual=manual, visual=visual, num_trajs=num_trajs)
   
    if algo == 'ppo':
        pposgd_simple.sample(env, env_id=env_id, policy_fn=policy_fn, 
                             load_dir=load_dir, save_path=save_path, 
                             manual=manual, visual=visual, num_trajs=num_trajs)
    env.close()
    return save_path


def exec_random_policy(env_id:str,
                       random_init=0,
                       save=False
                       ):
    '''Executes random policy on environment
    '''
    env = gym.make(env_id)

    if random_init==-1: 
        pass 
    elif random_init==0: 
        env.unwrapped.reset = return_random_reset(env_id, env, 
                              traj_seed=np.random.randint(low=0, high=2**32-1, size=1, dtype="int64"))

    reward_list = []
    obs_list = []
    acs_list = []
    for i in range(1000):
        obs, full_obs, acs, rew, _ = traj_segment_generator(env_id=env_id, 
                                                  env=env,
                                                  demo=None,
                                                  ac_gen=[], # empty actions, since random policy
                                                  random_pol_horizon = env._max_episode_steps,
                                                  random_policy = True)
        obs_list.append(full_obs)
        acs_list.append(acs)
        reward_list.append(rew)
    obs_list = np.concatenate(obs_list)
    acs_list = np.concatenate(acs_list)
    print("Random policy performance across 100 trajectories: average ", np.mean(reward_list), " std ", np.std(reward_list))
    if save:
        # save random performance 
        save_dir = f'../data/random_traj'
        if not osp.isdir(save_dir):
            os.makedirs(save_dir)
        np.savez(f'{save_dir}/random_pol_{env_id}', 
                 obs=obs_list, acs=acs_list,
                 rew_mean=np.mean(reward_list), rew_std=np.std(reward_list)) 

    env.close()


if __name__ == '__main__':
    args = argparser()
    logger.configure()

    if args.task=='train':
        ppo_params = {'adam_epsilon': args.adam_epsilon,
                      'clip_param': args.clip_param,
                      'entcoeff': args.entcoeff,
                      'gamma': args.gamma, 
                      'lam': args.lam}

        task_name = train(env_id=args.env_id, 
                          algo=params['algo'], 
                          seed=params['rng_seed'],
                          random_init=args.random_init,
                          traj_seed=args.traj_seed,
                          ckpt_dir=paths['model_ckpt_dir'], 
                          log_dir=paths['log_dir'],
                          **ppo_params)

        if args.task_name_save_path:
            with open(args.task_name_save_path, 'w+') as w: 
                w.write(task_name)

    elif args.task=='sample':
        num_trajs=100
        print(f"Sampling {num_trajs} trajectories from trained model with seed {args.traj_seed}")
        # sample 100 trajectories
        saved_traj_path = sample(env_id=args.env_id, 
                                 algo=params['algo'], 
                                 num_trajs=num_trajs,
                                 load_dir=args.load_model_path,  # HOW TO DECIDE WHICH MODEL TO LOAD???
                                 data_dir=paths['sample_save_dir'], 
                                 manual=False, 
                                 visual=False,
                                 seed=params['rng_seed'],
                                 random_init=args.random_init,
                                 traj_seed=args.traj_seed)
        print(f"Sampled trajectories for model with seed {args.traj_seed} \n can be found at {saved_traj_path}")

        print("Converting sampled trajectories to raw state only")
        get_joint_angles(env_id=args.env_id,
                         expert_demo_path=saved_traj_path,
                         paths=paths,
                         random_init=args.random_init
                         )

    elif args.task=='exec_random_policy': 
        exec_random_policy(env_id=args.env_id,
                           random_init=args.random_init,
                           save=True
                           )

