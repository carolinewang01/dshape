import os, sys
import argparse
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from poc.agent import Q_Agent
from poc.poc_utils import extend_demo
from poc.run_agent import make_env
from utils.helpers import str2bool
from utils.load_confs import load_parameters, load_paths

paths = load_paths()
params = load_parameters()

# def argparser():
#     parser = argparse.ArgumentParser("Running zero shot eval")
#     parser.add_argument('--demo_style', type=str, choices=["lower", "middle", "upper"], default=None)
#     return parser.parse_args()


if __name__ == '__main__':
    # args = argparser()
    demo_attrs = {
        # "train_condition": {"demo_goal": None, "demo_style": "lower"},
        # "test_demo_style": {"demo_goal": None, "demo_style": "upper"},
        "test_demo_lower": {"demo_goal": (0, 0), "demo_style": "lower"},
        "test_demo_middle": {"demo_goal": (0, 0), "demo_style": "middle"},
        "test_demo_upper": {"demo_goal": (0, 0), "demo_style": "upper"},

    }

    for gridworld_size in [10, 
                           20, 30
                           ]:
        print(f"\nGRIDWORLD SIZE {gridworld_size}")
        for expt_name, demo_dict in demo_attrs.items():
            demo_dict["demo_goal"] = (0, gridworld_size-2)
            print(f"{expt_name}, demo goal is {demo_dict['demo_goal']}")
            env, opt_rew = make_env(gridworld_size=gridworld_size, 
                                    gridworld_goal=demo_dict["demo_goal"], # have true env goal match demo goal for eval only
                                    reward_base_value=-1,
                                    reward_type="pbrs_demo", 
                                    demo_style=demo_dict["demo_style"], 
                                    demo_extend_type="extend_last", demo_extend_num=None,
                                    demo_goal=demo_dict["demo_goal"],
                                    negate_potential=False, 
                                    state_aug=True, time_feat=True,
                                    max_steps_per_episode=500)
            # print("OPT REW IS ", opt_rew)

            for agent_type in ["dshape", "pbrs+state-aug"]:
                agent = Q_Agent(env, 
                                # epsilon=0.2, alpha=0.1,
                                # init_value=0,
                                n_episodes=500, 
                                max_steps_per_episode=500, 
                                eval_freq=50,
                                reward_modified=True,
                                # use_buffer=True, buffer_size=buffer_size,
                                # relabel=True,  
                                # n_sampled_goal=3, # can only be true if state_aug and potential_shaping is also true
                                # updates_per_step=20, 
                                # show_progress=True,
                                save_policy=False,
                                eval_only=True
                                )

                states_all = []
                rews_all = []
                for agent_idx in range(30):
                    eval_rews = []
                    agent.load_agent(load_path=f"/scratch/cluster/clw4542/gridworld_results/{agent_type}_world=basic_size={gridworld_size}/trial={agent_idx}/q-table.npz")
                    # run 20 episode
                    for i in range(50):
                        cumulative_env_reward, _, _, states = agent.run_episode(test_mode=True)
                        eval_rews.append(cumulative_env_reward)
                        rews_all.append(cumulative_env_reward)
                        states_all += states
                    # print(f"EVAL REW FOR AGENT {agent_idx}: ", np.mean(eval_rews))
                print(f"{agent_type} AGENT; MEAN REW {np.mean(rews_all)}, STD REW {np.std(rews_all)}")

    # plot state visitation
    # state_visitation = env.unwrapped.traj_on_map(states_all)
    # # plot state visitation
    # fig, ax = plt.subplots(1,1, figsize=(4, 3))
    # # ax[0].axhline(opt_rew, label="optimal reward", linestyle="--")

    # cmap = copy.copy(matplotlib.cm.get_cmap('Blues'))
    # ax.imshow(state_visitation, cmap=cmap, interpolation='none')
    # ax.set_title("state visitation")

    # plt.show()