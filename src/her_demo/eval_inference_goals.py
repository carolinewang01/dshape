import os
import numpy as np
from types import SimpleNamespace

from her_demo.her_custom import CustomHER as HER
from her_demo.her_sac_demo import make_custom_env
from utils.load_confs import load_parameters, load_paths
from utils.stable_baselines_helpers import evaluate_policy
from utils.helpers import get_demo


paths = load_paths()
params = load_parameters()
EXPT_NAME = params["her_demo"]["expt_name"]
RESULTS_DIR = paths['rl_demo']['results_dir']
ENV_IDS = ['Swimmer-v2', 'HalfCheetah-v2', 'Ant-v2', 'Hopper-v2']

def normalizer(x:np.ndarray, max_value:float, min_value:float):
    return (x - min_value) / (max_value - min_value)


if __name__ == '__main__':
    # get normalization values
    random_perf = {
                   'HalfCheetah-v2': {"ret": -282.7423069}, 
                   'Ant-v2': {"ret": -54.35667815}, 
                   'Swimmer-v2': {"ret": 1.4335161}, 
                   'Hopper-v2': {"ret":18.14677599}
                }

    expert_perf = {}
    for env_id in ENV_IDS:
        expert_perf[env_id] = {}
        expert_perf[env_id]["optimal"] = {}
        demo_algo = params["her_demo"]["demo_algo"][env_id.replace("-v2", "").lower()]["optimal"]
        data = np.load(f"../data/expert_joint_angle_traj/original_init/joint_angles_{demo_algo}.{env_id}.seed=None.npz")
        expert_perf[env_id]["optimal"] = data['rew']
    print("EXPERT PERF IS ", expert_perf)

    args = SimpleNamespace(**{"raw": True,
                              "time_feat": True,
                              "displace_t": 1,
                              # var used for naming purposes only
                              "rew": "env",
                              "goal_strat": "episode",
                              "demo_level": None,
                              "n_sampled_goal": 10
                              })

    model_basename = "her_demo_rew=potential_dense_raw=True_termphi0=True_goal-strat=episode_n-goal=10_time-feat=True_displace-t=1_demo=medium"

    results = {}
    for env_id in ENV_IDS:
        results[env_id] = {}
        args.env_id = env_id

        for demo_level in ["optimal", "medium", "worst", "random"]:
            print(f"Loading {args.env_id} env with {demo_level} demonstration. ")
            # setup env
            demo_algo = params["her_demo"]["demo_algo"][args.env_id.replace("-v2", "").lower()][demo_level]
            expert_demo = get_demo(args.env_id, demo_algo=demo_algo,
                                   raw=True, shuffle=False, time_feat=args.time_feat)
            env, eval_env = make_custom_env(args, expert_demo)

            for run_id in [1,2,3,4,5]:
                print(f"Loading model {run_id} for {env_id}...")
                model_path = os.path.join(RESULTS_DIR, model_basename, "checkpoint", f"sac_{env_id}_{run_id}", "best_model.zip")
                model = HER.load(model_path, env=None)

                # eval
                ep_rews, _, _ = evaluate_policy(model, eval_env, 
                    n_eval_episodes=64, deterministic=True, return_episode_rewards=True)
                ep_rews = normalizer(np.array(ep_rews), 
                                     max_value=expert_perf[env_id]["optimal"], 
                                     min_value=random_perf[env_id]["ret"]
                                   )

                results[env_id][demo_level] = {"mean": np.mean(ep_rews),
                                               "std": np.std(ep_rews)}
            eval_env.close()
    print(results)

