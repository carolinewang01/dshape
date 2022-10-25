#!/bin/bash
python -m reward_shaping.hyperparam_search --env_id Reacher-v2 --task raw_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Swimmer-v2 --task raw_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Ant-v2 --task raw_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Walker2d-v2 --task raw_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Hopper-v2 --task raw_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id HalfCheetah-v2 --task raw_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Reacher-v2 --task full_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Swimmer-v2 --task full_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Ant-v2 --task full_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Walker2d-v2 --task full_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id Hopper-v2 --task full_demo_huber_reward
python -m reward_shaping.hyperparam_search --env_id HalfCheetah-v2 --task full_demo_huber_reward