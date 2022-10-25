import functools
from stable_baselines.her.her import HER as HER_orig
from utils.her_utils import HindsightExperienceReplayWrapper as CustomHERBufferWrapper, CustomHERGoalEnvWrapper

class CustomHER(HER_orig):
    """
    Hindsight Experience Replay (HER) https://arxiv.org/abs/1707.01495
    :param policy: (BasePolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param model_class: (OffPolicyRLModel) The off policy RL model to apply Hindsight Experience Replay
        currently supported: DQN, DDPG, SAC
    :param n_sampled_goal: (int)
    :param goal_selection_strategy: (GoalSelectionStrategy or str)
    """

    def __init__(self, policy, env, model_class, goal_selection_strategy, total_timesteps=None,
                 n_sampled_goal=4, expert_demo=None, time_feat=False, sin_cos_repr=False, raw=False, env_id=None,
                 *args, **kwargs):
        # make sure not to pass in goal_selection_strategy as str
        super(CustomHER, self).__init__(policy=policy, env=env, model_class=model_class, n_sampled_goal=n_sampled_goal,
                                        goal_selection_strategy=goal_selection_strategy, *args, **kwargs)
        self.total_timesteps = total_timesteps
        self.time_feat = time_feat
        self.sin_cos_repr = sin_cos_repr
        self.raw = raw
        self.env_id = env_id
        self.expert_demo = expert_demo
        self._redefine_replay_wrapper()

    def _redefine_replay_wrapper(self):
        """
        Create the replay buffer wrapper.
        """
        self.replay_wrapper = functools.partial(CustomHERBufferWrapper,
                                                n_sampled_goal=self.n_sampled_goal,
                                                goal_selection_strategy=self.goal_selection_strategy,
                                                wrapped_env=self.env,
                                                expert_demo=self.expert_demo,
                                                time_feat=self.time_feat,
                                                sin_cos_repr=self.sin_cos_repr,
                                                raw=self.raw,
                                                env_id=self.env_id,
                                                total_timesteps=self.total_timesteps)

    def learn(self, callback=None, log_interval=100, tb_log_name="HER",
              save_checkpoints=True, save_at_end=True, save_interval=200, save_name=None, save_path=None,
              reset_num_timesteps=True):
        return self.model.learn(self.total_timesteps, callback=callback, log_interval=log_interval,
                                tb_log_name=tb_log_name,
                                save_checkpoints=save_checkpoints,
                                save_at_end=save_at_end,
                                save_interval=save_interval,
                                save_name=save_name,
                                save_path=save_path,
                                # set to False to resume curve on a tensorboard
                                reset_num_timesteps=reset_num_timesteps,
                                replay_wrapper=self.replay_wrapper)

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        return self.model.pretrain(dataset=dataset, n_epochs=n_epochs, learning_rate=learning_rate,
                 adam_epsilon=adam_epsilon, val_interval=val_interval)

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        data, _ = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=env, model_class=data['model_class'],
                    n_sampled_goal=data['n_sampled_goal'],
                    goal_selection_strategy=data['goal_selection_strategy'],
                    _init_setup_model=False, )
        model.__dict__['observation_space'] = data['her_obs_space']
        model.__dict__['action_space'] = data['her_action_space']

        model.model = data['model_class'].load(
            load_path, model.get_env(), **kwargs)
        model.model._save_to_file = model._save_to_file
        return model
