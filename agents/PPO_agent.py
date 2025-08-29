from operator import neg

import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import TimeLimit, TransformReward
from mpcrl.wrappers.envs import MonitorEpisodes
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from greenhouse.env import LettuceGreenHouse
from greenhouse.model import Model


class AugmentedObservationWrapper(ObservationWrapper):
    """Wrapper that augments the observation with the one-step yield improvement,
    weather disturbances and previous input."""

    def __init__(self, env: LettuceGreenHouse) -> None:
        super().__init__(env)
        nx, nd = env.get_wrapper_attr("nx"), env.get_wrapper_attr("nd")
        act = env.action_space
        lby, uby = [-np.inf, 0.0, -273.15, 0.0], np.full(nx, np.inf)
        lbd, ubd = [0.0, 0.0, -273.15, 0.0], np.full(nd, np.inf)
        self.observation_space = spaces.Box(
            np.concatenate((lby, lbd, act.low)),
            np.concatenate((uby, ubd, act.high)),
            dtype=act.dtype,
            seed=act.np_random,
        )

    def observation(self, state: np.ndarray) -> np.ndarray:
        env: LettuceGreenHouse = self.env.unwrapped
        output = Model.output(state, env.p)
        output[0] -= env.previous_lettuce_yield
        new_state = np.concatenate(
            (output, env.current_disturbance, env.previous_action), axis=None
        )
        assert self.observation_space.contains(new_state), "Invalid observation."
        return new_state


def make_env(
    gamma: float, days: int, evaluation: bool = False, seed: int | None = None
) -> tuple[VecNormalize, int]:
    """Creates and appropriately wraps the greenhouse env for training or evaluation,
    and returns also the number of steps per episode."""
    greenhouse = LettuceGreenHouse(
        growing_days=days,
        model_type="continuous",
        cost_parameters_dict={
            "c_u": [10, 1, 1],
            "c_y": 0.0,
            "c_dy": 100,
            "w_y": np.full((1, 4), 1e5),
        },
        disturbance_profiles_type="single",
        noisy_disturbance=True,
        testing="none",
        clip_action_variation=True,
    )
    max_episode_steps = days * LettuceGreenHouse.steps_per_day
    env = MonitorEpisodes(TimeLimit(greenhouse, max_episode_steps=max_episode_steps))

    # make the env compatible with RL - augment state, and flip cost into reward
    env = TransformReward(AugmentedObservationWrapper(env), neg)

    # add wrappers for SB3
    env = Monitor(env)
    venv = DummyVecEnv([lambda: env])
    venv.set_options({"initial_day": 0, "noise_coeff": 1.0})
    venv = VecNormalize(
        venv, not evaluation, clip_obs=np.inf, clip_reward=np.inf, gamma=gamma
    )
    venv.seed(seed)
    return venv, min(max_episode_steps, greenhouse.yield_step)


def train_ppo(
    agent_num: int,
    episodes: int,
    days_per_episode: int,
    learning_rate: float,
    gradient_threshold: float,   # maps to max_grad_norm
    l2_regularization: float,
    batch_size: int,
    gamma: float,
    seed: int,
    device: str,
    verbose: int,
) -> tuple[MonitorEpisodes, MonitorEpisodes]:
    """Trains a PPO agent on the greenhouse environment."""
    set_random_seed(seed, using_cuda=device.startswith("cuda"))

    # create the model
    env, steps_per_episode = make_env(gamma, days_per_episode, seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gamma=gamma,
        n_steps=3840,  # 60 episodes * 64 steps
        max_grad_norm=gradient_threshold,
        policy_kwargs={
            "net_arch": [256, 256],
            "optimizer_kwargs": {"weight_decay": l2_regularization},
        },
        verbose=verbose,
        seed=seed,
        device=device,
    )

    # unique log dir per agent
    model.set_logger(configure(".", ["log"]))

    # eval callback (frozen normalization)
    eval_env, _ = make_env(gamma, days_per_episode, evaluation=True, seed=seed)
    eval_env.training = False
    eval_env.norm_reward = False
    cb = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=20,
        eval_freq=max(1, steps_per_episode * 50),  # every ~50 episodes
        best_model_save_path=f"./logs/ppo_agent_{agent_num}",
        log_path=f"./logs/ppo_agent_{agent_num}",
        verbose=verbose,
    )

    # train
    total_timesteps = steps_per_episode * episodes
    model.learn(total_timesteps=total_timesteps, log_interval=1, callback=cb)

    # save agent + env (with normalizations)
    env = model.get_env()
    env.save(f"ppo_env_{agent_num}.pkl")
    model.save(f"ppo_agent_{agent_num}")

    # unwrap helper
    def unwrap_env(venv):
        e = venv.envs[0]
        while hasattr(e, "env"):
            e = e.env
        return e

    return unwrap_env(env), unwrap_env(eval_env)
