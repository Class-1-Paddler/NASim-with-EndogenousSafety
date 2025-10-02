""" 
The main Environment class for NASim: NASimEnv. 
The NASimEnv class is the main interface for agents interacting with NASim. 
"""

import gym
import numpy as np
from gym import spaces

from .state import State
from .render import Viewer
from .network import Network
from .observation import Observation
from .action import Action, FlatActionSpace, ParameterisedActionSpace


class NASimEnv(gym.Env):
    """
    A simulated computer network environment for pen-testing.
    Implements the OpenAI gym interface.

    Attributes
    ----------
    name : str
        the environment scenario name
    scenario : Scenario
        Scenario object, defining the properties of the environment
    action_space : FlatActionSpace or ParameterisedActionSpace
        Action space for environment.
        If flat_action=True then discrete integer actions,
        else parameterised MultiDiscrete actions.
    observation_space : gym.spaces.Box
        Observation space for environment.
        If flat_obs=True -> 1D vector, else 2D matrix.
    current_state : State
        The current state of the environment
    last_obs : Observation
        The last observation generated
    steps : int
        The number of steps since last reset
    """

    metadata = {'rendering.modes': ["readable"]}
    reward_range = (-float('inf'), float('inf'))

    action_space = None
    observation_space = None
    current_state = None
    last_obs = None

    def __init__(self, scenario, fully_obs=False, flat_actions=True, flat_obs=True):
        self.name = scenario.name
        self.scenario = scenario
        self.fully_obs = fully_obs
        self.flat_actions = flat_actions
        self.flat_obs = flat_obs

        self.network = Network(scenario)
        self.current_state = State.generate_initial_state(self.network)
        self._renderer = None
        self.reset()

        if self.flat_actions:
            self.action_space = FlatActionSpace(self.scenario)
        else:
            self.action_space = ParameterisedActionSpace(self.scenario)

        if self.flat_obs:
            obs_shape = self.last_obs.shape_flat()
        else:
            obs_shape = self.last_obs.shape()

        obs_low, obs_high = Observation.get_space_bounds(self.scenario)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, shape=obs_shape
        )
        self.steps = 0

    def reset(self):
        self.steps = 0
        self.current_state = self.network.reset(self.current_state)
        self.last_obs = self.current_state.get_initial_observation(self.fully_obs)
        if self.flat_obs:
            return self.last_obs.numpy_flat()
        return self.last_obs.numpy()

    def step(self, action):
        next_state, obs, reward, done, info = self.generative_step(
            self.current_state, action
        )
        self.current_state = next_state
        self.last_obs = obs

        if self.flat_obs:
            obs = obs.numpy_flat()
        else:
            obs = obs.numpy()

        self.steps += 1
        if not done and self.scenario.step_limit is not None:
            done = self.steps >= self.scenario.step_limit

        return obs, reward, done, info

    def generative_step(self, state, action):
        if not isinstance(action, Action):
            action = self.action_space.get_action(action)

        next_state, action_obs = self.network.perform_action(state, action)
        obs = next_state.get_observation(action, action_obs, self.fully_obs)
        done = self.goal_reached(next_state)
        reward = action_obs.value - action.cost
        return next_state, obs, reward, done, action_obs.info()

    def generate_random_initial_state(self):
        return State.generate_random_initial_state(self.network)

    def generate_initial_state(self):
        return State.generate_initial_state(self.network)

    def render(self, mode="readable", obs=None):
        if obs is None:
            obs = self.last_obs
        if not isinstance(obs, Observation):
            obs = Observation.from_numpy(obs, self.current_state.shape())
        if self._renderer is None:
            self._renderer = Viewer(self.network)
        if mode == "readable":
            self._renderer.render_readable(obs)
        else:
            raise NotImplementedError(
                f"Please choose correct render mode from: {self.metadata['rendering.modes']}"
            )

    def render_state(self, mode="readable", state=None):
        if state is None:
            state = self.current_state
        if not isinstance(state, State):
            state = State.from_numpy(
                state,
                self.current_state.shape(),
                self.current_state.host_num_map
            )
        if self._renderer is None:
            self._renderer = Viewer(self.network)
        if mode == "readable":
            self._renderer.render_readable_state(state)
        else:
            raise NotImplementedError(
                f"Please choose correct render mode from: {self.metadata['rendering.modes']}"
            )

    def render_action(self, action):
        if not isinstance(action, Action):
            action = self.action_space.get_action(action)
        print(action)

    def render_episode(self, episode, width=7, height=7):
        if self._renderer is None:
            self._renderer = Viewer(self.network)
        self._renderer.render_episode(episode, width, height)

    def render_network_graph(self, ax=None, show=False):
        if self._renderer is None:
            self._renderer = Viewer(self.network)
        state = self.current_state
        self._renderer.render_graph(state, ax, show)

    def get_minimum_actions(self):
        return self.network.get_minimal_steps()

    def get_action_mask(self):
        assert isinstance(self.action_space, FlatActionSpace), \
            "Can only use action mask function when using flat action space"
        mask = np.zeros(self.action_space.n, dtype=np.int64)
        for a_idx in range(self.action_space.n):
            action = self.action_space.get_action(a_idx)
            if self.network.host_discovered(action.target):
                mask[a_idx] = 1
        return mask

    def get_score_upper_bound(self):
        max_reward = self.network.get_total_sensitive_host_value()
        max_reward += self.network.get_total_discovery_value()
        max_reward -= self.network.get_minimal_steps()
        return max_reward

    def goal_reached(self, state=None):
        if state is None:
            state = self.current_state
        return self.network.all_sensitive_hosts_compromised(state)

    def __str__(self):
        output = [
            "NASimEnv:",
            f"name={self.name}",
            f"fully_obs={self.fully_obs}",
            f"flat_actions={self.flat_actions}",
            f"flat_obs={self.flat_obs}"
        ]
        return "\n ".join(output)
