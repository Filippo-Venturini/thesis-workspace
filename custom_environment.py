import math

import numpy as np
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_to_aec_wrapper

import magent2
from magent2.environments.magent_env import magent_parallel_env, make_env


default_map_size = 45
max_cycles_default = 1000
KILL_REWARD = 0
minimap_mode_default = False
default_reward_args = dict(
    step_reward=-0.005,
    dead_penalty=-0.0,
    attack_penalty=-0.0,
    attack_opponent_reward=0.0,
)


def parallel_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    render_mode=None,
    seed=None,
    **reward_args,
):
    env_reward_args = dict(**default_reward_args)
    env_reward_args.update(reward_args)
    return _parallel_env(
        map_size,
        minimap_mode,
        env_reward_args,
        max_cycles,
        extra_features,
        render_mode,
        seed,
    )


def raw_env(
    map_size=default_map_size,
    max_cycles=max_cycles_default,
    minimap_mode=minimap_mode_default,
    extra_features=False,
    seed=None,
    **reward_args,
):
    return parallel_to_aec_wrapper(
        parallel_env(
            map_size, max_cycles, minimap_mode, extra_features, seed=seed, **reward_args
        )
    )


env = make_env(raw_env)


def get_config(
    map_size,
    minimap_mode,
    seed,
    step_reward,
    dead_penalty,
    attack_penalty,
    attack_opponent_reward,
):
    gw = magent2.gridworld
    cfg = gw.Config()

    cfg.set({"map_width": map_size, "map_height": map_size})
    cfg.set({"minimap_mode": minimap_mode})
    cfg.set({"embedding_size": 10})
    if seed is not None:
        cfg.set({"seed": seed})

    options = {
        "width": 1,
        "length": 1,
        "hp": 10,
        "speed": 2,
        "view_range": gw.CircleRange(6),
        "attack_range": gw.CircleRange(1.5),
        "damage": 0,
        "kill_reward": KILL_REWARD,
        "step_recover": 0.0,
        "step_reward": step_reward,
        "dead_penalty": dead_penalty,
        "attack_penalty": attack_penalty,
    }
    small = cfg.register_agent_type("small", options)

    g0 = cfg.add_group(small)
    g1 = cfg.add_group(small)

    a = gw.AgentSymbol(g0, index="any")
    b = gw.AgentSymbol(g1, index="any")

    # reward shaping to encourage attack
    cfg.add_reward_rule(
        gw.Event(a, "attack", b), receiver=a, value=attack_opponent_reward
    )
    cfg.add_reward_rule(
        gw.Event(b, "attack", a), receiver=b, value=attack_opponent_reward
    )

    return cfg


class _parallel_env(magent_parallel_env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "battle_v4",
        "render_fps": 5,
    }

    def __init__(
        self,
        map_size,
        minimap_mode,
        reward_args,
        max_cycles,
        extra_features,
        render_mode=None,
        seed=None,
    ):
        EzPickle.__init__(
            self,
            map_size,
            minimap_mode,
            reward_args,
            max_cycles,
            extra_features,
            render_mode,
            seed,
        )
        assert map_size >= 12, "size of map must be at least 12"
        env = magent2.GridWorld(
            get_config(map_size, minimap_mode, seed, **reward_args), map_size=map_size
        )

        self.agentGroupID = 0

        reward_vals = np.array([KILL_REWARD] + list(reward_args.values()))
        reward_range = [
            np.minimum(reward_vals, 0).sum(),
            np.maximum(reward_vals, 0).sum(),
        ]
        names = ["red", "blue"]
        super().__init__(
            env,
            env.get_handles(),
            names,
            map_size,
            max_cycles,
            reward_range,
            minimap_mode,
            extra_features,
            render_mode,
        )

    def generate_map(self):
        env, map_size, handles = self.env, self.map_size, self.handles
        
        width = height = map_size
        
        center_x = width // 2
        center_y = height // 2
        
        # Aggiunta degli agenti 3x3 al centro della griglia
        for i in range(center_x - 1, center_x + 2):
            for j in range(center_y - 1, center_y + 2):
                env.add_agents(handles[self.agentGroupID], method="custom", pos=[[i, j, 0]])
