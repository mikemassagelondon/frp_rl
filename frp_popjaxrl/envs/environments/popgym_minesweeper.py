import jax
import jax.numpy as jnp
from jax import lax
from gymnax.environments import environment, spaces
from typing import Tuple, Optional
import chex
from flax import struct

@struct.dataclass
class EnvState:
    timestep: int
    mine_grid: jnp.ndarray
    neighbor_grid: jnp.ndarray

@struct.dataclass
class EnvParams:
    pass

def count_neighbors(grid):
    """Count neighbors using pure JAX operations without convolution."""
    rows, cols = grid.shape
    result = jnp.zeros_like(grid)
    
    # Define relative positions of neighbors
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Pad the grid to handle edges
    padded = jnp.pad(grid, pad_width=1, mode='constant', constant_values=0)
    
    # For each offset, shift the grid and add to result
    for dx, dy in offsets:
        # Extract the shifted window from padded array
        result = result + padded[1+dx:1+dx+rows, 1+dy:1+dy+cols]
    
    # Ensure result is int8
    return jnp.array(result, dtype=jnp.int8)

class MineSweeper(environment.Environment):
    def __init__(self, dims=(4, 4), num_mines=2):
        super().__init__()
        self.dims = dims
        self.num_mines = num_mines
        self.max_episode_length = dims[0] * dims[1] - num_mines
        self.success_reward_scale = 1 / self.max_episode_length
        self.fail_reward_scale = -0.5 - self.success_reward_scale
        self.bad_action_reward_scale = -0.5 / (self.max_episode_length - 2)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        mine = state.mine_grid[action] == 1
        viewed = state.mine_grid[action] == 2

        new_grid = state.mine_grid.at[action].set(2)

        reward = self.success_reward_scale
        reward = jnp.where(viewed, self.bad_action_reward_scale, reward)
        reward = jnp.where(mine, self.fail_reward_scale, reward)
    
        terminated = state.timestep == self.max_episode_length
        terminated = jnp.where(mine, True, terminated)
        terminated = jnp.logical_or(terminated, jnp.all(new_grid == 2))

        obs = jnp.zeros((self.num_mines,))
        obs = obs.at[state.neighbor_grid[action]].set(1)

        new_state = EnvState(
            timestep=state.timestep + 1,
            mine_grid=new_grid,
            neighbor_grid=state.neighbor_grid,
        )

        return obs, new_state, reward, terminated, {}

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        hidden_grid = jnp.zeros((self.dims[0] * self.dims[1],), dtype=jnp.int8)
        mines_flat = jax.random.choice(key, hidden_grid.shape[0], shape=(self.num_mines,), replace=False)
        hidden_grid = hidden_grid.at[mines_flat].set(1)
        hidden_grid = hidden_grid.reshape(self.dims)
        
        # Use pure JAX neighbor counting instead of convolution
        neighbor_grid = count_neighbors(hidden_grid)
        
        state = EnvState(
            timestep=0,
            mine_grid=jnp.ravel(hidden_grid),
            neighbor_grid=jnp.ravel(neighbor_grid),
        )

        return jnp.zeros((self.num_mines,)), state

    def action_space(
        self, params: Optional[EnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(self.dims[0] * self.dims[1])

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((self.num_mines,)), jnp.ones((self.num_mines,)), (self.num_mines,), dtype=jnp.float32)

class MineSweeperEasy(MineSweeper):
    def __init__(self):
        super().__init__(dims=(4, 4), num_mines=2)

class MineSweeperMedium(MineSweeper):
    def __init__(self):
        super().__init__(dims=(6, 6), num_mines=6)

class MineSweeperHard(MineSweeper):
    def __init__(self):
        super().__init__(dims=(8, 8), num_mines=10)
