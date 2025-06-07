import jax.numpy as jnp
import jax
#import flax.linen as nn
from gymnax.environments import environment, spaces
from flax import struct
import chex
from typing import Tuple, Optional, Any, Dict
#from flax.linen.initializers import constant, orthogonal
#import numpy as np
from .popgym_cartpole import NoisyStatelessCartPole, EnvParams, EnvState
from flax.core import freeze
from frp.orthogonal import create_orthogonal_matrices, create_words, random_choice, MetaAugNetwork, detect_identity_matrices
#from .metaaug.padding import PaddingAugmentatoin

from .metaaug.padding import create_periodic_weight

@struct.dataclass
class MetaEnvState:
    ## not to be sharing 
    #obs_params: chex.Array
    env_index: int
    obs_words: chex.Array
    trial_num: int
    total_steps: int
    env_state: EnvState
    init_state: Optional[chex.Array]
    init_obs: Optional[chex.Array]
    
@struct.dataclass
class MetaEnvParams:
    ## should be sharing
    num_trials_per_episode: int = 16
    env_params: EnvParams = EnvParams()

class NoisyStatelessMetaCartPole(environment.Environment):
    def __init__(self, **meta_kwargs):
        super().__init__()
        self.env = NoisyStatelessCartPole(max_steps_in_episode=200, noise_sigma=0.0)
        self.input_dim = self.env.obs_shape[0]

        # Meta-learning specific parameters
        self.meta_depth = meta_kwargs.get('meta_depth', 1)
        self.meta_const_aug = meta_kwargs.get('meta_const_aug', False)
        self.meta_dim = meta_kwargs.get('meta_dim', 4)
        self.meta_max_depth = meta_kwargs.get('meta_max_depth', 2)
        self.meta_with_adjoint = meta_kwargs.get('meta_with_adjoint', False)
        self.rng = meta_kwargs.get('meta_rng', jax.random.PRNGKey(42))

        if self.meta_const_aug == "padding":
            self.eval_weight =   jnp.eye(self.meta_dim)[:self.input_dim,:]       
        elif self.meta_const_aug == "tiling":
            self.eval_weight = create_periodic_weight(input_dim=self.input_dim, output_dim=self.meta_dim, period=round(self.meta_dim/2)) 
        else:
            ...

        self.obs_shape = (self.meta_dim+3,)

        # Initialize meta-augmentation
        self.initialize_meta_augmentation(key=self.rng)

    def create_words(self, key):
        matrices = create_orthogonal_matrices(
            key, 
            self.meta_depth, 
            size=self.meta_dim, 
            max_depth=self.meta_max_depth, 
            with_adjoint=self.meta_with_adjoint
        )
        words = create_words(
            matrices, 
            self.meta_depth, 
            out_size=self.meta_dim, 
            max_depth=self.meta_max_depth
        )
        return words

    def initialize_meta_augmentation(self, key):
        self.words = self.create_words(key)
        # exclude before cutof
        self.exclude = detect_identity_matrices(self.words)        
        self.words = self.words[:, :self.input_dim, :]        
        self.total_words = self.words.shape[0]
        #self.obs_aug = MetaAugNetwork(out_size=self.meta_dim, words=self.words)
        print(f"Valid total_words: {self.total_words - len(self.exclude)}")
    
    
    def reset_meta_augmentation(self, key, env_obs):
        self.words = self.create_words(key)
        self.words = self.words[:, :self.input_dim, :]       
 
        

    @property
    def default_params(self) -> MetaEnvParams:
        return MetaEnvParams()
    
    def step_env(
        self, key: chex.PRNGKey, state: MetaEnvState, action: int, params: MetaEnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        key, key_reset = jax.random.split(key)

        env_obs_st, env_state_st, reward, env_done, info = self.env.step_env(key, state.env_state, action, params.env_params)
        # env_obs_re, env_state_re = self.env.reset_env(key_reset, params.env_params)
        env_obs_re, env_state_re = state.init_obs, state.init_state

        env_state = jax.tree_util.tree_map(
            lambda x, y: jax.lax.select(env_done, x, y), env_state_re, env_state_st
        )
        env_obs = jax.lax.select(env_done, env_obs_re, env_obs_st)
        def callback(is_eval):
            print("eval:" , is_eval)
        if self.meta_const_aug in ["padding", "tiling"]:
            env_obs = (env_obs[None,:] @ self.eval_weight )[0]
            
        else:
            weight = jnp.sqrt(2)*jax.lax.dynamic_slice(state.obs_words, (state.env_index, 0, 0), (1, 2, self.meta_dim))[0]
            #env_obs = self.obs_aug.apply(state.obs_params, env_obs[None,:])[0]
            env_obs = (env_obs[None,:] @ weight)[0]
        # print(env_obs.shape) (2,)
        #env_obs = self.obs_aug.apply(state.obs_params, env_obs[None,:])[0]
        # print(env_obs.shape) (4,)


        # trail num increases when env has done 
        trial_num = state.trial_num + env_done
        total_steps = state.total_steps + 1
        done = trial_num >= params.num_trials_per_episode

        state = MetaEnvState(
            env_index= state.env_index,
            obs_words=state.obs_words,
            trial_num=trial_num,
            total_steps=total_steps,
            env_state=env_state,
            init_state=state.init_state,
            init_obs=state.init_obs,
        )

        obs = jnp.concatenate([env_obs, jnp.array([action, env_done, 0.0])])

        return (
            obs,
            state,
            reward,
            done,
            info,
        )
    
    def reset_env(
        self, key: chex.PRNGKey, params: MetaEnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """
        reset_env is called in envrioment.Enviroment.step
        For parallel computing for envrioments, it is called in *each* step.
        Meta state is replaced when epoisode is done.
        """
        #env_key, obs_key= jax.random.split(key, 2)
        env_key, obs_key, index_key = jax.random.split(key, 3)
        env_obs, env_state = self.env.reset_env(env_key, params.env_params)

        if len(self.exclude) ==0:
            env_index = jax.random.randint(index_key, (), 0, self.total_words).astype(jnp.int32)
        else:
            env_index = random_choice(index_key, 
                                  total_words=self.total_words, 
                                  exclude=self.exclude).astype(jnp.int32)
        #obs_params = self.obs_aug.init(obs_key, env_obs[None,:])
        
        #test = self.obs_aug.apply(obs_params, env_obs[None,:])[0]
        state = MetaEnvState(
            env_index= env_index,
            obs_words=self.words,
            trial_num=0,
            total_steps=0,
            env_state=env_state,
            init_state=env_state,
            init_obs=env_obs,
        )
        weight = jnp.sqrt(2)*jax.lax.dynamic_slice(self.words, (env_index, 0, 0), (1, 2, self.meta_dim))[0]
        test = (env_obs[None,:] @ weight )[0]
        obs = jnp.concatenate([test, jnp.array([0.0, 0.0, 1.0])])

        return obs, state
    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 2

    def action_space(
        self, params: Optional[MetaEnvParams] = None
    ) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(2)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        high = jnp.ones([self.meta_dim+3])
        return spaces.Box(-high, high, (self.meta_dim+3,), dtype=jnp.float32)
