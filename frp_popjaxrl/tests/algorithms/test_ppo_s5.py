import unittest
import jax
from algorithms.ppo_s5_in_context import make_train
from envs.wrappers import AliasPrevActionV2
from envs import make
import wandb

class TestPPOS5(unittest.TestCase):
    def test_on_MetaCartPole(self):
        wandb.init()
        rng = jax.random.PRNGKey(30)

        env_name = "NoisyStatelessMetaCartPole"
        env_kwargs={"meta_dim":2,
                    "rng": rng,
                    "meta_max_depth":8,
                    "meta_depth": 2,
                    "meta_with_adjoint": 1
                    } 
        env, env_params = make(env_name, **env_kwargs) 

        eval_env_kwargs = env_kwargs
        eval_env_kwargs["meta_const_aug"]="padding"

        eval_env, eval_env_params = make(env_name, **eval_env_kwargs)


        config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e4, 
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES":1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 1.0,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "ENV": AliasPrevActionV2(env),
        "ENV_PARAMS": env_params,
        "EVAL_ENV": AliasPrevActionV2(eval_env),
        "EVAL_ENV_PARAMS": eval_env_params,
        "ANNEAL_LR": False,
        "DEBUG": True,
        "S5_D_MODEL": 256,
        "S5_SSM_SIZE": 256,
        "S5_N_LAYERS": 1,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": False,
        "S5_PRENORM": False,
        "S5_DO_GTRXL_NORM": False,
        }

        jit_train = jax.jit(make_train(config))

        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)
        # runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate, _rng) 
        # train_state = TrainState.create(
        #    apply_fn=network.apply,
        #    params=network_params,
        #    tx=tx,
        #)
    def test_on_eval_uniform(self):
        wandb.init()
        rng = jax.random.PRNGKey(30)

        env_name = "NoisyStatelessMetaCartPole"
        env_kwargs={"meta_dim":2,
                    "rng": rng,
                    "meta_max_depth":8,
                    "meta_depth": 2,
                    "meta_with_adjoint": 1
                    } 
        env, env_params = make(env_name, **env_kwargs) 

        eval_env_kwargs = env_kwargs
        eval_env_kwargs["meta_const_aug"]=False
        eval_env_kwargs["meta_with_adjoint"]=0
        eval_env_kwargs["meta_depth"]=1
        rng, _rng = jax.random.split(rng)
        eval_env_kwargs["rng"] = _rng

        

        eval_env, eval_env_params = make(env_name, **eval_env_kwargs)


        config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 1,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1e4, 
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES":1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 1.0,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "ENV": AliasPrevActionV2(env),
        "ENV_PARAMS": env_params,
        "EVAL_ENV": AliasPrevActionV2(eval_env),
        "EVAL_ENV_PARAMS": eval_env_params,
        "ANNEAL_LR": False,
        "DEBUG": True,
        "S5_D_MODEL": 256,
        "S5_SSM_SIZE": 256,
        "S5_N_LAYERS": 1,
        "S5_BLOCKS": 1,
        "S5_ACTIVATION": "full_glu",
        "S5_DO_NORM": False,
        "S5_PRENORM": False,
        "S5_DO_GTRXL_NORM": False,
        }

        jit_train = jax.jit(make_train(config))

        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)
        # runner_state = (train_state, env_state, obsv, jnp.zeros((config["NUM_ENVS"]), dtype=bool), init_hstate, _rng) 
        # train_state = TrainState.create(
        #    apply_fn=network.apply,
        #    params=network_params,
        #    tx=tx,
        #)


