import unittest


import jax.numpy as jnp
from envs.environments.metaaug.padding import *

class TestPadding(unittest.TestCase):
    def test_tiling(self):
        w = create_periodic_weight(2,128,64)
        x = jnp.array([0,1])
        y = x @ w
        #print(y)
        self.assertAlmostEqual(jnp.linalg.norm(y), 1.0)
        