import unittest
import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from frp_popjaxrl.frp.orthogonal import get_weight_matrix

class TestWeightMatrixFunctions(unittest.TestCase):
    def setUp(self):
        # Set a fixed random seed for reproducibility
        self.key = jax.random.PRNGKey(42)
        
        # Create mock data
        self.num_words = 10
        self.input_dim = 4
        self.output_dim = 8
        
        # Create random words array
        self.words = jax.random.normal(
            self.key, 
            (self.num_words, self.input_dim, self.output_dim)
        )
        
    
    def test_get_weight_matrix_with_jit(self):
        """Test that the new get_weight_matrix function works with jax.jit."""
        
        # Create a JIT-compiled version of the function
        jitted_func = jax.jit(
            lambda words, env_index: get_weight_matrix(
                words, env_index, self.input_dim, self.output_dim
            )
        )
        
        # Test with different environment indices
        for env_index in range(self.num_words):
            # This should not raise any JIT-related errors
            result = jitted_func(self.words, env_index)
            
            # Check that the result has the expected shape
            self.assertEqual(
                result.shape, 
                (self.input_dim, self.output_dim),
                f"Incorrect shape for env_index={env_index}"
            )
    
    def test_get_weight_matrix_with_traced_index(self):
        """Test that the function works with traced indices in jax.jit."""
        
        # Create a function that uses the env_index as a parameter
        def func_with_index(words, env_index):
            return get_weight_matrix(words, env_index, self.input_dim, self.output_dim)
        
        # JIT-compile the function
        jitted_func = jax.jit(func_with_index)
        
        # Test with different environment indices
        for env_index in range(self.num_words):
            # Get results from both the original and JIT-compiled functions
            expected_result = func_with_index(self.words, env_index)
            actual_result = jitted_func(self.words, env_index)
            
            # Check that the results are identical
            np.testing.assert_allclose(
                expected_result, actual_result, 
                rtol=1e-7, atol=1e-7,
                err_msg=f"JIT results differ for env_index={env_index}"
            )
    
    def test_different_dimensions(self):
        """Test the function with different input and output dimensions."""
        
        # Test with different dimensions
        test_cases = [
            (2, 4),   # Small input, small output
            (8, 16),  # Medium input, medium output
            (32, 64), # Large input, large output
            (4, 16),  # Small input, medium output
            (16, 4)   # Medium input, small output
        ]
        
        for input_dim, output_dim in test_cases:
            # Create words with the specified dimensions
            key, _ = jax.random.split(self.key)
            words = jax.random.normal(
                key, 
                (self.num_words, input_dim, output_dim)
            )
            
            # Test with a specific environment index
            env_index = 3
            
            # Define the original function using dynamic_slice
            def original_get_weight(words, env_index, input_dim, output_dim):
                return jnp.sqrt(2) * jax.lax.dynamic_slice(
                    words, 
                    (env_index, 0, 0), 
                    (1, input_dim, output_dim)
                )[0]
            
            # Get results from both functions
            original_result = original_get_weight(
                words, env_index, input_dim, output_dim
            )
            
            new_result = get_weight_matrix(
                words, env_index, input_dim, output_dim
            )
            
            # Check that the results are identical
            np.testing.assert_allclose(
                original_result, new_result, 
                rtol=1e-7, atol=1e-7,
                err_msg=f"Results differ for dimensions ({input_dim}, {output_dim})"
            )
    
    def test_edge_cases(self):
        """Test the function with edge cases."""
        
        # Test with very large values
        large_words = self.words * 1e6
        
        # Test with very small values
        small_words = self.words * 1e-6
        
        # Test with a mix of positive and negative values
        mixed_words = self.words * jnp.array([-1, 1])[jax.random.randint(
            self.key, (self.num_words, self.input_dim, self.output_dim), 0, 2
        )]
        
        # Test with all zeros
        zero_words = jnp.zeros_like(self.words)
        
        # Test with all ones
        one_words = jnp.ones_like(self.words)
        
        # Test with NaN values (should be avoided in practice, but testing for robustness)
        # nan_words = self.words.at[0, 0, 0].set(jnp.nan)
        
        # Test cases
        test_cases = [
            (large_words, "large values"),
            (small_words, "small values"),
            (mixed_words, "mixed values"),
            (zero_words, "zeros"),
            (one_words, "ones")
            # Skipping NaN test as it's expected to fail
        ]
        
        for words, case_name in test_cases:
            # Test with a specific environment index
            env_index = 3
            
            # Define the original function using dynamic_slice
            def original_get_weight(words, env_index, input_dim, output_dim):
                return jnp.sqrt(2) * jax.lax.dynamic_slice(
                    words, 
                    (env_index, 0, 0), 
                    (1, input_dim, output_dim)
                )[0]
            
            # Get results from both functions
            original_result = original_get_weight(
                words, env_index, self.input_dim, self.output_dim
            )
            
            new_result = get_weight_matrix(
                words, env_index, self.input_dim, self.output_dim
            )
            
            # Check that the results are identical
            np.testing.assert_allclose(
                original_result, new_result, 
                rtol=1e-7, atol=1e-7,
                err_msg=f"Results differ for case: {case_name}"
            )
    
    def test_gradients(self):
        """Test that the function is differentiable and gradients match."""
        # Define a loss function that uses the weight matrix
        def loss_fn_original(words, env_index):
            weight = jnp.sqrt(2) * jax.lax.dynamic_slice(
                words, 
                (env_index, 0, 0), 
                (1, self.input_dim, self.output_dim)
            )[0]
            return jnp.sum(weight**2)
        
        def loss_fn_new(words, env_index):
            weight = get_weight_matrix(
                words, env_index, self.input_dim, self.output_dim
            )
            return jnp.sum(weight**2)
        
        # Compute gradients with respect to words
        grad_original = jax.grad(loss_fn_original, argnums=0)(self.words, 3)
        grad_new = jax.grad(loss_fn_new, argnums=0)(self.words, 3)
        
        # Check that the gradients are identical
        np.testing.assert_allclose(
            grad_original, grad_new, 
            rtol=1e-7, atol=1e-7,
            err_msg="Gradients differ"
        )
    
    def test_nested_jit(self):
        """Test the function with nested JIT compilations."""
        
        # Define a function that uses get_weight_matrix inside
        def inner_func(words, env_index):
            return get_weight_matrix(words, env_index, self.input_dim, self.output_dim)
        
        # JIT-compile the inner function
        jitted_inner = jax.jit(inner_func)
        
        # Define an outer function that uses the JIT-compiled inner function
        def outer_func(words, env_index):
            weight = jitted_inner(words, env_index)
            return jnp.sum(weight**2)
        
        # JIT-compile the outer function
        jitted_outer = jax.jit(outer_func)
        
        # Test with a specific environment index
        env_index = 3
        
        # This should not raise any JIT-related errors
        result = jitted_outer(self.words, env_index)
        
        # Check that the result is a scalar
        self.assertEqual(result.shape, (), "Incorrect shape for nested JIT result")
    
    def test_vmap_compatibility(self):
        """Test that the function works with jax.vmap."""
        
        # Define a function that uses get_weight_matrix
        def func(words, env_index):
            return get_weight_matrix(words, env_index, self.input_dim, self.output_dim)
        
        # Create a vmapped version of the function that processes multiple env_indices at once
        vmapped_func = jax.vmap(func, in_axes=(None, 0))
        
        # Create an array of environment indices
        env_indices = jnp.array([0, 2, 4, 6, 8])
        
        # Apply the vmapped function
        results = vmapped_func(self.words, env_indices)
        
        # Check that the results have the expected shape
        self.assertEqual(
            results.shape, 
            (len(env_indices), self.input_dim, self.output_dim),
            "Incorrect shape for vmapped results"
        )
        
        # Check that each result matches the expected output
        for i, env_index in enumerate(env_indices):
            expected_result = func(self.words, env_index)
            np.testing.assert_allclose(
                results[i], expected_result, 
                rtol=1e-7, atol=1e-7,
                err_msg=f"Vmapped result differs for env_index={env_index}"
            )
    
    def test_jit_with_vmap(self):
        """Test that the function works with a combination of jax.jit and jax.vmap."""
        
        # Define a function that uses get_weight_matrix
        def func(words, env_index):
            return get_weight_matrix(words, env_index, self.input_dim, self.output_dim)
        
        # Create a vmapped version of the function
        vmapped_func = jax.vmap(func, in_axes=(None, 0))
        
        # JIT-compile the vmapped function
        jitted_vmapped_func = jax.jit(vmapped_func)
        
        # Create an array of environment indices
        env_indices = jnp.array([0, 2, 4, 6, 8])
        
        # Apply the JIT-compiled vmapped function
        results = jitted_vmapped_func(self.words, env_indices)
        
        # Check that the results have the expected shape
        self.assertEqual(
            results.shape, 
            (len(env_indices), self.input_dim, self.output_dim),
            "Incorrect shape for JIT-compiled vmapped results"
        )
        
        # Check that each result matches the expected output
        for i, env_index in enumerate(env_indices):
            expected_result = func(self.words, env_index)
            np.testing.assert_allclose(
                results[i], expected_result, 
                rtol=1e-7, atol=1e-7,
                err_msg=f"JIT-compiled vmapped result differs for env_index={env_index}"
            )

if __name__ == "__main__":
    unittest.main()
