import jax
import jax.numpy as jnp
import numpy as np
import pytest
import time

@pytest.fixture
def random_key():
    """Fixture to provide a JAX random key"""
    return jax.random.PRNGKey(0)

@pytest.fixture
def small_matrices():
    """Fixture to provide small test matrices"""
    x = jnp.array([[1, 2], [3, 4]])
    y = jnp.array([[5, 6], [7, 8]])
    return x, y

def test_jax_setup():
    """Test JAX configuration and available devices"""
    assert jax.__version__ is not None, "JAX version should be available"
    assert len(jax.devices()) > 0, "At least one device should be available"
    backend = jax.lib.xla_bridge.get_backend().platform
    assert backend in ["cpu", "gpu", "tpu"], f"Unexpected backend: {backend}"

def test_basic_matrix_operations(small_matrices):
    """Test basic matrix operations with small matrices"""
    x, y = small_matrices
    
    # Test addition
    addition = x + y
    expected_addition = jnp.array([[6, 8], [10, 12]])
    assert jnp.allclose(addition, expected_addition), "Matrix addition failed"
    
    # Test element-wise multiplication
    multiplication = x * y
    expected_multiplication = jnp.array([[5, 12], [21, 32]])
    assert jnp.allclose(multiplication, expected_multiplication), "Element-wise multiplication failed"
    
    # Test matrix multiplication
    matmul = jnp.matmul(x, y)
    expected_matmul = jnp.array([[19, 22], [43, 50]])
    assert jnp.allclose(matmul, expected_matmul), "Matrix multiplication failed"

def test_large_matrix_performance(random_key):
    """Test performance of large matrix operations"""
    size = 1000
    key1, key2 = jax.random.split(random_key)
    large_x = jax.random.normal(key1, shape=(size, size))
    large_y = jax.random.normal(key2, shape=(size, size))
    
    # Warm-up run
    _ = jnp.matmul(large_x, large_y)
    
    # Timed run
    start_time = time.time()
    result = jnp.matmul(large_x, large_y)
    gpu_time = time.time() - start_time
    
    # Basic shape check
    assert result.shape == (size, size), "Result matrix has incorrect shape"
    # Performance threshold (adjust based on your hardware)
    assert gpu_time < 10.0, f"Matrix multiplication took too long: {gpu_time:.4f} seconds"

def test_cpu_gpu_memory_transfer():
    """Test CPU to GPU memory transfer"""
    size = 1000
    cpu_array = np.random.normal(size=(size, size))
    
    start_time = time.time()
    gpu_array = jnp.array(cpu_array)
    transfer_time = time.time() - start_time
    
    # Verify array was transferred correctly
    assert gpu_array.shape == cpu_array.shape, "Shape mismatch after transfer"
    assert jnp.allclose(gpu_array, cpu_array), "Values changed during transfer"
    # Performance threshold (adjust based on your hardware)
    assert transfer_time < 5.0, f"Memory transfer took too long: {transfer_time:.4f} seconds"

def test_complex_operations(random_key):
    """Test complex mathematical operations"""
    key = jax.random.split(random_key)[0]
    a = jax.random.normal(key, shape=(100, 100))
    
    start_time = time.time()
    result = jnp.sin(jnp.matmul(a, a.T)) + jnp.exp(jnp.diag(a))
    complex_time = time.time() - start_time
    
    # Verify result properties
    assert result.shape == (100, 100), "Result has incorrect shape"
    assert not jnp.any(jnp.isnan(result)), "Result contains NaN values"
    assert not jnp.any(jnp.isinf(result)), "Result contains infinite values"
    # Performance threshold (adjust based on your hardware)
    assert complex_time < 1.0, f"Complex operations took too long: {complex_time:.4f} seconds"
