"""Integration Test
[Warning]Each environment needs several miniutes without good GPU.
"""

import pytest
import subprocess
from typing import List
import re

@pytest.fixture
def base_command() -> List[str]:
    """Fixture providing base command with common arguments."""
    return [
        "python",
        "run_meta_popgym.py",
        "--arch", "s5",
        "--debug", "1",  # Use small S5 model
        "--depth", "1",
        "--dim", "64",  # meta_dim is output dimension, works for all environments
        "--num_runs", "1",
        "--seed", "42"
    ]

@pytest.mark.parametrize("env_name,dim,eval_method", [
    ("cartpole", 8, None),
    #("pendulum", 64, None),
    #("higherlower", 64, None),
    #("autoencode", 64, None),
    #("battleship", 64, None),
    #("count_recall", 64, None),
    #("repeat_first", 64, None),
    #("repeat_previous", 64, None),
    #("minesweeper", 64, None),
    #("multiarmedbandit", 64, None),
    #("concentration", 256, "tiling")  # Use larger dimension and tiling for concentration
])
def test_meta_environment_execution(base_command: List[str], env_name: str, dim: int, eval_method: str):
    """Test execution of meta environments through CLI interface."""
    cmd = base_command.copy()
    # Update dim in base command if different from default
    if dim != 64:
        dim_index = cmd.index("--dim") + 1
        cmd[dim_index] = str(dim)
    cmd.extend(["--env", env_name])
    if eval_method:
        cmd.extend(["--eval_method", eval_method])
    run_test_with_command(cmd, env_name)

def run_test_with_command(cmd: List[str], env_name: str):
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        # Check return code
        assert result.returncode == 0, f"Process failed for {env_name} with exit code {result.returncode}\nError: {result.stderr}"
        
        # Check for common error patterns in stderr
        assert "error" not in result.stderr.lower(), f"Error detected in stderr for {env_name}: {result.stderr}"
        assert "exception" not in result.stderr.lower(), f"Exception detected in stderr for {env_name}: {result.stderr}"
        
        # Verify expected output patterns
        assert "**********" in result.stdout, \
            f"Missing expected asterisk separator in output for {env_name}"
            
        # Check for successful completion (script ran without errors)
        assert "s5 time:" in result.stdout, \
            f"Missing completion timing message in output for {env_name}"
        
    except subprocess.TimeoutExpired:
        pytest.fail(f"Test timed out after 300 seconds for environment: {env_name}")
    except subprocess.SubprocessError as e:
        pytest.fail(f"Subprocess error occurred for {env_name}: {str(e)}")
    except Exception as e:
        pytest.fail(f"Unexpected error occurred for {env_name}: {str(e)}")

def test_invalid_environment(base_command: List[str]):
    """Test behavior with invalid environment name."""
    cmd = base_command + ["--env", "nonexistent_env"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Should fail with non-zero exit code
    assert result.returncode != 0, "Process should fail with invalid environment"
    
    # Error message should mention invalid environment
    assert "error" in result.stderr.lower() or "invalid" in result.stderr.lower(), \
        "Error message should indicate invalid environment"
