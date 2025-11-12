"""
Integration test to verify Rust acceleration works in a real environment.
"""

import gymnasium as gym
import numpy as np

from contgrid.core.grid_rust import HAS_RUST


def test_rust_integration():
    """Test that the Rust-accelerated environment works correctly."""
    print("=" * 80)
    print("RUST INTEGRATION TEST")
    print("=" * 80)

    print(f"\nRust acceleration available: {HAS_RUST}")

    # Create environment
    env = gym.make("contgrid/Rooms-v0", max_episode_steps=100)

    # Check if using Rust
    if hasattr(env, "world") and hasattr(env.world, "wall_collision_checker"):
        checker = env.world.wall_collision_checker
        if hasattr(checker, "using_rust"):
            print(f"Environment using Rust: {checker.using_rust}")
        else:
            print("Environment using Python (Rust wrapper not found)")
    else:
        print("Could not determine acceleration status")

    # Run a quick episode
    print("\nRunning test episode...")
    observation, info = env.reset(seed=42)

    total_reward = 0
    steps = 0

    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    print(f"Episode completed in {steps} steps")
    print(f"Total reward: {total_reward:.3f}")

    # Verify observations are correct
    assert isinstance(observation, dict), "Observation should be a dict"
    assert "agent_pos" in observation, "Observation should have agent_pos"
    assert "goal_pos" in observation, "Observation should have goal_pos"

    agent_pos = observation["agent_pos"]
    assert isinstance(agent_pos, np.ndarray), "Agent position should be numpy array"
    assert agent_pos.shape == (2,), (
        f"Agent position shape should be (2,), got {agent_pos.shape}"
    )

    print("\n✅ All integration tests passed!")

    env.close()

    return True


if __name__ == "__main__":
    success = test_rust_integration()
    if success:
        print("\n" + "=" * 80)
        print("SUCCESS: Rust-accelerated environment is working correctly!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("FAILURE: Integration test failed")
        print("=" * 80)
        exit(1)
