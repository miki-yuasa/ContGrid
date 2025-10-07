import os
from pprint import pprint

import gymnasium as gym
import imageio
import numpy as np

import contgrid

num_steps = 1

env = gym.make("contgrid/Rooms-v0", max_episode_steps=num_steps)
observation, info = env.reset(seed=42)
print(f"Environment observation space: {env.observation_space}")
print(f"Environment action space: {env.action_space}")
print(f"Initial observation: {observation}")
print(f"Initial info: {info}")

frames = []
rewards = []
positions = []

# Collect frames for animation
for step in range(num_steps):
    # Render current state
    print(f"Step {step + 1}/{num_steps}")
    rendered = env.render()
    assert rendered is not None
    frames.append(rendered.astype(np.uint8))  # Ensure uint8 for imageio

    # Store position and reward for later analysis
    positions.append(observation["agent_pos"].copy())

    # Take random action
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(
        f"- Action taken: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}"
    )
    print("- Info")
    pprint(info)
    print("- New observation: ")
    pprint(observation)
    rewards.append(reward)

    if terminated or truncated:
        break

# Create output directory
output_dir = os.path.join("tests", "out", "animations")
os.makedirs(output_dir, exist_ok=True)

# Save individual frames as images for verification
frame_dir = os.path.join(output_dir, "frames")
os.makedirs(frame_dir, exist_ok=True)

for i, frame in enumerate(frames):
    # Save individual frame using imageio
    frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
    imageio.imwrite(frame_path, frame)

# Create and save GIF animation using imageio
if len(frames) > 1:
    gif_path = os.path.join(output_dir, "random_actions_animation.gif")
    imageio.mimsave(
        gif_path,
        frames,
        duration=0.2,  # 0.2 seconds per frame (5 FPS)
        loop=0,  # Loop forever
    )

    # Verify GIF was created
    assert os.path.exists(gif_path)
    assert os.path.getsize(gif_path) > 0
    print(f"Animation saved to: {gif_path}")

    # Also save as MP4 for better quality and smaller file size
    mp4_path = os.path.join(output_dir, "random_actions_animation.mp4")
    try:
        imageio.mimsave(
            mp4_path,
            frames,
            fps=5,  # 5 frames per second
            quality=8,  # Good quality
        )
        print(f"MP4 animation saved to: {mp4_path}")
    except Exception as e:
        print(f"Could not save MP4 (missing codec?): {e}")

# Verify we have collected data
assert len(frames) > 0
assert len(rewards) > 0
assert len(positions) > 0
assert len(frames) == len(rewards)
assert len(frames) == len(positions)

# Verify frame properties
for frame in frames:
    assert isinstance(frame, np.ndarray)
    assert len(frame.shape) == 3
    assert frame.shape[2] == 3  # RGB
    assert frame.dtype == np.uint8  # Correct dtype for imageio

# Create summary statistics
total_reward = sum(rewards)
avg_reward = total_reward / len(rewards)

print(f"Animation created with {len(frames)} frames")
print(f"Total reward: {total_reward:.3f}")
print(f"Average reward per step: {avg_reward:.3f}")
print(f"Final position: {positions[-1]}")
print(f"Position change: {np.linalg.norm(positions[-1] - positions[0]):.3f}")

env.close()
