import gymnasium as gym

# Register custom gymnasium environments

### Rooms Environment ###
gym.register(
    id="contgrid/Rooms-v0",
    entry_point="contgrid.envs.rooms:RoomsEnv",
)
