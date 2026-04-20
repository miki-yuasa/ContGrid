import numpy as np

from contgrid.core import Grid, WorldConfig
from contgrid.envs.zone import (
    ObjConfig,
    SpawnConfig,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
    ZoneEnv,
    ZoneScenarioConfig,
)


class TestUniformRandomSpawnStrategy:
    def test_uniform_random_spawn_strategy_respects_constraints(self):
        """Uniform random spawning keeps spacing, wall validity, and agent clearance."""
        min_spacing = 0.8
        num_yellow_zones = 4
        zone_size = 0.2
        agent_size = 0.1

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=None) for _ in range(num_yellow_zones)],
            red_zone=[ObjConfig(pos=None)],
            white_zone=[ObjConfig(pos=None)],
            black_zone=[ObjConfig(pos=None)],
            zone_size=zone_size,
            agent_size=agent_size,
            spawn_method=UniformRandomConfig(min_spacing=min_spacing),
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(
            grid=Grid(
                layout=[
                    "#########",
                    "#       #",
                    "#       #",
                    "#       #",
                    "#       #",
                    "#       #",
                    "#       #",
                    "#       #",
                    "#########",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            for seed in (7, 17, 37):
                env.reset(seed=seed)

                strategy = env.scenario.spawn_strategy
                assert isinstance(strategy, UniformRandomSpawnStrategy)

                agent_pos = env.world.agents[0].state.pos.copy()
                positions = strategy.spawn_obstacles(
                    num_obstacles=num_yellow_zones,
                    obstacle_type="yellow",
                    world=env.world,
                    scenario=env.scenario,
                    np_random=np.random.default_rng(seed + 100),
                    agent_pos=agent_pos,
                )

                assert len(positions) == num_yellow_zones

                min_agent_distance = zone_size + agent_size
                for idx, pos in enumerate(positions):
                    candidate = (float(pos[0]), float(pos[1]))
                    assert env.world.wall_collision_checker.is_position_valid(
                        zone_size,
                        env.world.contact_margin,
                        candidate,
                    ), f"Zone {idx} spawned in invalid location: {candidate}"

                    dist_to_agent = np.linalg.norm(np.array(pos, dtype=np.float64) - agent_pos)
                    assert dist_to_agent >= (min_agent_distance - 1e-9), (
                        f"Zone {idx} too close to agent: {dist_to_agent:.4f} < "
                        f"{min_agent_distance:.4f}"
                    )

                positions_array = np.array(positions, dtype=np.float64)
                for i in range(len(positions_array)):
                    for j in range(i + 1, len(positions_array)):
                        dist = np.linalg.norm(positions_array[i] - positions_array[j])
                        assert dist >= (min_spacing - 1e-9), (
                            f"Zones {i} and {j} too close: {dist:.4f} < {min_spacing:.4f}"
                        )
        finally:
            env.close()
