from pathlib import Path
from typing import cast

import imageio
import numpy as np

from contgrid.core import Grid, WorldConfig
from contgrid.envs.zone import (
    ObjConfig,
    SpawnConfig,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
    ZoneEnv,
    ZoneScenario,
    ZoneScenarioConfig,
)


class TestUniformRandomSpawnStrategy:
    def test_uniform_random_spawn_strategy_respects_constraints(self):
        """Uniform random spawning keeps spacing, wall validity, and agent clearance."""
        min_spacing = 1.2
        num_zones = 2
        zone_size = 0.5
        agent_size = 0.1
        output_dir = Path("tests") / "out" / "zone_uniform_random_spawn"
        output_dir.mkdir(parents=True, exist_ok=True)

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            red_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            white_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            black_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            zone_size=zone_size,
            agent_size=agent_size,
            spawn_method=UniformRandomConfig(min_spacing=min_spacing),
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(
            grid=Grid(
                layout=[
                    "############",
                    "#          #",
                    "#          #",
                    "#          #",
                    "#          #",
                    "#          #",
                    "#          #",
                    "#          #",
                    "#          #",
                    "#          #",
                    "############",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            for seed in (7, 17, 37):
                env.reset(seed=seed)
                rendered = env.render()
                assert rendered is not None
                image_path = output_dir / f"uniform_random_zone_map_seed_{seed}.png"
                imageio.imwrite(image_path, rendered.astype(np.uint8))
                assert image_path.exists()

                scenario = cast(ZoneScenario, env.scenario)

                strategy = scenario.spawn_strategy
                assert isinstance(strategy, UniformRandomSpawnStrategy)

                zone_positions_by_color = {
                    "yellow": scenario.yellow_pos,
                    "red": scenario.red_pos,
                    "white": scenario.white_pos,
                    "black": scenario.black_pos,
                }

                all_zone_positions: list[tuple[str, int, np.ndarray]] = []
                for color, color_positions in zone_positions_by_color.items():
                    assert len(color_positions) == num_zones
                    for idx, pos in enumerate(color_positions):
                        all_zone_positions.append((color, idx, pos))

                agent_pos = env.world.agents[0].state.pos.copy()
                min_agent_distance = zone_size + agent_size
                for color, idx, pos in all_zone_positions:
                    candidate = (float(pos[0]), float(pos[1]))
                    assert env.world.wall_collision_checker.is_position_valid(
                        zone_size,
                        env.world.contact_margin,
                        candidate,
                    ), f"{color} zone {idx} spawned in invalid location: {candidate}"

                    dist_to_agent = np.linalg.norm(pos - agent_pos)
                    assert dist_to_agent >= (min_agent_distance - 1e-9), (
                        f"{color} zone {idx} too close to agent: {dist_to_agent:.4f} < "
                        f"{min_agent_distance:.4f}"
                    )

                for i in range(len(all_zone_positions)):
                    color_i, idx_i, pos_i = all_zone_positions[i]
                    for j in range(i + 1, len(all_zone_positions)):
                        color_j, idx_j, pos_j = all_zone_positions[j]
                        dist = np.linalg.norm(pos_i - pos_j)
                        assert dist >= (min_spacing - 1e-9), (
                            f"{color_i}[{idx_i}] and {color_j}[{idx_j}] too close: "
                            f"{dist:.4f} < {min_spacing:.4f}"
                        )
        finally:
            env.close()
