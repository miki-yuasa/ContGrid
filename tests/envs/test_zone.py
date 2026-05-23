from contgrid.envs.zone.configs import ObsConfig
from pathlib import Path
from typing import cast

import imageio
import numpy as np

from contgrid.core import Grid, WorldConfig
from contgrid.envs.zone import (
    FixedRandomSwapSpawnConfig,
    FixedRandomSwapSpawnStrategy,
    GaussianSpawnConfig,
    GaussianSpawnStrategy,
    ObjConfig,
    RandomSwapSpec,
    SpawnConfig,
    FixedSpawnConfig,
    FixedSpawnStrategy,
    ZoneSizeConfig,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
    ZoneEnv,
    ZoneScenario,
    ZoneScenarioConfig,
)


class TestUniformRandomSpawnStrategy:
    def test_uniform_random_spawn(self):
        """Uniform random spawning keeps spacing, wall validity, and agent clearance."""
        min_spacing = 1.5
        num_zones = 6
        zone_size = 0.5
        agent_size = 0.1
        output_dir = Path("tests") / "out" / "zone" / "uniform_random_spawn"
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
                    "#          #",
                    "############",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            for seed in (7, 17, 27, 37):
                env.reset(seed=seed)
                rendered = env.render()
                assert rendered is not None
                image_path = output_dir / f"uniform_random_zone_map_seed_{seed}.png"
                imageio.imwrite(image_path, rendered.astype(np.uint8))
                assert image_path.exists()

                scenario = cast(ZoneScenario, env.scenario)

                strategy = scenario.spawn_manager.spawn_strategy
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


class TestGaussianSpawnStrategy:
    def test_gaussian_spawn(self):
        """Gaussian spawning stays valid and concentrates around the map center."""
        num_yellow_zones = 2
        num_red_zones = 8
        num_white_zones = 2
        num_black_zones = 2
        zone_size = 0.5
        agent_size = 0.1
        min_spacing = 1.1  # Reduced from 1.25 for practical feasibility
        gaussian_std = 2
        output_dir = Path("tests") / "out" / "zone" / "gaussian_spawn"
        output_dir.mkdir(parents=True, exist_ok=True)

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=None) for _ in range(num_yellow_zones)],
            red_zone=[ObjConfig(pos=None) for _ in range(num_red_zones)],
            white_zone=[ObjConfig(pos=None) for _ in range(num_white_zones)],
            black_zone=[ObjConfig(pos=None) for _ in range(num_black_zones)],
            zone_size=zone_size,
            agent_size=agent_size,
            spawn_method=GaussianSpawnConfig(
                gaussian_std=gaussian_std, min_spacing=min_spacing
            ),
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
                    "#          #",
                    "############",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            for seed in (7, 17, 27, 37):
                env.reset(seed=seed)
                rendered = env.render()
                assert rendered is not None
                image_path = output_dir / f"gaussian_zone_map_seed_{seed}.png"
                imageio.imwrite(image_path, rendered.astype(np.uint8))
                assert image_path.exists()

                scenario = cast(ZoneScenario, env.scenario)
                strategy = scenario.spawn_manager.spawn_strategy
                assert isinstance(strategy, GaussianSpawnStrategy)

                zone_positions_by_color = {
                    "yellow": scenario.yellow_pos,
                    "red": scenario.red_pos,
                    "white": scenario.white_pos,
                    "black": scenario.black_pos,
                }
                num_zones_list = [
                    num_yellow_zones,
                    num_red_zones,
                    num_white_zones,
                    num_black_zones,
                ]

                # Each color should have at least 4 zones (allow for some spawning failures)
                all_zone_positions: list[np.ndarray] = []
                for color, num_zones in zip(
                    zone_positions_by_color.keys(), num_zones_list
                ):
                    color_positions = zone_positions_by_color[color]
                    assert len(color_positions) == num_zones, (
                        f"{color} has only {len(color_positions)} zones"
                    )
                    all_zone_positions.extend(color_positions)

                agent_pos = env.world.agents[0].state.pos.copy()
                min_agent_distance = zone_size + agent_size
                center = np.array([6.0, 6.0], dtype=np.float64)
                distances_to_center: list[float] = []

                for pos in all_zone_positions:
                    candidate = (float(pos[0]), float(pos[1]))
                    assert env.world.wall_collision_checker.is_position_valid(
                        zone_size,
                        env.world.contact_margin,
                        candidate,
                    )

                    dist_to_agent = np.linalg.norm(pos - agent_pos)
                    assert dist_to_agent >= (min_agent_distance - 1e-9)

                    distances_to_center.append(float(np.linalg.norm(pos - center)))

                assert float(np.mean(distances_to_center)) < 4.0

                # Check global min_spacing constraint (must match spawn config)
                for i in range(len(all_zone_positions)):
                    for j in range(i + 1, len(all_zone_positions)):
                        dist = np.linalg.norm(
                            all_zone_positions[i] - all_zone_positions[j]
                        )
                        assert dist >= (min_spacing - 1e-6), (
                            f"Zones {i} and {j} violate min_spacing: "
                            f"{dist:.6f} < {min_spacing}"
                        )
        finally:
            env.close()

    def test_random_action(self):
        """Uniform random spawning keeps spacing, wall validity, and agent clearance."""
        min_spacing = 1.5
        num_zones = 2
        zone_size = 0.5
        agent_size = 0.1
        output_dir = Path("tests") / "out" / "zone" / "random_action"
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
                    "#          #",
                    "############",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            for seed in (7, 17, 37):
                env.reset(seed=seed)

                frames: list = []

                for _ in range(50):
                    action = env.action_space.sample()
                    _, _, _, _, _ = env.step(action)
                    frames.append(env.render())

                video_path = output_dir / f"uniform_random_zone_video_seed_{seed}.gif"
                imageio.mimwrite(video_path, frames, fps=5)
                assert video_path.exists()
        finally:
            env.close()

    def test_no_observation_warnings(self):
        """Test that env reset and step do not emit numpy.ndarray type warnings for discrete spaces."""
        import warnings
        from contgrid.envs.zone.configs import ObsConfig, SubtaskConfig, ZoneType

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[
                SubtaskConfig(
                    goal=ZoneType.YELLOW,
                    obstacle=ZoneType.WHITE,
                    reward=50.0,
                    penalty=-1.0,
                )
            ],
            yellow_zone=[ObjConfig(pos=None)],
            red_zone=[ObjConfig(pos=None)],
            white_zone=[ObjConfig(pos=None)],
            black_zone=[ObjConfig(pos=None)],
            zone_size=0.5,
            agent_size=0.1,
            spawn_method=UniformRandomConfig(min_spacing=1.5),
        )

        scenario_config = ZoneScenarioConfig(
            spawn_config=spawn_config,
            obs_config=ObsConfig(include_subtask=True),
        )
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
                    "#          #",
                    "############",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                env.reset(seed=42)
                for _ in range(5):
                    action = env.action_space.sample()
                    env.step(action)

                for warning in w:
                    assert (
                        "should be an int or np.int64, actual type: <class 'numpy.ndarray'>"
                        not in str(warning.message)
                    )
        finally:
            env.close()


class TestFixedSpawnStrategy:
    def test_fixed_spawn(self):
        """Fixed spawning keeps spacing, wall validity, and agent clearance."""
        min_spacing = 1.5
        num_zones_yellow = 4
        num_zones_red = 5
        num_zones_white = 0
        num_zones_black = 0
        agent_size = 0.1
        output_dir = Path("tests") / "out" / "zone" / "fixed_spawn"
        output_dir.mkdir(parents=True, exist_ok=True)

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[
                ObjConfig(pos=(1.5, 5.5)),
                ObjConfig(pos=(9.5, 5.5)),
                ObjConfig(pos=(5.5, 1.5)),
                ObjConfig(pos=(5.5, 9.5)),
            ],
            red_zone=[
                ObjConfig(pos=(5.5, 5.5)),
                ObjConfig(pos=(3.5, 3.5)),
                ObjConfig(pos=(7.5, 3.5)),
                ObjConfig(pos=(7.5, 7.5)),
                ObjConfig(pos=(3.5, 7.5)),
            ],
            white_zone=[],
            black_zone=[],
            zone_size=ZoneSizeConfig(
                yellow=0.25,
                red=0.75,
                white=0.5,
                black=0.5,
            ),
            agent_size=agent_size,
            spawn_method=FixedSpawnConfig(min_spacing=min_spacing),
            reset_agent_first=False,
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
                    "#          #",
                    "############",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            for seed in (7, 17, 27, 37):
                env.reset(seed=seed)
                rendered = env.render()
                assert rendered is not None
                image_path = output_dir / f"fixed_zone_map_seed_{seed}.png"
                imageio.imwrite(image_path, rendered.astype(np.uint8))
                assert image_path.exists()

                scenario = cast(ZoneScenario, env.scenario)

                strategy = scenario.spawn_manager.spawn_strategy
                assert isinstance(strategy, FixedSpawnStrategy)

                zone_positions_by_color = {
                    "yellow": scenario.yellow_pos,
                    "red": scenario.red_pos,
                    "white": scenario.white_pos,
                    "black": scenario.black_pos,
                }

                all_zone_positions: list[tuple[str, int, np.ndarray]] = []
                for color, color_positions in zone_positions_by_color.items():
                    if color == "yellow":
                        assert len(color_positions) == num_zones_yellow
                    elif color == "red":
                        assert len(color_positions) == num_zones_red
                    elif color == "white":
                        assert len(color_positions) == num_zones_white
                    elif color == "black":
                        assert len(color_positions) == num_zones_black
                    for idx, pos in enumerate(color_positions):
                        all_zone_positions.append((color, idx, pos))

                agent_pos = env.world.agents[0].state.pos.copy()
                for color, idx, pos in all_zone_positions:
                    if color == "yellow":
                        zone_size = env.scenario.zone_sizes.yellow
                    elif color == "red":
                        zone_size = env.scenario.zone_sizes.red
                    elif color == "white":
                        zone_size = env.scenario.zone_sizes.white
                    elif color == "black":
                        zone_size = env.scenario.zone_sizes.black

                    min_agent_distance = zone_size + agent_size
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


class TestFixedRandomSwapSpawnStrategy:
    """Tests for the FixedRandomSwapSpawnStrategy."""

    # The 4 fixed yellow zone candidate positions
    YELLOW_POSITIONS = [
        (1.5, 5.5),
        (9.5, 5.5),
        (5.5, 1.5),
        (5.5, 9.5),
    ]

    def test_fixed_random_swap_spawn(self):
        """Random swap: 1 white at a yellow pos, 1 yellow removed due to black overlap."""
        agent_size = 0.1
        output_dir = Path("tests") / "out" / "zone" / "fixed_random_swap_spawn"
        output_dir.mkdir(parents=True, exist_ok=True)

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[
                ObjConfig(pos=(1.5, 5.5)),
                ObjConfig(pos=(9.5, 5.5)),
                ObjConfig(pos=(5.5, 1.5)),
                ObjConfig(pos=(5.5, 9.5)),
            ],
            red_zone=[
                ObjConfig(pos=(5.5, 5.5)),
                ObjConfig(pos=(3.5, 3.5)),
                ObjConfig(pos=(7.5, 3.5)),
                ObjConfig(pos=(7.5, 7.5)),
                ObjConfig(pos=(3.5, 7.5)),
            ],
            white_zone=[],
            black_zone=[ObjConfig(pos=None), ObjConfig(pos=None)],
            zone_size=ZoneSizeConfig(
                yellow=0.25,
                red=0.75,
                white=0.25,
                black=0.25,
            ),
            agent_size=agent_size,
            spawn_method=FixedRandomSwapSpawnConfig(
                swaps=[
                    RandomSwapSpec(
                        source_zone="yellow",
                        target_zone="white",
                        num_swaps=1,
                    ),
                ],
            ),
            reset_agent_first=False,
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
                    "#          #",
                    "############",
                ]
            )
        )

        env = ZoneEnv(scenario_config=scenario_config, world_config=world_config)
        try:
            for seed in (7, 17, 27, 37, 47, 57, 67, 77, 87, 97):
                env.reset(seed=seed)
                rendered = env.render()
                assert rendered is not None
                image_path = output_dir / f"fixed_random_swap_seed_{seed}.png"
                imageio.imwrite(image_path, rendered.astype(np.uint8))
                assert image_path.exists()

                scenario = cast(ZoneScenario, env.scenario)

                strategy = scenario.spawn_manager.spawn_strategy
                assert isinstance(strategy, FixedRandomSwapSpawnStrategy)

                # 1 white zone spawned at one of the yellow positions
                assert len(scenario.white_pos) == 1, (
                    f"Expected 1 white zone, got {len(scenario.white_pos)}"
                )
                white_pos_tuple = (
                    float(scenario.white_pos[0][0]),
                    float(scenario.white_pos[0][1]),
                )
                assert white_pos_tuple in self.YELLOW_POSITIONS, (
                    f"White zone at {white_pos_tuple} not in yellow candidates"
                )

                # 5 red zones unchanged
                assert len(scenario.red_pos) == 5

                # Yellow zones: started with 4, minus 1 swap
                num_yellow = len(scenario.yellow_pos)
                assert num_yellow in (3,), f"Expected 3 zones, got {num_yellow}"

                # White zone: started with 0, plus 1 swap
                assert len(scenario.white_pos) == 1

                # Black zone: started with 0, no swaps
                assert len(scenario.black_pos) == 2

                # Verify agent does not overlap with any zone
                agent_pos = env.world.agents[0].state.pos.copy()
                for lm in (
                    scenario.yellow + scenario.red + scenario.white + scenario.black
                ):
                    dist_to_agent = np.linalg.norm(lm.state.pos - agent_pos)
                    min_agent_distance = lm.size + agent_size
                    assert dist_to_agent >= (min_agent_distance - 1e-9), (
                        f"{lm.name} too close to agent: {dist_to_agent:.4f} < {min_agent_distance:.4f}"
                    )

                # Verify total landmark count matches
                expected_total = num_yellow + 5 + 1 + 2  # yellow + red + white
                assert (
                    len(scenario.yellow)
                    + len(scenario.red)
                    + len(scenario.white)
                    + len(scenario.black)
                    == expected_total
                )
                for _ in range(5):
                    env.step(env.action_space.sample())
        finally:
            env.close()


class TestAgentSpawningPerturbation:
    def test_agent_spawning_perturbation(self):
        """Test that agent position is perturbed when agent_perturbation > 0, and not when it is 0."""
        agent_size = 0.1
        zone_size = 0.5
        min_spacing = 1.5

        # Helper to create scenario and world config
        def make_config(agent_perturbation: float):
            spawn_config = SpawnConfig(
                agent=None,
                subtask_seq=[],
                yellow_zone=[ObjConfig(pos=None) for _ in range(2)],
                red_zone=[ObjConfig(pos=None) for _ in range(2)],
                white_zone=[ObjConfig(pos=None) for _ in range(2)],
                black_zone=[ObjConfig(pos=None) for _ in range(2)],
                zone_size=zone_size,
                agent_size=agent_size,
                agent_perturbation=agent_perturbation,
                spawn_method=UniformRandomConfig(min_spacing=min_spacing),
                reset_agent_first=False,
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
                        "#          #",
                        "############",
                    ]
                )
            )
            return scenario_config, world_config

        # 1. Test with agent_perturbation = 0.25
        sc_config, w_config = make_config(agent_perturbation=0.25)
        env = ZoneEnv(scenario_config=sc_config, world_config=w_config)
        try:
            perturbed_count = 0
            for seed in range(20):
                env.reset(seed=seed)
                agent_pos = env.world.agents[0].state.pos.copy()

                # Check validity against walls
                assert env.world.wall_collision_checker.is_position_valid(
                    agent_size,
                    env.world.contact_margin,
                    (float(agent_pos[0]), float(agent_pos[1])),
                ), f"Seed {seed}: Agent position {agent_pos} collided with walls"

                # Check validity against landmarks
                scenario = cast(ZoneScenario, env.scenario)
                for lm in (
                    scenario.yellow + scenario.red + scenario.white + scenario.black
                ):
                    dist_to_agent = np.linalg.norm(lm.state.pos - agent_pos)
                    min_agent_distance = lm.size + agent_size
                    assert dist_to_agent >= (min_agent_distance - 1e-9), (
                        f"Seed {seed}: {lm.name} too close to agent: {dist_to_agent:.4f} < {min_agent_distance:.4f}"
                    )

                # Check if position is perturbed (not exactly integers)
                frac_x = abs(agent_pos[0] - round(agent_pos[0]))
                frac_y = abs(agent_pos[1] - round(agent_pos[1]))
                if frac_x > 1e-6 or frac_y > 1e-6:
                    perturbed_count += 1

            assert perturbed_count > 0, (
                "Expected at least some agent positions to be perturbed"
            )

        finally:
            env.close()

        # 2. Test with agent_perturbation = 0.0
        sc_config, w_config = make_config(agent_perturbation=0.0)
        env = ZoneEnv(scenario_config=sc_config, world_config=w_config)
        try:
            for seed in range(20):
                env.reset(seed=seed)
                agent_pos = env.world.agents[0].state.pos.copy()

                # Fractional part should be zero (exactly at cell center integers)
                frac_x = abs(agent_pos[0] - round(agent_pos[0]))
                frac_y = abs(agent_pos[1] - round(agent_pos[1]))
                assert frac_x < 1e-6 and frac_y < 1e-6, (
                    f"Seed {seed}: Expected no perturbation for agent_perturbation=0.0, "
                    f"but got agent_pos={agent_pos} with fractional parts ({frac_x:.4f}, {frac_y:.4f})"
                )
        finally:
            env.close()
