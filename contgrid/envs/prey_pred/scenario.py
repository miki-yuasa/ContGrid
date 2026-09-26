"""PreyPredScenario implementation managing entities, movement, and task dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from contgrid.core import (
    Agent,
    BaseScenario,
    Color,
    Landmark,
    World,
    WorldConfig,
)
from contgrid.core.typing import Position

from .configs import (
    DEFAULT_SCENARIO_CONFIG,
    DEFAULT_WORLD_CONFIG,
    OrderedTaskConfig,
    PreyPredScenarioConfig,
    StationaryTrajectoryConfig,
    TaskConfig,
    UnorderedTaskConfig,
)
from .observations import PreyPredObsFactory
from .trajectories import BaseTrajectory
from .utils import (
    PRED_COLORS,
    PREY_COLORS,
    TrajectoryRenderItem,
    build_region_entities,
    compute_wall_distances,
    enforce_bounds_and_clearance,
    sample_neutral_corridor,
)


class PreyPredScenario(BaseScenario[PreyPredScenarioConfig, dict[str, NDArray]]):
    """A scenario for the Prey-Predator multi-entity navigation world."""

    def __init__(
        self,
        config: PreyPredScenarioConfig = DEFAULT_SCENARIO_CONFIG,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
    ) -> None:
        super().__init__(config, world_config)
        self.obs_factory = PreyPredObsFactory()
        self.preys: list[Landmark] = []
        self.predators: list[Landmark] = []
        self.prey_trajectories: list[BaseTrajectory] = []
        self.pred_trajectories: list[BaseTrajectory] = []
        self.trajectory_items: list[TrajectoryRenderItem] = []
        self.sim_time: float = 0.0
        self.dt: float = world_config.dt
        self.wall_bounds: NDArray[np.float64] = np.zeros((0, 4), dtype=np.float64)
        self.prey_capture_counts: NDArray[np.int32] = np.zeros(4, dtype=np.int32)
        self.prey_was_inside: list[bool] = [False] * 4
        self.current_subtask_idx: int = 0
        self.is_success: bool = False
        self.predator_collisions: int = 0

    def init_agents(
        self, world: World, np_random: np.random.Generator | None = None
    ) -> list[Agent]:
        cfg = self.config.agent_spawn
        return [
            Agent(
                name="agent_0",
                size=cfg.size,
                color=Color.SKY_BLUE.name,
                u_range=cfg.u_range,
            )
        ]

    def init_landmarks(
        self, world: World, np_random: np.random.Generator | None = None
    ) -> list[Landmark]:
        self.preys.clear()
        self.predators.clear()
        self.prey_trajectories.clear()
        self.pred_trajectories.clear()
        self.trajectory_items.clear()

        self.wall_bounds = np.array(
            [
                [
                    w.state.pos[0] - w.size / 2.0, w.state.pos[0] + w.size / 2.0,
                    w.state.pos[1] - w.size / 2.0, w.state.pos[1] + w.size / 2.0,
                ]
                for w in world.walls
            ],
            dtype=np.float64,
        )

        landmarks: list[Landmark] = []
        for reg in self.config.regions:
            prey, p_traj, pred, q_traj = build_region_entities(reg)
            rid = reg.region_id
            self.preys.append(prey)
            self.prey_trajectories.append(p_traj)
            self.predators.append(pred)
            self.pred_trajectories.append(q_traj)
            landmarks.extend([prey, pred])
            self.trajectory_items.extend([
                TrajectoryRenderItem(trajectory=p_traj, color=PREY_COLORS[rid % 4]),
                TrajectoryRenderItem(trajectory=q_traj, color=PRED_COLORS[rid % 4]),
            ])

        return landmarks

    def reset_agents(
        self, world: World, np_random: np.random.Generator
    ) -> list[Agent]:
        agent = world.agents[0]
        cfg = self.config.agent_spawn
        pos: Position = (
            sample_neutral_corridor(world, agent.size, np_random)
            if cfg.mode == "neutral_corridor"
            else (cfg.fixed_pos if cfg.fixed_pos is not None else (7.0, 7.0))
        )

        agent.state.pos = np.array(pos, dtype=np.float64)
        agent.state.vel = np.zeros(2, dtype=np.float64)
        agent.terminated = False
        self.sim_time = 0.0
        self.prey_capture_counts.fill(0)
        self.current_subtask_idx = 0
        self.is_success = False
        self.predator_collisions = 0
        self.prey_was_inside = [
            bool(np.linalg.norm(agent.state.pos - p.state.pos) < agent.size + p.size)
            for p in self.preys
        ]
        return world.agents

    def _update_entities_at_time(self, t: float) -> None:
        """Evaluate and apply trajectory states and clearance at time t."""
        for reg, prey, p_traj, pred, q_traj in zip(
            self.config.regions,
            self.preys,
            self.prey_trajectories,
            self.predators,
            self.pred_trajectories,
        ):
            p_pos, p_vel = p_traj.get_state(t)
            q_pos, q_vel = q_traj.get_state(t)
            p_mov = not isinstance(reg.prey.trajectory, StationaryTrajectoryConfig)
            q_mov = not isinstance(reg.predator.trajectory, StationaryTrajectoryConfig)
            p_pos, q_pos = enforce_bounds_and_clearance(
                reg, p_pos, prey.size, p_mov, q_pos, pred.size, q_mov
            )
            prey.state.pos, prey.state.vel = p_pos, p_vel
            pred.state.pos, pred.state.vel = q_pos, q_vel

    def reset_landmarks(
        self, world: World, np_random: np.random.Generator
    ) -> list[Landmark]:
        self.sim_time = 0.0
        self._update_entities_at_time(0.0)
        return world.landmarks

    def step_entities(self, world: World) -> None:
        """Advance simulation time and update positions of all entities."""
        self.sim_time += self.dt
        self._update_entities_at_time(self.sim_time)

    def _update_capture_counts(self, agent: Agent) -> list[bool]:
        """Detect prey zone entries and increment edge-triggered capture counters."""
        now_inside: list[bool] = []
        for i, prey in enumerate(self.preys):
            in_zone = float(np.linalg.norm(agent.state.pos - prey.state.pos)) < (
                agent.size + prey.size
            )
            if in_zone and not self.prey_was_inside[i]:
                self.prey_capture_counts[i] += 1
            now_inside.append(in_zone)
        return now_inside

    def _check_predator_contacts(
        self, agent: Agent, task: TaskConfig
    ) -> float:
        """Check collision with predators and return penalty."""
        for j, pred in enumerate(self.predators):
            d = float(np.linalg.norm(agent.state.pos - pred.state.pos))
            if d >= (agent.size + pred.size):
                continue

            self.predator_collisions += 1
            if not isinstance(task, OrderedTaskConfig):
                if task.predator_absorbing:
                    agent.terminated = True
                return float(task.predator_penalty)

            if self.current_subtask_idx < len(task.subtask_seq):
                st = task.subtask_seq[self.current_subtask_idx]
                if j in st.avoid_predators:
                    if st.predator_absorbing:
                        agent.terminated = True
                    return float(st.penalty)
            return 0.0

        return 0.0

    def _process_unordered_capture(
        self,
        agent: Agent,
        task: UnorderedTaskConfig,
        now_inside: Sequence[bool],
    ) -> float:
        """Process set-based prey captures using explicit dictionary lookup."""
        step_reward = float(
            sum(
                task.capture_rewards[i]
                for i in task.required_preys
                if now_inside[i]
                and not self.prey_was_inside[i]
                and self.prey_capture_counts[i] == 1
            )
        )
        if all(self.prey_capture_counts[p] >= 1 for p in task.required_preys):
            self.is_success = True
            step_reward += task.completion_reward
            agent.terminated = True

        return step_reward

    def _process_ordered_subtask(
        self,
        agent: Agent,
        task: OrderedTaskConfig,
        now_inside: Sequence[bool],
    ) -> float:
        """Process sequential reach-avoid subtasks."""
        if self.current_subtask_idx >= len(task.subtask_seq):
            return 0.0

        st = task.subtask_seq[self.current_subtask_idx]
        tgt = st.target_prey
        if not (now_inside[tgt] and not self.prey_was_inside[tgt]):
            return 0.0

        self.current_subtask_idx += 1
        if st.prey_absorbing or self.current_subtask_idx >= len(task.subtask_seq):
            agent.terminated = True
            if self.current_subtask_idx >= len(task.subtask_seq):
                self.is_success = True
                return float(st.reward + task.completion_reward)
        return float(st.reward)

    def reward(self, agent: Agent, world: World) -> float:
        task = self.config.task
        now_inside = self._update_capture_counts(agent)
        reward = -task.step_penalty + self._check_predator_contacts(agent, task)
        if not agent.terminated:
            if isinstance(task, UnorderedTaskConfig):
                reward += self._process_unordered_capture(agent, task, now_inside)
            elif isinstance(task, OrderedTaskConfig):
                reward += self._process_ordered_subtask(agent, task, now_inside)

        self.prey_was_inside = now_inside
        return float(reward)

    def observation(self, agent: Agent, world: World) -> dict[str, NDArray]:
        wall_d = compute_wall_distances(agent.state.pos, self.wall_bounds)
        return self.obs_factory.observation(
            agent=agent,
            wall_dist=wall_d,
            prey_positions=np.array([p.state.pos for p in self.preys], dtype=np.float64),
            prey_velocities=np.array([p.state.vel for p in self.preys], dtype=np.float64),
            predator_positions=np.array([q.state.pos for q in self.predators], dtype=np.float64),
            predator_velocities=np.array([q.state.vel for q in self.predators], dtype=np.float64),
            prey_capture_counts=self.prey_capture_counts,
        )

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        wl = world.grid.wall_limits
        return spaces.Dict(
            self.obs_factory.obs_space_dict(
                num_preys=len(self.preys),
                num_predators=len(self.predators),
                low_bound=np.array([wl.min_x, wl.min_y], dtype=np.float64),
                high_bound=np.array([wl.max_x, wl.max_y], dtype=np.float64),
            )
        )

    def info(self, agent: Agent, world: World) -> dict[str, Any]:
        return {
            "is_success": self.is_success,
            "prey_capture_counts": self.prey_capture_counts.copy(),
            "current_subtask_idx": self.current_subtask_idx,
            "sim_time": self.sim_time,
            "predator_collisions": self.predator_collisions,
        }
