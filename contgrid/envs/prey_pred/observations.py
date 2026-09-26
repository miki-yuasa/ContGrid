"""Observation factory classes for the PreyPred environment."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from contgrid.core import Agent, BaseObsFactory


class PreyPredObs(TypedDict):
    """Observation dictionary conforming to PreyPred observation space."""

    agent_pos: NDArray[np.float64]
    agent_vel: NDArray[np.float64]
    wall_dist: NDArray[np.float64]
    prey_rel_pos: NDArray[np.float64]
    prey_rel_vel: NDArray[np.float64]
    predator_rel_pos: NDArray[np.float64]
    predator_rel_vel: NDArray[np.float64]
    prey_capture_counts: NDArray[np.int32]


class PreyPredInfo(TypedDict):
    """Diagnostic info dictionary returned by PreyPredEnv."""

    is_success: bool
    prey_capture_counts: NDArray[np.int32]
    current_subtask_idx: int
    sim_time: float
    predator_collisions: int


class PreyPredObsFactory(BaseObsFactory):
    """Factory for pure state-based spatial-kinematic observations in PreyPredEnv.

    Constructs observation spaces and feature vectors including agent kinematics,
    wall distances, relative positions and velocities of all preys and predators,
    and edge-triggered prey capture counts.
    """

    def __init__(self, name: str = "prey_pred_obs") -> None:
        self.name = name

    def obs_space_dict(
        self,
        num_preys: int,
        num_predators: int,
        low_bound: NDArray[np.float64],
        high_bound: NDArray[np.float64],
        max_speed: float = 15.0,
    ) -> dict[str, spaces.Space]:
        """Construct the dictionary of observation spaces for the environment.

        Args:
            num_preys: Number of prey entities.
            num_predators: Number of predator entities.
            low_bound: 1D array of (min_x, min_y) world limits.
            high_bound: 1D array of (max_x, max_y) world limits.
            max_speed: Upper bound for velocity vectors in m/s.

        Returns:
            Dictionary mapping feature names to Gymnasium Box spaces.
        """
        max_dist = float(np.linalg.norm(high_bound - low_bound))
        rel_low_2d = low_bound - high_bound
        rel_high_2d = high_bound - low_bound

        prey_rel_low = np.tile(rel_low_2d, (num_preys, 1))
        prey_rel_high = np.tile(rel_high_2d, (num_preys, 1))

        pred_rel_low = np.tile(rel_low_2d, (num_predators, 1))
        pred_rel_high = np.tile(rel_high_2d, (num_predators, 1))

        obs_dict: dict[str, spaces.Space] = {
            "agent_pos": spaces.Box(
                low=low_bound,
                high=high_bound,
                shape=(2,),
                dtype=np.float64,
            ),
            "agent_vel": spaces.Box(
                low=-max_speed,
                high=max_speed,
                shape=(2,),
                dtype=np.float64,
            ),
            "wall_dist": spaces.Box(
                low=0.0,
                high=max_dist,
                shape=(4,),
                dtype=np.float64,
            ),
            "prey_rel_pos": spaces.Box(
                low=prey_rel_low,
                high=prey_rel_high,
                shape=(num_preys, 2),
                dtype=np.float64,
            ),
            "prey_rel_vel": spaces.Box(
                low=-max_speed,
                high=max_speed,
                shape=(num_preys, 2),
                dtype=np.float64,
            ),
            "predator_rel_pos": spaces.Box(
                low=pred_rel_low,
                high=pred_rel_high,
                shape=(num_predators, 2),
                dtype=np.float64,
            ),
            "predator_rel_vel": spaces.Box(
                low=-max_speed,
                high=max_speed,
                shape=(num_predators, 2),
                dtype=np.float64,
            ),
            "prey_capture_counts": spaces.Box(
                low=0,
                high=1000,
                shape=(num_preys,),
                dtype=np.int32,
            ),
        }
        return obs_dict

    def observation(
        self,
        agent: Agent,
        wall_dist: NDArray[np.float64],
        prey_positions: NDArray[np.float64],
        prey_velocities: NDArray[np.float64],
        predator_positions: NDArray[np.float64],
        predator_velocities: NDArray[np.float64],
        prey_capture_counts: NDArray[np.int32],
    ) -> dict[str, NDArray]:
        """Compute the observation dictionary for the current simulation step.

        Args:
            agent: The controllable agent.
            wall_dist: 1D array of 4 cardinal distances [top, right, bottom, left].
            prey_positions: (num_preys, 2) array of prey positions.
            prey_velocities: (num_preys, 2) array of prey velocities.
            predator_positions: (num_predators, 2) array of predator positions.
            predator_velocities: (num_predators, 2) array of predator velocities.
            prey_capture_counts: (num_preys,) array of capture counts.

        Returns:
            Observation dictionary conforming to `obs_space_dict`.
        """
        agent_pos = agent.state.pos.copy()
        agent_vel = agent.state.vel.copy()

        prey_rel_pos = prey_positions - agent_pos
        prey_rel_vel = prey_velocities - agent_vel

        predator_rel_pos = predator_positions - agent_pos
        predator_rel_vel = predator_velocities - agent_vel

        return {
            "agent_pos": agent_pos,
            "agent_vel": agent_vel,
            "wall_dist": wall_dist.copy(),
            "prey_rel_pos": prey_rel_pos.astype(np.float64),
            "prey_rel_vel": prey_rel_vel.astype(np.float64),
            "predator_rel_pos": predator_rel_pos.astype(np.float64),
            "predator_rel_vel": predator_rel_vel.astype(np.float64),
            "prey_capture_counts": prey_capture_counts.copy().astype(np.int32),
        }
