from __future__ import annotations

import numpy as np
from absl.testing import absltest, parameterized

from contgrid.core.grid import WallCollisionChecker


class ClipPositionTest(parameterized.TestCase):
    def setUp(self) -> None:
        super().setUp()
        layout = ["###", "#0#", "###"]
        self.checker = WallCollisionChecker(layout, L=0.1, verbose=False)
        self.robot_radius = 0.02
        self.allowed_overlap = 0.01

    @parameterized.named_parameters(
        ("free_space", (0.1, 0.1), (0.12, 0.12), True),
        ("zero_movement", (0.1, 0.1), (0.1, 0.1), True),
        ("positive_wall", (0.1, 0.1), (0.18, 0.1), False),
        ("negative_wall", (0.1, 0.1), (0.02, 0.1), False),
        ("diagonal_corner", (0.1, 0.1), (0.18, 0.18), False),
    )
    def test_clip_new_position(
        self,
        curr_pos: tuple[float, float],
        new_pos: tuple[float, float],
        expect_unclipped: bool,
    ) -> None:
        clipped = self.checker.clip_new_position(
            self.robot_radius, self.allowed_overlap, curr_pos, new_pos
        )
        if expect_unclipped:
            np.testing.assert_allclose(clipped, new_pos)
        else:
            self.assertTrue(
                self.checker.is_position_valid(
                    self.robot_radius, self.allowed_overlap, clipped
                )
            )


if __name__ == "__main__":
    absltest.main()
