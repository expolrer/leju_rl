import math
import unittest

import numpy as np

from prepare_kuavo_s53_stairs import (
    S53_JOINT_NAMES,
    Episode,
    canonicalize_root,
    choose_medoid,
    retarget_joint_positions,
)


class RetargetTests(unittest.TestCase):
    def test_joint_mapping_drops_head_and_adds_zero_waist(self):
        source_names = [
            "leg_l1_joint",
            "leg_r1_joint",
            "zarm_l1_joint",
            "zarm_r1_joint",
            "zhead_1_joint",
            "leg_l2_joint",
            "leg_r2_joint",
            "zarm_l2_joint",
            "zarm_r2_joint",
            "zhead_2_joint",
            "leg_l3_joint",
            "leg_r3_joint",
            "zarm_l3_joint",
            "zarm_r3_joint",
            "leg_l4_joint",
            "leg_r4_joint",
            "zarm_l4_joint",
            "zarm_r4_joint",
            "leg_l5_joint",
            "leg_r5_joint",
            "zarm_l5_joint",
            "zarm_r5_joint",
            "leg_l6_joint",
            "leg_r6_joint",
            "zarm_l6_joint",
            "zarm_r6_joint",
            "zarm_l7_joint",
            "zarm_r7_joint",
        ]
        source = np.arange(56, dtype=np.float64).reshape(2, 28)
        result = retarget_joint_positions(source, source_names)
        self.assertEqual(result.shape, (2, 27))
        self.assertTrue(np.all(result[:, S53_JOINT_NAMES.index("waist_yaw_joint")] == 0.0))
        for name in set(S53_JOINT_NAMES) - {"waist_yaw_joint"}:
            np.testing.assert_array_equal(
                result[:, S53_JOINT_NAMES.index(name)], source[:, source_names.index(name)]
            )

    def test_canonicalize_rotates_positive_y_motion_to_positive_x(self):
        root_pos = np.array([[2.0, 3.0, 0.5], [2.0, 4.0, 0.6]])
        yaw = 0.5 * math.pi
        quat = np.tile([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], (2, 1))
        pos, result_quat, initial_yaw = canonicalize_root(root_pos, quat, 0.925)
        self.assertAlmostEqual(initial_yaw, yaw)
        np.testing.assert_allclose(pos[0], [0.0, 0.0, 0.925], atol=1.0e-10)
        np.testing.assert_allclose(pos[1, :2], [1.0, 0.0], atol=1.0e-10)
        np.testing.assert_allclose(result_quat, [[1.0, 0.0, 0.0, 0.0]] * 2, atol=1.0e-10)

    def test_medoid_is_an_observed_episode(self):
        episodes = []
        for index, offset in enumerate((0.0, 0.1, 10.0)):
            episodes.append(
                Episode(
                    name=str(index),
                    source_path=None,
                    fps=50.0,
                    root_pos=np.array([[0.0, 0.0, 0.925], [offset, 0.0, 0.925]]),
                    root_quat=np.tile([1.0, 0.0, 0.0, 0.0], (2, 1)),
                    joint_pos=np.full((2, 27), offset),
                    stage=np.array([7, 8]),
                )
            )
        index, distances = choose_medoid(episodes)
        self.assertEqual(index, 1)
        self.assertEqual(distances.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
