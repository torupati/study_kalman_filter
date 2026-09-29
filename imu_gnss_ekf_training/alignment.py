"""2D rigid (SE(2)) alignment between an estimated and a true trajectory.

A navigation system without an absolute sensor (e.g. IMU + visual odometry only)
estimates the trajectory in its own frame, anchored at its start pose. Its
estimate then differs from the world-frame truth by the rigid transform

    p_world = R(yaw_offset) p_nav + translation,    yaw_world = yaw_nav + yaw_offset

which, for a nav frame anchored at the start, is the true start pose. This module
fits that transform (least squares, no scale; Kabsch/Umeyama in 2D) and reports
the error left after alignment (absolute trajectory error, ATE), which is the
drift the navigation itself accumulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ekf import wrap_angle as _wrap


def _rotation(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


@dataclass(frozen=True)
class RigidTransform2d:
    yaw: float  # [rad]
    translation: np.ndarray  # (2,) [m]

    def apply(self, xy: np.ndarray, yaw: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Map nav-frame positions (N, 2) (and headings (N,)) into the world frame."""
        world_xy = np.asarray(xy, dtype=float) @ _rotation(self.yaw).T + self.translation
        world_yaw = None if yaw is None else _wrap(np.asarray(yaw, dtype=float) + self.yaw)
        return world_xy, world_yaw


def fit_rigid_2d(source_xy: np.ndarray, target_xy: np.ndarray) -> RigidTransform2d:
    """Least-squares rotation + translation with target ≈ R source + t (points paired by row)."""
    source_xy = np.asarray(source_xy, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float)
    source_mean = source_xy.mean(axis=0)
    target_mean = target_xy.mean(axis=0)
    a = source_xy - source_mean
    b = target_xy - target_mean
    # For 2D, the optimal angle maximizes sum(b · R a): atan2(sum(a x b), sum(a · b)).
    yaw = float(np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1])))
    translation = target_mean - _rotation(yaw) @ source_mean
    return RigidTransform2d(yaw=yaw, translation=translation)


@dataclass(frozen=True)
class TrajectoryAlignment:
    fitted: RigidTransform2d  # best fit nav -> world
    anchor: RigidTransform2d  # nav -> world from the true start pose
    fitted_xy: np.ndarray  # (N, 2) estimate mapped by `fitted`
    anchored_xy: np.ndarray  # (N, 2) estimate mapped by `anchor`
    anchored_yaw: np.ndarray  # (N,)
    ate_fitted_rmse: float  # [m] position RMSE after the best-fit alignment
    ate_anchored_rmse: float  # [m] position RMSE after mapping with the true start pose
    yaw_anchored_rmse: float  # [rad]


def align_trajectory(estimate_xy: np.ndarray, estimate_yaw: np.ndarray, truth_xy: np.ndarray, truth_yaw: np.ndarray) -> TrajectoryAlignment:
    """Compare a nav-frame estimate with world-frame truth, assuming the nav frame is anchored at the start.

    The estimate's own first pose is mapped onto the true first pose to build the
    anchor transform, so a nav frame that starts at (0, 0, 0) gives exactly the
    true start pose.
    """
    estimate_xy = np.asarray(estimate_xy, dtype=float)
    truth_xy = np.asarray(truth_xy, dtype=float)
    fitted = fit_rigid_2d(estimate_xy, truth_xy)
    anchor_yaw = float(_wrap(truth_yaw[0] - estimate_yaw[0]))
    anchor = RigidTransform2d(yaw=anchor_yaw, translation=truth_xy[0] - _rotation(anchor_yaw) @ estimate_xy[0])
    fitted_xy, _ = fitted.apply(estimate_xy)
    anchored_xy, anchored_yaw = anchor.apply(estimate_xy, estimate_yaw)

    def rmse(xy):
        return float(np.sqrt(np.mean(np.sum((xy - truth_xy) ** 2, axis=1))))

    return TrajectoryAlignment(
        fitted=fitted,
        anchor=anchor,
        fitted_xy=fitted_xy,
        anchored_xy=anchored_xy,
        anchored_yaw=anchored_yaw,
        ate_fitted_rmse=rmse(fitted_xy),
        ate_anchored_rmse=rmse(anchored_xy),
        yaw_anchored_rmse=float(np.sqrt(np.mean(_wrap(anchored_yaw - truth_yaw) ** 2))),
    )
