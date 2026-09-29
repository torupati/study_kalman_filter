"""Navigation log: everything needed to replay/animate one EKF run, saved as `.npz`.

The file is a `np.savez_compressed` archive of plain arrays on the IMU time grid
(N steps) plus a `metadata_json` string. GNSS/innovation rows are NaN on steps
without a GNSS update. Optional arrays (`truth_states`, `imu_measurements`) are
simply absent from the archive when not available.

    key                      shape       notes
    time                     (N,)        [s]
    state_estimates          (N, 8)      posterior (after update)
    covariances              (N, 8, 8)
    prior_state_estimates    (N, 8)      after predict, before update
    prior_covariances        (N, 8, 8)
    gnss_available           (N,)        bool: a GNSS update was applied
    gnss_measurements        (N, 4)      [x, y, vx, vy]; NaN where unavailable
    gnss_covariance          (4, 4)      measurement noise R
    innovations              (N, 4)      z - H x_prior; NaN where unavailable
    innovation_covariances   (N, 4, 4)   H P_prior H^T + R; NaN where unavailable
    truth_states             (N, 8)      optional; NaN rows allowed
    imu_measurements         (N, 3)      optional; [ax, ay, wz] raw IMU
    metadata_json            ()          str: schema_version, state_names, ekf_config, ...
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .ekf import MEAS_SIZE, STATE_SIZE, EkfConfig, ImuGnssEkf

SCHEMA_VERSION = 1
STATE_NAMES = ("x", "y", "vx", "vy", "yaw", "accel_bias_x", "accel_bias_y", "gyro_bias")
MEASUREMENT_NAMES = ("x", "y", "vx", "vy")

_REQUIRED_ARRAYS = (
    "time",
    "state_estimates",
    "covariances",
    "prior_state_estimates",
    "prior_covariances",
    "gnss_available",
    "gnss_measurements",
    "gnss_covariance",
    "innovations",
    "innovation_covariances",
)
_OPTIONAL_ARRAYS = ("truth_states", "imu_measurements")


@dataclass
class NavLog:
    time: np.ndarray
    state_estimates: np.ndarray
    covariances: np.ndarray
    prior_state_estimates: np.ndarray
    prior_covariances: np.ndarray
    gnss_available: np.ndarray
    gnss_measurements: np.ndarray
    gnss_covariance: np.ndarray
    innovations: np.ndarray
    innovation_covariances: np.ndarray
    truth_states: np.ndarray | None = None
    imu_measurements: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gnss_available = np.asarray(self.gnss_available, dtype=bool)
        for name in (*_REQUIRED_ARRAYS, *_OPTIONAL_ARRAYS):
            value = getattr(self, name)
            if value is not None and name != "gnss_available":
                setattr(self, name, np.asarray(value, dtype=float))
        # Callers differ in how they fill rows without a fix (the simulator uses NaN,
        # the CSV loader zeros); normalize to NaN so "no fix" is unambiguous on disk.
        self.gnss_measurements = self.gnss_measurements.copy()
        self.gnss_measurements[~self.gnss_available] = np.nan
        self.validate()

    @property
    def num_steps(self) -> int:
        return int(self.time.shape[0])

    @property
    def has_truth(self) -> bool:
        return self.truth_states is not None

    def validate(self) -> None:
        """Raise ValueError if array shapes are inconsistent with the schema."""
        n = self.num_steps
        expected = {
            "time": (n,),
            "state_estimates": (n, STATE_SIZE),
            "covariances": (n, STATE_SIZE, STATE_SIZE),
            "prior_state_estimates": (n, STATE_SIZE),
            "prior_covariances": (n, STATE_SIZE, STATE_SIZE),
            "gnss_available": (n,),
            "gnss_measurements": (n, MEAS_SIZE),
            "gnss_covariance": (MEAS_SIZE, MEAS_SIZE),
            "innovations": (n, MEAS_SIZE),
            "innovation_covariances": (n, MEAS_SIZE, MEAS_SIZE),
            "truth_states": (n, STATE_SIZE),
            "imu_measurements": (n, 3),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value is not None and value.shape != shape:
                raise ValueError(f"NavLog.{name} has shape {value.shape}, expected {shape}")
        if n > 1 and not np.all(np.diff(self.time) > 0):
            raise ValueError("NavLog.time must be strictly increasing")

    @classmethod
    def from_run(
        cls,
        time: np.ndarray,
        result: dict[str, np.ndarray],
        gnss_measurements: np.ndarray,
        gnss_available: np.ndarray,
        config: EkfConfig,
        truth_states: np.ndarray | None = None,
        imu_measurements: np.ndarray | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NavLog:
        """Build a log from `run_filter`'s result dict and the inputs it was given.

        `metadata` is merged over the standard entries (schema version, state names, EKF config).
        """
        full_metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "state_names": list(STATE_NAMES),
            "measurement_names": list(MEASUREMENT_NAMES),
            "ekf_config": dataclasses.asdict(config),
        }
        full_metadata.update(metadata or {})
        return cls(
            time=time,
            state_estimates=result["state_estimates"],
            covariances=result["covariances"],
            prior_state_estimates=result["prior_state_estimates"],
            prior_covariances=result["prior_covariances"],
            gnss_available=gnss_available,
            gnss_measurements=gnss_measurements,
            gnss_covariance=ImuGnssEkf(config).R,
            innovations=result["innovations"],
            innovation_covariances=result["innovation_covariances"],
            truth_states=truth_states,
            imu_measurements=imu_measurements,
            metadata=full_metadata,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: getattr(self, name) for name in (*_REQUIRED_ARRAYS, *_OPTIONAL_ARRAYS) if getattr(self, name) is not None}
        metadata = {"schema_version": SCHEMA_VERSION, **self.metadata}
        np.savez_compressed(path, metadata_json=np.array(json.dumps(metadata)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> NavLog:
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"])) if "metadata_json" in archive else {}
            version = metadata.get("schema_version")
            if version != SCHEMA_VERSION:
                raise ValueError(f"{path}: unsupported NavLog schema_version {version!r} (expected {SCHEMA_VERSION})")
            missing = [name for name in _REQUIRED_ARRAYS if name not in archive]
            if missing:
                raise ValueError(f"{path}: NavLog is missing arrays {missing}")
            arrays = {name: archive[name] for name in (*_REQUIRED_ARRAYS, *_OPTIONAL_ARRAYS) if name in archive}
        return cls(**arrays, metadata=metadata)
