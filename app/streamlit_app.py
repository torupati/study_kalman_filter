"""Web UI for the 1D IMU + position KF (linear_kf/imu_1d): plots on every change, movie on request.

    uv run --group app streamlit run app/streamlit_app.py

The plots are recomputed on every input change (simulate + filter take well under a second).
The movie is the slow part (minutes), so each rendered movie is cached under a key hashed from
its inputs and the imu_1d source code; a repeated request is served from the cache.
The cache is the GCS bucket named by $IMU1D_MOVIE_BUCKET (for Cloud Run, where the local disk is
lost on restart), otherwise a local directory ($IMU1D_MOVIE_CACHE_DIR, default outputs/streamlit_movie_cache).
"""

import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.animation import FFMpegWriter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # `streamlit run` puts only app/ on sys.path

from linear_kf.imu_1d import animate, demo  # noqa: E402
from linear_kf.imu_1d.plotting import plot_compare_velocity, plot_errors, plot_overview  # noqa: E402
from linear_kf.imu_1d.simulator import SCENARIOS, SimConfig, simulate  # noqa: E402

MOVIE_MAX_DURATION = 30.0  # [s] of simulated time; render time grows linearly with it
CODE_FILES = ("animate.py", "kf.py", "simulator.py", "plotting.py", "demo.py")  # a change in any of them invalidates the cache


# ---------------------------------------------------------------------------- movie cache


class LocalStore:
    def __init__(self, root: Path):
        self.root = root

    def get(self, name: str) -> bytes | None:
        p = self.root / name
        return p.read_bytes() if p.exists() else None

    def put(self, name: str, data: bytes) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / name).write_bytes(data)


class GcsStore:
    def __init__(self, bucket: str, prefix: str = "imu1d"):
        from google.cloud import storage

        self.bucket = storage.Client().bucket(bucket)
        self.prefix = prefix

    def get(self, name: str) -> bytes | None:
        blob = self.bucket.blob(f"{self.prefix}/{name}")
        return blob.download_as_bytes() if blob.exists() else None

    def put(self, name: str, data: bytes) -> None:
        content_type = "video/mp4" if name.endswith(".mp4") else "image/gif"
        self.bucket.blob(f"{self.prefix}/{name}").upload_from_string(data, content_type=content_type)


@st.cache_resource
def movie_store() -> LocalStore | GcsStore:
    if bucket := os.environ.get("IMU1D_MOVIE_BUCKET"):
        return GcsStore(bucket)
    return LocalStore(Path(os.environ.get("IMU1D_MOVIE_CACHE_DIR", REPO_ROOT / "outputs" / "streamlit_movie_cache")))


@st.cache_resource
def code_version() -> str:
    h = hashlib.sha256()
    for name in CODE_FILES:
        h.update((REPO_ROOT / "linear_kf" / "imu_1d" / name).read_bytes())
    return h.hexdigest()[:12]


def movie_key(cfg: SimConfig, seed: int, movie: dict, suffix: str) -> str:
    payload = json.dumps({"cfg": dataclasses.asdict(cfg), "seed": seed, "movie": movie, "code": code_version()}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24] + suffix


def render_movie(cfg: SimConfig, seed: int, movie: dict, suffix: str, progress) -> bytes:
    sim = simulate(cfg, np.random.default_rng(seed))
    trk = animate.run_track(sim)
    m = animate.make_animation(sim, trk, movie["fps"], movie["speed"], movie["update_pause"])

    def callback(i: int, n: int):
        if (i + 1) % 10 == 0 or i + 1 == n:
            progress.progress((i + 1) / n, text=f"rendering frame {i + 1}/{n}")

    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"movie{suffix}"
            m.save(out, movie["fps"], movie["dpi"], progress_callback=callback)
            return out.read_bytes()
    finally:
        plt.close(m.fig)


# ---------------------------------------------------------------------------- UI


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def sidebar() -> tuple[SimConfig, int]:
    sb = st.sidebar
    sb.header("Simulation")
    d = SimConfig()
    scenario = sb.selectbox("scenario", sorted(SCENARIOS), index=sorted(SCENARIOS).index(d.scenario),
                            format_func=lambda s: f"{s} — {SCENARIOS[s].description}")
    duration = sb.slider("duration [s]", 10.0, 120.0, d.duration, 5.0)
    accel_noise_density = sb.number_input("accel noise density [m/s²/√Hz]", 0.0, 1.0, d.accel_noise_density, 0.01, format="%.3f")
    accel_bias = sb.number_input("accel bias [m/s²] (not modelled by the KF)", -0.5, 0.5, d.accel_bias, 0.01, format="%.3f")
    pos_noise_std = sb.number_input("position noise std [m]", 0.01, 10.0, d.pos_noise_std, 0.1)
    pos_rate = sb.selectbox("position rate [Hz]", [0.5, 1.0, 2.0, 5.0, 10.0], index=1)
    outage = None
    if sb.checkbox("position outage", value=True):
        start, end = sb.slider("outage [s]", 0.0, duration, (min(9.0, duration), min(15.0, duration)), 1.0)
        outage = (start, end) if end > start else None
    seed = int(sb.number_input("random seed", 0, 2**31 - 1, 0, 1))
    cfg = SimConfig(scenario=scenario, duration=duration, pos_rate=pos_rate, accel_noise_density=accel_noise_density,
                    accel_bias=accel_bias, pos_noise_std=pos_noise_std, pos_outage=outage)
    return cfg, seed


def plots_tab(cfg: SimConfig, seed: int):
    sim = simulate(cfg, np.random.default_rng(seed))
    res = demo.run_imu_kf(sim)
    res_pos = demo.run_position_only_kf(sim)

    rows = []
    for name, r in (("IMU + position KF", res), ("position-only KF", res_pos)):
        s = demo.summarize(sim, r)
        rows.append({"filter": name, "pos RMSE [m]": s["pos_rmse"], "pos σ RMS [m]": s["pos_sigma_rms"],
                     "vel RMSE [m/s]": s["vel_rmse"], "vel σ RMS [m/s]": s["vel_sigma_rms"], "mean NIS": s["mean_nis"]})
    st.caption("Metrics exclude the first 5 s (initial convergence). Mean NIS ≈ 1 means the filter's σ is consistent.")
    st.dataframe(rows, hide_index=True, column_config={k: st.column_config.NumberColumn(format="%.3f") for k in rows[0] if k != "filter"})

    show(plot_overview(sim, f"scenario {cfg.scenario}: truth and measurements"))
    show(plot_errors(sim, res, "IMU + position KF: error and $\\pm 2\\sigma$"))
    show(plot_compare_velocity(sim, {"IMU + position KF": res, "position-only KF (random acceleration model)": res_pos}, "velocity error"))


def movie_tab(cfg: SimConfig, seed: int):
    st.write("Position density spreading in predict and shrinking at each position update, with the (p, v) error ellipse. "
             "Uses the simulation settings from the sidebar, cut to the movie duration below.")
    c1, c2, c3, c4 = st.columns(4)
    duration = c1.slider("movie duration [s]", 5.0, MOVIE_MAX_DURATION, min(20.0, cfg.duration, MOVIE_MAX_DURATION), 1.0)
    quality = c2.radio("quality", ["preview (fast)", "full"], horizontal=True)
    speed = c3.slider("playback speed", 0.5, 4.0, 1.0, 0.5)
    pause = c4.checkbox("stop at each position update", value=False,
                        help="Stop 1.5 s at each fix while the likelihood fades in and the prior turns into the posterior.")
    fps, dpi = (15.0, 72) if quality.startswith("preview") else (30.0, 100)
    movie = {"fps": fps, "dpi": dpi, "speed": speed, "update_pause": 1.5 if pause else 0.0}

    outage = cfg.pos_outage
    if outage is not None:
        outage = (min(outage[0], duration), min(outage[1], duration))
        outage = outage if outage[1] > outage[0] else None
    mcfg = dataclasses.replace(cfg, duration=duration, pos_outage=outage)

    sim = simulate(mcfg, np.random.default_rng(seed))
    show(animate.plot_error_history(sim, animate.run_track(sim)))

    suffix = ".mp4" if FFMpegWriter.isAvailable() else ".gif"
    key = movie_key(mcfg, seed, movie, suffix)
    store = movie_store()
    data = store.get(key)
    if data is None:
        st.info("This movie has not been rendered yet. Rendering takes several seconds (preview) to a minute or two (full, with stops); "
                "once done it is cached for everyone.")
        if not st.button("Render movie", type="primary"):
            return
        data = render_movie(mcfg, seed, movie, suffix, st.progress(0.0, text="starting"))
        store.put(key, data)
    if suffix == ".mp4":
        st.video(data)
    else:
        st.image(data)
    st.download_button("Download", data, file_name=f"kf_imu1d_{key}", mime="video/mp4" if suffix == ".mp4" else "image/gif")


def main():
    st.set_page_config(page_title="1D IMU + position KF", layout="wide")
    st.title("1D Kalman filter: accelerometer input + position updates")
    st.write("State $x = [p, v]$, the accelerometer drives the prediction, position measurements correct it. "
             "Change the simulation in the sidebar.")
    cfg, seed = sidebar()
    plots, movie = st.tabs(["Plots", "Movie"])
    with plots:
        plots_tab(cfg, seed)
    with movie:
        movie_tab(cfg, seed)


main()
