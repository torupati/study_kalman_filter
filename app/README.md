# Web UI for the 1D IMU + position KF

`streamlit_app.py` puts `linear_kf/imu_1d` behind a web page: the simulation settings are in the sidebar,
the **Plots** tab (truth/measurements, error ± 2σ, velocity error vs. a position-only KF, RMSE/NIS table)
is recomputed on every change, and the **Movie** tab renders `animate.py`'s predict/update movie on request.

Rendering takes tens of milliseconds per frame (seconds to a couple of minutes per movie), so each movie is cached under a hash
of its inputs and of the `imu_1d` source files; anyone asking for the same movie later gets it at once, and
editing `imu_1d` invalidates the old ones. The cache is

- the GCS bucket `$IMU1D_MOVIE_BUCKET` (objects under `imu1d/`) when that variable is set — use this on Cloud Run,
  whose local disk is lost when an instance stops;
- otherwise the local directory `$IMU1D_MOVIE_CACHE_DIR` (default `outputs/streamlit_movie_cache/`).

## Run locally

```bash
uv run --group app streamlit run app/streamlit_app.py     # http://localhost:8501
```

Without ffmpeg the movie is written as a GIF instead of MP4.

## Deploy to Cloud Run (CI/CD)

Every push to `main` that touches `app/`, `linear_kf/` or the build files runs `cloudbuild.yaml` on Cloud Build:
**test** (`uv run pytest`) → **build** the root `Dockerfile` → **push** to Artifact Registry
(`<region>-docker.pkg.dev/<project>/apps/imu1d:<commit>` and `:latest`) → **deploy** to the Cloud Run service `imu1d`.
GitHub Actions (`.github/workflows/tests.yml`) still checks PRs; make its `lint`/`pytest` checks required
on `main` so only tested code reaches the trigger.

All Cloud Run settings are in `cloudbuild.yaml`'s deploy step, so change them there, not in the console:

- `--timeout=3600`: Streamlit talks over one websocket, which Cloud Run cuts at the request timeout; a full-quality render must fit in it.
- `--session-affinity`: Streamlit keeps session state in the instance, so a browser must keep hitting the same one.
- `--max-instances=3`: caps cost if the URL gets passed around.
- `_AUTH_FLAG` (substitution): `--allow-unauthenticated` makes the app public; set it to `--no-allow-unauthenticated`
  on the trigger to keep it private and grant `roles/run.invoker` to the people who should use it.

### One-time setup

`app/setup_gcp.sh` creates everything the pipeline needs and is safe to rerun:

```bash
PROJECT=<project-id> app/setup_gcp.sh
```

1. enables the APIs (Run, Cloud Build, Artifact Registry, Secret Manager, IAM, Storage);
2. creates the Artifact Registry repo `apps` and the movie-cache bucket `gs://<project>-imu1d-movies`;
3. creates two service accounts: `imu1d-build` (runs the build: `run.admin`, `artifactregistry.writer`,
   `logging.logWriter`, and `serviceAccountUser` on `imu1d-run`) and `imu1d-run` (the app's identity: only
   `storage.objectAdmin` on the bucket, so the public app cannot redeploy itself).

Then connect GitHub. `app/connect_github.sh` creates the Cloud Build connection `github` and waits for it,
printing the two links you open in a browser: authorize Cloud Build on your GitHub account, then install the
Cloud Build GitHub App on `torupati/study_kalman_filter`:

```bash
PROJECT=<project-id> bash app/connect_github.sh
```

Finally rerun the script with the connection. It links the repo, creates the trigger `imu1d-deploy`
(push to `main`) and starts a first build of `main`:

```bash
PROJECT=<project-id> GITHUB_CONNECTION=github app/setup_gcp.sh
gcloud run services describe imu1d --region=asia-northeast1 --project=<project-id> --format='value(status.url)'
```

Rollback: `gcloud run services update-traffic imu1d --region=asia-northeast1 --to-revisions=<revision>=100`,
or redeploy an older image tag.
