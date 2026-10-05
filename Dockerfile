# Web UI for linear_kf/imu_1d (app/streamlit_app.py), for Cloud Run. See app/README.md.
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --group app --no-install-project
COPY linear_kf ./linear_kf
COPY app ./app

ENV PATH=/opt/venv/bin:$PATH MPLBACKEND=Agg
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8080} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
