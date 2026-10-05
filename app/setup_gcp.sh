#!/usr/bin/env bash
# One-time GCP setup for the imu_1d web UI's CI/CD (cloudbuild.yaml). Safe to rerun: existing resources are kept.
#
#   PROJECT=<project-id> app/setup_gcp.sh                               # steps 1-3
#   PROJECT=<project-id> GITHUB_CONNECTION=<name> app/setup_gcp.sh      # + step 4: trigger, and a first build of main
#
# 1. enable the APIs
# 2. Artifact Registry repo `apps` (images) and bucket gs://<project>-imu1d-movies (movie cache)
# 3. service accounts: imu1d-build (runs Cloud Build: deploys, pushes images, writes logs)
#                      imu1d-run   (Cloud Run identity: only reads/writes the movie bucket)
# 4. link the GitHub repo through the Cloud Build connection GITHUB_CONNECTION and create the trigger
#    `imu1d-deploy` (push to main, only when app/linear_kf/build files change). The connection itself
#    needs a browser login to GitHub, so it is not created here; without GITHUB_CONNECTION the script
#    prints how to make one.
set -euo pipefail

: "${PROJECT:?set PROJECT to the GCP project id}"
REGION=${REGION:-asia-northeast1}
SERVICE=${SERVICE:-imu1d}           # must match _SERVICE in cloudbuild.yaml
REPO=${REPO:-apps}                  # must match _IMAGE in cloudbuild.yaml
BUCKET=${PROJECT}-imu1d-movies    # fixed in cloudbuild.yaml
GITHUB_REPO=${GITHUB_REPO:-https://github.com/torupati/study_kalman_filter.git}
GITHUB_CONNECTION=${GITHUB_CONNECTION:-}
TRIGGER=${SERVICE}-deploy

BUILD_SA=${SERVICE}-build@${PROJECT}.iam.gserviceaccount.com
RUN_SA=${SERVICE}-run@${PROJECT}.iam.gserviceaccount.com
gc() { gcloud --project="$PROJECT" --quiet "$@"; }
step() { printf '\n== %s\n' "$*"; }

step "1. enable APIs"
gc services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com iam.googleapis.com storage.googleapis.com

step "2. Artifact Registry repo '$REPO' and bucket gs://$BUCKET"
gc artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1 \
  || gc artifacts repositories create "$REPO" --repository-format=docker --location="$REGION"
gc storage buckets describe "gs://$BUCKET" >/dev/null 2>&1 \
  || gc storage buckets create "gs://$BUCKET" --location="$REGION" --uniform-bucket-level-access

step "3. service accounts"
for sa in build run; do
  gc iam service-accounts describe "${SERVICE}-${sa}@${PROJECT}.iam.gserviceaccount.com" >/dev/null 2>&1 \
    || gc iam service-accounts create "${SERVICE}-${sa}" --display-name="${SERVICE} ${sa}"
done
for role in roles/run.admin roles/artifactregistry.writer roles/logging.logWriter; do
  gc projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:$BUILD_SA" --role="$role" --condition=None >/dev/null
done
# deploying a service that runs as RUN_SA requires actAs on it
gc iam service-accounts add-iam-policy-binding "$RUN_SA" --member="serviceAccount:$BUILD_SA" \
  --role=roles/iam.serviceAccountUser >/dev/null
gc storage buckets add-iam-policy-binding "gs://$BUCKET" --member="serviceAccount:$RUN_SA" \
  --role=roles/storage.objectAdmin >/dev/null
# the Cloud Build service agent keeps the GitHub connection's token in Secret Manager
gcloud beta services identity create --service=cloudbuild.googleapis.com --project="$PROJECT" --quiet >/dev/null
NUMBER=$(gc projects describe "$PROJECT" --format='value(projectNumber)')
gc projects add-iam-policy-binding "$PROJECT" --member="serviceAccount:service-${NUMBER}@gcp-sa-cloudbuild.iam.gserviceaccount.com" \
  --role=roles/secretmanager.admin --condition=None >/dev/null

if [[ -z "$GITHUB_CONNECTION" ]]; then
  cat <<EOF

== 4. skipped: GITHUB_CONNECTION is not set
Create a Cloud Build connection to GitHub (once; opens a browser login and the Cloud Build GitHub App install):
  gcloud builds connections create github github --region=$REGION --project=$PROJECT
  # follow the printed URL, install the app on ${GITHUB_REPO#https://github.com/}, then check:
  gcloud builds connections describe github --region=$REGION --project=$PROJECT   # installationState: COMPLETE
and rerun:
  PROJECT=$PROJECT GITHUB_CONNECTION=github $0
EOF
  exit 0
fi

step "4. link $GITHUB_REPO and create trigger '$TRIGGER'"
REPO_NAME=$(basename "$GITHUB_REPO" .git)
gc builds repositories describe "$REPO_NAME" --connection="$GITHUB_CONNECTION" --region="$REGION" >/dev/null 2>&1 \
  || gc builds repositories create "$REPO_NAME" --remote-uri="$GITHUB_REPO" --connection="$GITHUB_CONNECTION" --region="$REGION"
if ! gc builds triggers describe "$TRIGGER" --region="$REGION" >/dev/null 2>&1; then
  gc builds triggers create github --name="$TRIGGER" --region="$REGION" \
    --repository="projects/$PROJECT/locations/$REGION/connections/$GITHUB_CONNECTION/repositories/$REPO_NAME" \
    --branch-pattern='^main$' --build-config=cloudbuild.yaml \
    --service-account="projects/$PROJECT/serviceAccounts/$BUILD_SA" \
    --included-files='app/**,linear_kf/**,Dockerfile,.dockerignore,pyproject.toml,uv.lock,.python-version,cloudbuild.yaml' \
    --substitutions="_SERVICE=$SERVICE,_REGION=$REGION"
fi

step "first build of main"
gc builds triggers run "$TRIGGER" --region="$REGION" --branch=main
cat <<EOF

Build started; follow it at https://console.cloud.google.com/cloud-build/builds;region=$REGION?project=$PROJECT
When it finishes:  gcloud run services describe $SERVICE --region=$REGION --project=$PROJECT --format='value(status.url)'
EOF
