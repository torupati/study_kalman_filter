#!/usr/bin/env bash
# Create the Cloud Build connection to GitHub that app/setup_gcp.sh's trigger uses, and wait until it is usable.
#
#   PROJECT=<project-id> bash app/connect_github.sh
#
# Two steps need a web browser, and the script prints the link for each as the connection reaches it:
#   PENDING_USER_OAUTH   authorize Cloud Build to use your GitHub account
#   PENDING_INSTALL_APP  install the Cloud Build GitHub App on the repository
# It polls the connection until its state is COMPLETE (TIMEOUT seconds, default 900). Rerunning it is safe:
# an existing connection is reused and the script just continues from its current state.
set -euo pipefail

: "${PROJECT:?set PROJECT to the GCP project id}"
REGION=${REGION:-asia-northeast1}
CONNECTION=${CONNECTION:-github}
GITHUB_REPO=${GITHUB_REPO:-torupati/study_kalman_filter}
TIMEOUT=${TIMEOUT:-900}
POLL=5
gc() { gcloud --project="$PROJECT" --quiet "$@"; }

if gc builds connections describe "$CONNECTION" --region="$REGION" >/dev/null 2>&1; then
  echo "connection '$CONNECTION' exists in $REGION; checking its state"
else
  echo "creating connection '$CONNECTION' in $REGION"
  gc builds connections create github "$CONNECTION" --region="$REGION" >/dev/null
fi

last=""
deadline=$((SECONDS + TIMEOUT))
while :; do
  IFS=$'\t' read -r stage uri < <(gc builds connections describe "$CONNECTION" --region="$REGION" \
    --format='value(installationState.stage,installationState.actionUri)')
  [[ "$stage" == COMPLETE ]] && break
  if [[ "$stage" != "$last" ]]; then
    case "$stage" in
      PENDING_USER_OAUTH) echo; echo "Open this link and authorize Cloud Build to access your GitHub account:" ;;
      PENDING_INSTALL_APP) echo; echo "Open this link and install the Cloud Build GitHub App on $GITHUB_REPO (only that repository is enough):" ;;
      *) echo; echo "connection state: ${stage:-unknown}" ;;
    esac
    [[ -n "$uri" ]] && echo "  $uri"
    echo "waiting for it to finish (checking every ${POLL} s, Ctrl-C to stop; rerun to resume)"
    last=$stage
  fi
  if ((SECONDS >= deadline)); then
    echo "gave up after ${TIMEOUT} s in state $stage; finish the step above and rerun this script" >&2
    exit 1
  fi
  sleep "$POLL"
done

cat <<EOF

connection '$CONNECTION' is COMPLETE.
Next, once cloudbuild.yaml is on main, create the trigger and start the first deploy:
  PROJECT=$PROJECT GITHUB_CONNECTION=$CONNECTION bash app/setup_gcp.sh
EOF
