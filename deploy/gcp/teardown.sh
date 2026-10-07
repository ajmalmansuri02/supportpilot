#!/usr/bin/env bash
# Delete everything deploy.sh created, so nothing keeps using credits.
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
gcloud config set project "$PROJECT_ID"
gcloud run services delete supportpilot-web --region "$REGION" --quiet || true
gcloud run services delete supportpilot-api --region "$REGION" --quiet || true
gcloud artifacts repositories delete supportpilot --location "$REGION" --quiet || true
gcloud secrets delete supportpilot-database-url --quiet || true
gcloud secrets delete supportpilot-gemini-key --quiet || true
echo "Done. The service account supportpilot-api is left in place (it costs nothing)."
