#!/usr/bin/env bash
# Deploy SupportPilot to Google Cloud Run (week 12). Run from the repo root:
#
#   export PROJECT_ID=my-project DATABASE_URL='postgresql://...' GEMINI_API_KEY=...
#   bash deploy/gcp/deploy.sh
#
# Chat runs on Gemini through Vertex AI (paid from your free-trial credits), embeddings on
# the Gemini free tier, and Postgres on a free Neon database. See deploy/README.md.
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
: "${DATABASE_URL:?set DATABASE_URL (a Postgres with pgvector, e.g. Neon free tier)}"
: "${GEMINI_API_KEY:?set GEMINI_API_KEY (used for embeddings)}"
REGION="${REGION:-us-central1}"
CHAT_MODEL="${CHAT_MODEL:-google/gemini-2.5-flash}"
FAST_MODEL="${FAST_MODEL:-google/gemini-2.5-flash-lite}"
REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/supportpilot"
SA="supportpilot-api@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"

echo "==> Enabling APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com

echo "==> Image repository"
gcloud artifacts repositories describe supportpilot --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create supportpilot --repository-format docker --location "$REGION"

echo "==> Secrets (stored in Secret Manager, never in the image)"
put_secret() {
  if gcloud secrets describe "$1" >/dev/null 2>&1; then
    printf '%s' "$2" | gcloud secrets versions add "$1" --data-file=-
  else
    printf '%s' "$2" | gcloud secrets create "$1" --data-file=-
  fi
}
put_secret supportpilot-database-url "$DATABASE_URL"
put_secret supportpilot-gemini-key "$GEMINI_API_KEY"

echo "==> Service account with only the permissions it needs"
gcloud iam service-accounts describe "$SA" >/dev/null 2>&1 ||
  gcloud iam service-accounts create supportpilot-api --display-name "SupportPilot API"
for role in roles/aiplatform.user roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member "serviceAccount:$SA" \
    --role "$role" --condition None >/dev/null
done

echo "==> Building and deploying the API"
gcloud builds submit --config deploy/gcp/cloudbuild.yaml \
  --substitutions "_IMAGE=${REPO}/api,_DOCKERFILE=backend/Dockerfile,_CONTEXT=."
ENV_FILE="$(mktemp)"
cat > "$ENV_FILE" <<YAML
LLM_PROVIDER: vertex
VERTEX_PROJECT: "${PROJECT_ID}"
VERTEX_LOCATION: global
CHAT_MODEL: "${CHAT_MODEL}"
FAST_MODEL: "${FAST_MODEL}"
EMBED_PROVIDER: gemini
EMBED_MODEL: gemini-embedding-001
EMBED_DIM: "768"
EMBED_QUERY_PREFIX: ""
EMBED_DOCUMENT_PREFIX: ""
RAG_MIN_SIMILARITY: "0.5"
YAML
gcloud run deploy supportpilot-api --image "${REPO}/api" --region "$REGION" \
  --service-account "$SA" --allow-unauthenticated --max-instances 1 --memory 1Gi \
  --env-vars-file "$ENV_FILE" \
  --set-secrets "DATABASE_URL=supportpilot-database-url:latest,GEMINI_API_KEY=supportpilot-gemini-key:latest"
rm -f "$ENV_FILE"
API_URL="$(gcloud run services describe supportpilot-api --region "$REGION" --format 'value(status.url)')"

echo "==> Building and deploying the web app"
gcloud builds submit --config deploy/gcp/cloudbuild.yaml \
  --substitutions "_IMAGE=${REPO}/web,_DOCKERFILE=frontend/Dockerfile,_CONTEXT=frontend,_API_URL=${API_URL}"
gcloud run deploy supportpilot-web --image "${REPO}/web" --region "$REGION" \
  --allow-unauthenticated --max-instances 1 --port 3000
WEB_URL="$(gcloud run services describe supportpilot-web --region "$REGION" --format 'value(status.url)')"

echo "==> Allowing the web app to call the API"
gcloud run services update supportpilot-api --region "$REGION" \
  --update-env-vars "CORS_ORIGINS=${WEB_URL}"

echo
echo "API: ${API_URL}/docs"
echo "App: ${WEB_URL}"
