# Deploying SupportPilot to the cloud (week 12)

Locally SupportPilot runs on Ollama. Employers also want to see that you can run the same
app on a managed AI platform, with secrets, identity and a cost limit set up properly. The
provider switch from week 1 means no code changes: only settings.

| Piece | Where it runs | Cost |
| --- | --- | --- |
| Chat model | Gemini on **Vertex AI** (`LLM_PROVIDER=vertex`) | Pay per token, covered by the Google Cloud free trial credits |
| Embeddings | Gemini API free tier (`EMBED_PROVIDER=gemini`) | Free |
| Database | [Neon](https://neon.tech) free tier (Postgres with pgvector) | Free |
| Backend and frontend | Cloud Run, scaled to zero, max 1 instance | Inside Cloud Run's free tier for a demo |
| Secrets | Secret Manager | Free tier |

> Not tested end to end from this repo's build environment (no Google Cloud project there).
> Model names change: check the Vertex AI Model Garden and pass `CHAT_MODEL` / `FAST_MODEL`
> if the defaults in `deploy.sh` are gone.

## 0. Set a budget alert first

New Google Cloud accounts get free trial credits. Before deploying anything, create a budget
so you get an email long before you spend real money:

```bash
gcloud billing accounts list                   # copy the ACCOUNT_ID
gcloud billing budgets create --billing-account ACCOUNT_ID \
  --display-name "supportpilot" --budget-amount 5USD \
  --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0
```

A budget alerts; it does not stop spending. `--max-instances 1`, scale to zero and
`teardown.sh` are what keep the bill near zero.

## 1. Try Vertex AI from your laptop

```bash
gcloud auth application-default login          # Vertex uses your identity, not an API key
gcloud services enable aiplatform.googleapis.com
cd backend && uv sync --extra dev --extra vertex
```

In `.env`:

```bash
LLM_PROVIDER=vertex
VERTEX_PROJECT=your-project-id
VERTEX_LOCATION=global
CHAT_MODEL=google/gemini-2.5-flash
FAST_MODEL=google/gemini-2.5-flash-lite
```

Keep `EMBED_PROVIDER=ollama` locally if you like: the chat and embedding providers are
independent. Run the evals (`app.evals.rag_eval`, `agent_eval`, `redteam_eval`) with Vertex
and compare against your Ollama results. That table is a strong portfolio item.

## 2. Create a free database

Create a Neon project, run `CREATE EXTENSION vector;` in its SQL editor, and copy the
**direct** (not pooled) connection string. The backend creates its tables on first start.

## 3. Deploy

```bash
export PROJECT_ID=your-project-id
export DATABASE_URL='postgresql://user:pass@ep-xxx.neon.tech/neondb?sslmode=require'
export GEMINI_API_KEY=your-ai-studio-key
bash deploy/gcp/deploy.sh                      # from the repo root
```

The script enables the APIs, stores the two secrets in Secret Manager, creates a service
account that can only call Vertex AI and read those secrets, builds both images with Cloud
Build (no local Docker needed) and deploys them to Cloud Run. It prints the app URL at the end.

On first start the backend loads the help-centre docs and demo customers into the database.

**What to point out in an interview:** no keys in the image or the repo; the backend calls
Vertex with its service account identity (no API key at all); least-privilege roles; a
cost ceiling; and the same evals run against local and cloud models.

## 4. Tear it down

```bash
bash deploy/gcp/teardown.sh
```

The services are public (`--allow-unauthenticated`) so you can show them to people. Anyone
with the URL can spend your credits, so delete them after the demo, or redeploy without
that flag.

## Alternative: Azure OpenAI

If you have Azure credits instead, create an Azure OpenAI (Azure AI Foundry) resource and
deploy a model, then use the generic provider. The v1 endpoint works with the standard
OpenAI client:

```bash
LLM_PROVIDER=openai_compatible
OPENAI_COMPATIBLE_BASE_URL=https://YOUR-RESOURCE.openai.azure.com/openai/v1/
OPENAI_COMPATIBLE_API_KEY=your-azure-key
CHAT_MODEL=your-deployment-name
FAST_MODEL=your-small-deployment-name
```

The backend image runs unchanged on Azure Container Apps. Set a budget in Cost Management
first, as above.
