-- SupportPilot database schema. Safe to run many times (everything is IF NOT EXISTS).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Week 2: conversation memory ------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at  timestamptz NOT NULL DEFAULT now(),
    summary     text,                      -- rolling summary of older messages
    escalated   boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS messages (
    id              bigserial PRIMARY KEY,
    conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            text NOT NULL CHECK (role IN ('user', 'assistant')),
    content         text NOT NULL,
    summarized      boolean NOT NULL DEFAULT false,  -- folded into conversations.summary
    meta            jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS messages_conversation_idx ON messages (conversation_id, id);

-- Week 3: documents for RAG -------------------------------------------------
-- {embed_dim} is filled in from EMBED_DIM when the schema is applied.
CREATE TABLE IF NOT EXISTS documents (
    id           bigserial PRIMARY KEY,
    source       text NOT NULL UNIQUE,       -- file name, e.g. refunds.md
    title        text NOT NULL,
    content_hash text NOT NULL,              -- skip re-embedding unchanged files
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    document_id bigint NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index int NOT NULL,
    heading     text NOT NULL,               -- "Refund policy > Annual plans"
    content     text NOT NULL,
    embedding   vector({embed_dim}) NOT NULL,
    -- Week 4: full-text search column for hybrid (keyword + vector) search.
    tsv         tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', heading), 'A') ||
                    setweight(to_tsvector('english', content), 'B')) STORED
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
