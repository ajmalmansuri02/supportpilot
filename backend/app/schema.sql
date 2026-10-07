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

-- Week 6: mock customer data the agent can look up ------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    id            bigserial PRIMARY KEY,
    email         text NOT NULL UNIQUE,
    name          text NOT NULL,
    plan          text NOT NULL CHECK (plan IN ('free', 'pro', 'team')),
    billing_cycle text CHECK (billing_cycle IN ('monthly', 'annual')),
    seats         int NOT NULL DEFAULT 1,
    status        text NOT NULL DEFAULT 'active',
    created_at    timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS invoices (
    id          text PRIMARY KEY,                 -- e.g. INV-204518
    customer_id bigint NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    amount_usd  numeric(10, 2) NOT NULL,
    description text NOT NULL,
    status      text NOT NULL DEFAULT 'paid' CHECK (status IN ('paid', 'refunded', 'failed')),
    issued_at   timestamptz NOT NULL
);

-- Weeks 6-8: support tickets (also exposed through the MCP server) -------------------
CREATE TABLE IF NOT EXISTS tickets (
    id              bigserial PRIMARY KEY,
    conversation_id uuid REFERENCES conversations(id) ON DELETE SET NULL,
    customer_email  text,
    subject         text NOT NULL,
    description     text NOT NULL,
    category        text NOT NULL DEFAULT 'other',
    priority        text NOT NULL DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high')),
    status          text NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'pending', 'resolved')),
    assignee        text NOT NULL DEFAULT 'ai' CHECK (assignee IN ('ai', 'human')),
    notes           jsonb NOT NULL DEFAULT '[]'::jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    updated_at      timestamptz NOT NULL DEFAULT now()
);

-- Week 7: risky actions wait here until a human approves them ------------------------
CREATE TABLE IF NOT EXISTS pending_actions (
    id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id uuid REFERENCES conversations(id) ON DELETE CASCADE,
    tool            text NOT NULL,
    args            jsonb NOT NULL,
    description     text NOT NULL,
    status          text NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'approved', 'rejected', 'failed')),
    result          jsonb,
    created_at      timestamptz NOT NULL DEFAULT now(),
    decided_at      timestamptz
);
