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
