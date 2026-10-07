"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { ChatEvent, Mode, Source, decideAction, getHealth, streamChat } from "@/lib/api";
import styles from "./page.module.css";

type Message = {
  role: "user" | "assistant";
  text: string;
  error?: string;
  sources?: Source[];
  query?: string;
  classification?: { category: string; priority: string; sentiment: string };
  steps?: Step[];
  approvals?: Approval[];
  escalatedTicket?: number;
  cacheHit?: string;
  guardrails?: string[];
};

type Step = { id: string; name: string; args: Record<string, unknown>; result?: unknown };

type Approval = {
  actionId: string;
  description: string;
  status: "pending" | "approved" | "rejected" | "failed";
  message?: string;
};

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "agent", label: "Agent", hint: "Uses tools: docs search, accounts, tickets, refunds" },
  { value: "rag", label: "Docs (RAG)", hint: "Answers from the CloudNotes docs with citations" },
  { value: "chat", label: "Plain chat", hint: "Plain LLM conversation with memory, no documents" },
];

const EXAMPLES = [
  "How much does the Pro plan cost?",
  "I was charged twice this month. My email is priya@example.com",
  "What plan am I on? My email is rahul@example.com",
  "I'd like to talk to a human, please",
  "Is there a Linux desktop app?",
];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("agent");
  const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [health, setHealth] = useState<string>("connecting…");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getHealth()
      .then((h) => setHealth(`${h.provider} · ${h.chat_model}`))
      .catch(() => setHealth("backend offline"));
  }, []);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [messages]);

  function updateLast(fn: (m: Message) => Message) {
    setMessages((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])]);
  }

  function handleEvent(event: ChatEvent) {
    switch (event.type) {
      case "meta":
        setConversationId(event.conversation_id);
        break;
      case "sources":
        updateLast((m) => ({
          ...m,
          sources: event.sources,
          query: event.query,
          cacheHit: event.cache_hit ? event.cached_question : undefined,
        }));
        break;
      case "classification":
        updateLast((m) => ({ ...m, classification: event }));
        break;
      case "tool_call":
        updateLast((m) => ({
          ...m,
          steps: [...(m.steps ?? []), { id: event.id, name: event.name, args: event.args }],
        }));
        break;
      case "tool_result":
        updateLast((m) => ({
          ...m,
          steps: (m.steps ?? []).map((s) => (s.id === event.id ? { ...s, result: event.result } : s)),
        }));
        break;
      case "approval_required":
        updateLast((m) => ({
          ...m,
          approvals: [
            ...(m.approvals ?? []),
            { actionId: event.action_id, description: event.description, status: "pending" },
          ],
        }));
        break;
      case "escalated":
        updateLast((m) => ({ ...m, escalatedTicket: event.ticket_id }));
        break;
      case "guardrail":
        updateLast((m) => ({
          ...m,
          guardrails: [
            ...(m.guardrails ?? []),
            event.kind === "injection" ? "Blocked by the input guardrail" : `Redacted: ${event.detail}`,
          ],
        }));
        break;
      case "replace":
        updateLast((m) => ({ ...m, text: event.text, guardrails: [...(m.guardrails ?? []), "Reply replaced: prompt leak"] }));
        break;
      case "token":
        updateLast((m) => ({ ...m, text: m.text + event.text }));
        break;
      case "error":
        updateLast((m) => ({ ...m, error: event.message }));
        break;
    }
  }

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setInput("");
    setBusy(true);
    setMessages((prev) => [...prev, { role: "user", text }, { role: "assistant", text: "" }]);
    try {
      await streamChat({ message: text, conversation_id: conversationId, mode }, handleEvent);
    } catch (err) {
      updateLast((m) => ({ ...m, error: String(err) }));
    } finally {
      setBusy(false);
    }
  }

  async function decide(index: number, actionId: string, approve: boolean) {
    try {
      const res = await decideAction(actionId, approve);
      setMessages((prev) =>
        prev.map((m, i) =>
          i !== index
            ? m
            : {
                ...m,
                approvals: m.approvals?.map((a) =>
                  a.actionId === actionId
                    ? { ...a, status: res.status as Approval["status"], message: res.message }
                    : a,
                ),
              },
        ),
      );
    } catch (err) {
      alert(String(err));
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  function newChat() {
    setMessages([]);
    setConversationId(null);
  }

  return (
    <main className={styles.shell}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>SupportPilot</h1>
          <p className={styles.subtitle}>CloudNotes customer support · {health}</p>
        </div>
        <div className={styles.controls}>
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value as Mode)}
            title={MODES.find((m) => m.value === mode)?.hint}
            aria-label="Mode"
          >
            {MODES.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          <a href="/metrics" className={styles.secondary}>
            Metrics
          </a>
          <button onClick={newChat} className={styles.secondary}>
            New chat
          </button>
        </div>
      </header>

      <section className={styles.messages}>
        {messages.length === 0 && (
          <div className={styles.empty}>
            <p>
              Ask anything about CloudNotes. Demo accounts: priya@example.com (charged twice),
              rahul@example.com (Team), maria@example.com, alex@example.com (Free).
            </p>
            {EXAMPLES.map((ex) => (
              <button key={ex} className={styles.example} onClick={() => send(ex)}>
                {ex}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? styles.user : styles.assistant}>
            <div className={styles.bubble}>
              {m.guardrails?.map((g) => (
                <div key={g} className={styles.guard}>
                  🛡 {g}
                </div>
              ))}
              {m.cacheHit && <div className={styles.muted}>⚡ Cached answer (similar to “{m.cacheHit}”)</div>}
              {m.classification && (
                <div className={styles.badges}>
                  <span className={styles.badge}>{m.classification.category}</span>
                  <span className={styles.badge}>priority: {m.classification.priority}</span>
                  <span className={styles.badge}>{m.classification.sentiment}</span>
                </div>
              )}
              {m.steps && m.steps.length > 0 && (
                <details className={styles.steps}>
                  <summary>
                    {m.steps.length} tool call{m.steps.length > 1 ? "s" : ""}:{" "}
                    {m.steps.map((s) => s.name).join(" → ")}
                  </summary>
                  {m.steps.map((s) => (
                    <div key={s.id} className={styles.step}>
                      <code>
                        {s.name}({JSON.stringify(s.args)})
                      </code>
                      {s.result !== undefined && (
                        <pre>{JSON.stringify(s.result, null, 2).slice(0, 1200)}</pre>
                      )}
                    </div>
                  ))}
                </details>
              )}
              {m.text || (busy && i === messages.length - 1 ? <span className={styles.typing}>…</span> : null)}
              {m.error && <p className={styles.error}>{m.error}</p>}
              {m.escalatedTicket && (
                <p className={styles.notice}>Handed to a human agent · ticket #{m.escalatedTicket}</p>
              )}
              {m.approvals?.map((a) => (
                <div key={a.actionId} className={styles.approval}>
                  <div className={styles.approvalTitle}>Support agent approval needed</div>
                  <div>{a.description}</div>
                  {a.status === "pending" ? (
                    <div className={styles.approvalButtons}>
                      <button className={styles.primary} onClick={() => decide(i, a.actionId, true)}>
                        Approve
                      </button>
                      <button className={styles.secondary} onClick={() => decide(i, a.actionId, false)}>
                        Reject
                      </button>
                    </div>
                  ) : (
                    <div className={a.status === "approved" ? styles.ok : styles.muted}>
                      {a.message ?? a.status}
                    </div>
                  )}
                </div>
              ))}
              {m.sources && m.sources.length > 0 && (
                <details className={styles.sources}>
                  <summary>
                    {m.sources.length} sources
                    {m.query ? ` · searched for “${m.query}”` : ""}
                  </summary>
                  <ol>
                    {m.sources.map((s) => (
                      <li key={s.n}>
                        <strong>[{s.n}] {s.heading}</strong>{" "}
                        <span className={styles.muted}>({s.source})</span>
                        <p>{s.content}</p>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </div>
          </div>
        ))}
        <div ref={bottom} />
      </section>

      <form onSubmit={onSubmit} className={styles.composer}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          placeholder="Type your question… (Enter to send, Shift+Enter for a new line)"
          rows={2}
          aria-label="Message"
        />
        <button type="submit" disabled={busy || !input.trim()} className={styles.primary}>
          {busy ? "…" : "Send"}
        </button>
      </form>
    </main>
  );
}
