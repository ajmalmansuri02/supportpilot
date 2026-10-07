"use client";

import { FormEvent, useEffect, useRef, useState } from "react";
import { ChatEvent, Mode, getHealth, streamChat } from "@/lib/api";
import styles from "./page.module.css";

type Message = {
  role: "user" | "assistant";
  text: string;
  error?: string;
};

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "chat", label: "Chat", hint: "Plain LLM conversation with memory" },
];

const EXAMPLES = ["Hi! What can you help me with?", "Write a haiku about taking notes."];

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<Mode>("chat");
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
          <button onClick={newChat} className={styles.secondary}>
            New chat
          </button>
        </div>
      </header>

      <section className={styles.messages}>
        {messages.length === 0 && (
          <div className={styles.empty}>
            <p>Ask anything about CloudNotes. Try:</p>
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
              {m.text || (busy && i === messages.length - 1 ? <span className={styles.typing}>…</span> : null)}
              {m.error && <p className={styles.error}>{m.error}</p>}
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
