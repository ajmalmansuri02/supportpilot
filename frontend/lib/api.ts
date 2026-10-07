// Talks to the FastAPI backend. The chat endpoint streams Server-Sent Events:
// each `data:` line is one JSON event (meta, token, sources, tool_call, ..., done).

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Mode = "chat" | "rag";

export type Source = { n: number; source: string; heading: string; content: string; score: number };

export type ChatEvent =
  | { type: "meta"; conversation_id: string; mode: Mode }
  | { type: "sources"; query: string; sources: Source[] }
  | { type: "token"; text: string }
  | { type: "done"; prompt_version?: number }
  | { type: "error"; message: string };

export async function streamChat(
  body: { message: string; conversation_id: string | null; mode: Mode },
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Backend returned ${res.status}: ${await res.text()}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Events are separated by a blank line.
    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      for (const line of raw.split("\n")) {
        if (line.startsWith("data: ")) onEvent(JSON.parse(line.slice(6)) as ChatEvent);
      }
    }
  }
}

export async function getHealth(): Promise<Record<string, string>> {
  const res = await fetch(`${API_URL}/api/health`);
  return res.json();
}
