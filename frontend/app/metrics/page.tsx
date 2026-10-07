"use client";

import { useEffect, useState } from "react";
import { Metrics, getMetrics } from "@/lib/api";
import styles from "../page.module.css";

const usd = (n: number | null | undefined) => (n == null ? "–" : `$${n.toFixed(n < 0.01 ? 5 : 3)}`);

export default function MetricsPage() {
  const [hours, setHours] = useState(24);
  const [data, setData] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics(hours).then(setData).catch((e) => setError(String(e)));
  }, [hours]);

  const totalCost = data?.by_purpose.reduce((sum, r) => sum + r.cost_usd, 0) ?? 0;
  const totalCalls = data?.by_purpose.reduce((sum, r) => sum + r.calls, 0) ?? 0;

  return (
    <main className={styles.shell} style={{ height: "auto", paddingBottom: 32 }}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Metrics</h1>
          <p className={styles.subtitle}>Cost, latency, cache and guardrails</p>
        </div>
        <div className={styles.controls}>
          <select value={hours} onChange={(e) => setHours(Number(e.target.value))} aria-label="Time range">
            <option value={1}>Last hour</option>
            <option value={24}>Last 24 hours</option>
            <option value={168}>Last 7 days</option>
          </select>
          <a href="/" className={styles.secondary}>
            Back to chat
          </a>
        </div>
      </header>

      {error && <p className={styles.error}>{error}</p>}
      {data && (
        <>
          <section className={styles.tiles}>
            <div className={styles.tile}>
              <span>LLM calls</span>
              <strong>{totalCalls}</strong>
            </div>
            <div className={styles.tile}>
              <span>Total cost</span>
              <strong>{usd(totalCost)}</strong>
            </div>
            <div className={styles.tile}>
              <span>Cost per conversation</span>
              <strong>{usd(data.per_conversation.avg_cost_usd)}</strong>
            </div>
            <div className={styles.tile}>
              <span>Cache hits</span>
              <strong>{data.cache.hits}</strong>
            </div>
          </section>

          <h2 className={styles.h2}>By purpose</h2>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Purpose</th>
                  <th>Model</th>
                  <th>Calls</th>
                  <th>Avg ms</th>
                  <th>p95 ms</th>
                  <th>Tokens in / out</th>
                  <th>Cost</th>
                  <th>Errors</th>
                </tr>
              </thead>
              <tbody>
                {data.by_purpose.map((r) => (
                  <tr key={r.purpose + r.model}>
                    <td>{r.purpose}</td>
                    <td>{r.model}</td>
                    <td>{r.calls}</td>
                    <td>{r.avg_latency_ms}</td>
                    <td>{r.p95_latency_ms}</td>
                    <td>
                      {r.input_tokens} / {r.output_tokens}
                    </td>
                    <td>{usd(r.cost_usd)}</td>
                    <td>{r.errors}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className={styles.h2}>Guardrails</h2>
          {Object.keys(data.guardrails).length === 0 ? (
            <p className={styles.muted}>Nothing blocked or redacted in this period.</p>
          ) : (
            <ul>
              {Object.entries(data.guardrails).map(([kind, n]) => (
                <li key={kind}>
                  {kind}: {n}
                </li>
              ))}
            </ul>
          )}
          <p className={styles.muted}>
            Costs use PRICE_INPUT_PER_M and PRICE_OUTPUT_PER_M from .env (0 for local models).
          </p>
        </>
      )}
    </main>
  );
}
