/**
 * useKeepAlive.ts
 * ===============
 * Pings the Render backend every 10 minutes to prevent cold starts.
 * Render free tier spins down after 15 min inactivity → 30-60s cold start.
 * This keeps it warm so charts load instantly for interviewers.
 *
 * Usage: call useKeepAlive() once in App.tsx
 */

import { useEffect } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const PING_INTERVAL_MS = 10 * 60 * 1000; // 10 minutes

export function useKeepAlive(): void {
  useEffect(() => {
    // Ping immediately on app load — wakes backend before user clicks anything
    const ping = async () => {
      try {
        await fetch(`${API}/health`, {
          method:  "GET",
          // Short timeout — we don't want to block the UI waiting for ping
          signal:  AbortSignal.timeout(8000),
        });
        console.debug("[KeepAlive] Backend ping OK");
      } catch {
        // Silent fail — ping is best-effort, never breaks the app
      }
    };

    // Fire immediately on mount
    ping();

    // Then every 10 minutes to prevent Render spin-down
    const timer = setInterval(ping, PING_INTERVAL_MS);

    return () => clearInterval(timer);
  }, []); // runs once on mount
}