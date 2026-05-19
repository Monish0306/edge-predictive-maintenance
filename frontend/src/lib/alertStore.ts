import { useEffect, useState } from "react";

export interface AlertEntry {
  id: string;
  timestamp: string;
  severity: "HIGH" | "CRITICAL";
  probability: number;
  health_score: number;
  root_cause: string;
  cost_saved: number;
  actions: string[];
}

let listeners: Array<() => void> = [];
let state: { alerts: AlertEntry[] } = { alerts: [] };

export const alertStore = {
  get: () => state.alerts,
  add: (a: AlertEntry) => { state = { alerts: [a, ...state.alerts].slice(0, 50) }; listeners.forEach((l) => l()); },
  clear: () => { state = { alerts: [] }; listeners.forEach((l) => l()); },
  subscribe: (fn: () => void) => { listeners.push(fn); return () => { listeners = listeners.filter((l) => l !== fn); }; },
};

export const useAlerts = () => {
  const [, force] = useState(0);
  useEffect(() => alertStore.subscribe(() => force((n) => n + 1)), []);
  return alertStore.get();
};
