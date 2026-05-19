const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface PredictionResult {
  engine_id: number;
  anomaly_probability: number;
  rul_cycles: number;
  health_score: number;
  severity: 'NORMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  root_cause: string;
  maintenance_schedule: string;
  estimated_downtime: string;
  cost_saved: string;
  recommended_actions: string[];
  timeline: any;
  sensor_data: number[];
  timestamp: string;
}

export interface FleetEngine {
  engine_id: number;
  anomaly_probability: number;
  rul_cycles: number;
  health_score: number;
  severity: string;
}

export type SimulateMode = 'normal' | 'warning' | 'fault';
export type Severity = 'NORMAL' | 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

export async function simulateReading(
  mode: SimulateMode = 'normal',
  engineId: number = 1
): Promise<PredictionResult> {
  const res = await fetch(`${API_URL}/simulate?mode=${mode}&engine_id=${engineId}`);
  if (!res.ok) throw new Error('API error');
  return res.json();
}

export async function getFleet(count: number = 50): Promise<{ engines: FleetEngine[] }> {
  const res = await fetch(`${API_URL}/fleet?count=${count}`);
  if (!res.ok) throw new Error('API error');
  return res.json();
}

export async function getMetadata() {
  const res = await fetch(`${API_URL}/metadata`);
  if (!res.ok) return {};
  return res.json();
}

export async function getEvaluation() {
  const res = await fetch(`${API_URL}/evaluation`);
  if (!res.ok) return {};
  return res.json();
}

export const formatNum = (n: number) => n.toLocaleString("en-US");
export const formatPct = (n: number) => `${(n * 100).toFixed(1)}%`;
export const formatDate = (d: Date | string) => new Date(d).toISOString().slice(0, 10);

export const severityColor = (s: string) => {
  switch (s) {
    case "NORMAL": return "success";
    case "LOW": return "warning";
    case "MEDIUM": return "warning";
    case "HIGH": return "danger";
    case "CRITICAL": return "critical";
    default: return "success";
  }
};
export const severityEmoji = (s: string) => (({ NORMAL: "🟢", LOW: "🟡", MEDIUM: "🟠", HIGH: "🔴", CRITICAL: "💀" } as Record<string, string>)[s] || "🟢");
export const severityHex = (s: string) => (({ NORMAL: "#22C55E", LOW: "#EAB308", MEDIUM: "#F97316", HIGH: "#EF4444", CRITICAL: "#A855F7" } as Record<string, string>)[s] || "#22C55E");
export const healthGrade = (h: number) => (h >= 90 ? "A" : h >= 75 ? "B" : h >= 60 ? "C" : h >= 40 ? "D" : "F");
