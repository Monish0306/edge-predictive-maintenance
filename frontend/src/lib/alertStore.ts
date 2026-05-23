// ── GLOBAL ALERT STORE ─────────────────────────────────
// Manages all alerts across the entire application

export type Severity = "NORMAL" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface Alert {
  id: string;
  engine_id: number;
  severity: Severity;
  anomaly_probability: number;
  health_score: number;
  root_cause: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  escalated: boolean;
}

export interface EscalationRule {
  severity: Severity;
  notify: string[];
  channel: "email" | "sms" | "both";
  delay_minutes: number;
}

export interface NotificationSettings {
  email_enabled: boolean;
  sms_enabled: boolean;
  email: string;
  phone: string;
  min_severity: Severity;
  escalation_rules: EscalationRule[];
  sound_enabled: boolean;
}

// ── DEFAULT SETTINGS ────────────────────────────────────
const DEFAULT_SETTINGS: NotificationSettings = {
  email_enabled: true,
  sms_enabled: false,
  email: "maintenance@factory.com",
  phone: "+1-555-0100",
  min_severity: "LOW",
  sound_enabled: true,
  escalation_rules: [
    {
      severity: "LOW",
      notify: ["Shift Supervisor"],
      channel: "email",
      delay_minutes: 30,
    },
    {
      severity: "MEDIUM",
      notify: ["Shift Supervisor", "Maintenance Lead"],
      channel: "email",
      delay_minutes: 15,
    },
    {
      severity: "HIGH",
      notify: ["Maintenance Lead", "Plant Manager"],
      channel: "both",
      delay_minutes: 5,
    },
    {
      severity: "CRITICAL",
      notify: ["Plant Manager", "Safety Officer", "CEO"],
      channel: "both",
      delay_minutes: 0,
    },
  ],
};

// ── SEVERITY CONFIG ─────────────────────────────────────
export const SEVERITY_CONFIG = {
  NORMAL:   { color: "#22C55E", bg: "rgba(34,197,94,0.1)",   border: "rgba(34,197,94,0.2)",   label: "Normal",   priority: 0 },
  LOW:      { color: "#EAB308", bg: "rgba(234,179,8,0.1)",   border: "rgba(234,179,8,0.2)",   label: "Low",      priority: 1 },
  MEDIUM:   { color: "#F97316", bg: "rgba(249,115,22,0.1)",  border: "rgba(249,115,22,0.2)",  label: "Medium",   priority: 2 },
  HIGH:     { color: "#EF4444", bg: "rgba(239,68,68,0.1)",   border: "rgba(239,68,68,0.2)",   label: "High",     priority: 3 },
  CRITICAL: { color: "#A855F7", bg: "rgba(168,85,247,0.1)",  border: "rgba(168,85,247,0.2)",  label: "Critical", priority: 4 },
};

// ── STORE CLASS ─────────────────────────────────────────
class AlertStore {
  private alerts: Alert[] = [];
  private settings: NotificationSettings = DEFAULT_SETTINGS;
  private listeners: Set<() => void> = new Set();
  private toastListeners: Set<(alert: Alert) => void> = new Set();

  // Subscribe to store changes
  subscribe(fn: () => void) {
    this.listeners.add(fn);
    return () => this.listeners.delete(fn);
  }

  // Subscribe to new alerts (for toast notifications)
  onNewAlert(fn: (alert: Alert) => void) {
    this.toastListeners.add(fn);
    return () => this.toastListeners.delete(fn);
  }

  private notify() {
    this.listeners.forEach(fn => fn());
  }

  // Add new alert
  addAlert(data: Omit<Alert, "id" | "timestamp" | "acknowledged" | "escalated">) {
    const minPriority = SEVERITY_CONFIG[this.settings.min_severity].priority;
    const alertPriority = SEVERITY_CONFIG[data.severity].priority;

    if (alertPriority < minPriority) return;
    if (data.severity === "NORMAL") return;

    const alert: Alert = {
      ...data,
      id: `alert-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
      timestamp: new Date().toISOString(),
      acknowledged: false,
      escalated: false,
    };

    this.alerts = [alert, ...this.alerts].slice(0, 100);
    this.notify();
    this.toastListeners.forEach(fn => fn(alert));

    // Auto-escalate critical
    if (data.severity === "CRITICAL" || data.severity === "HIGH") {
      setTimeout(() => this.escalateAlert(alert.id), 500);
    }

    return alert;
  }

  escalateAlert(id: string) {
    this.alerts = this.alerts.map(a =>
      a.id === id ? { ...a, escalated: true } : a
    );
    this.notify();
  }

  acknowledgeAlert(id: string) {
    this.alerts = this.alerts.map(a =>
      a.id === id ? { ...a, acknowledged: true } : a
    );
    this.notify();
  }

  acknowledgeAll() {
    this.alerts = this.alerts.map(a => ({ ...a, acknowledged: true }));
    this.notify();
  }

  clearAll() {
    this.alerts = [];
    this.notify();
  }

  getAlerts() { return this.alerts; }

  getUnreadCount() {
    return this.alerts.filter(a => !a.acknowledged).length;
  }

  getSettings() { return this.settings; }

  updateSettings(s: Partial<NotificationSettings>) {
    this.settings = { ...this.settings, ...s };
    this.notify();
  }

  getBySevertiy(severity: Severity) {
    return this.alerts.filter(a => a.severity === severity);
  }

  getSeverityCounts() {
    const counts: Record<Severity, number> = {
      NORMAL: 0, LOW: 0, MEDIUM: 0, HIGH: 0, CRITICAL: 0
    };
    this.alerts.forEach(a => counts[a.severity]++);
    return counts;
  }
}

// ── SINGLETON EXPORT ────────────────────────────────────
export const alertStore = new AlertStore();