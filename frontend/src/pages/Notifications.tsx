import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import {
  Bell, Mail, MessageSquare, Shield,
  ChevronRight, Save, Trash2, CheckCheck,
  Activity, Clock, Users
} from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import {
  alertStore, Alert, SEVERITY_CONFIG,
  NotificationSettings, Severity
} from "@/lib/alertStore";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts";

const SEVERITY_ORDER: Severity[] = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"];
const COLORS = ["#22C55E", "#EAB308", "#F97316", "#EF4444", "#A855F7"];

export default function Notifications() {
  const [alerts, setAlerts]     = useState<Alert[]>([]);
  const [settings, setSettings] = useState<NotificationSettings>(
    alertStore.getSettings()
  );
  const [saved, setSaved]       = useState(false);

  useEffect(() => {
    const unsub = alertStore.subscribe(() => {
      setAlerts(alertStore.getAlerts());
      setSettings(alertStore.getSettings());
    });
    setAlerts(alertStore.getAlerts());
    return () => {
      unsub();
    };
  }, []);

  const handleSave = () => {
    alertStore.updateSettings(settings);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const counts = alertStore.getSeverityCounts();

  const chartData = SEVERITY_ORDER.slice(1).map((sev, i) => ({
    name: sev,
    count: counts[sev],
    color: COLORS[i + 1],
  }));

  return (
    <div>
      <PageHeader
        title="Alert Notifications"
        subtitle="Configure escalation rules, notification channels, and alert history"
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* LEFT: Settings ──────────────────────────────── */}
        <div className="lg:col-span-2 space-y-6">

          {/* Summary cards */}
          <div className="grid grid-cols-4 gap-3">
            {SEVERITY_ORDER.slice(1).map((sev, i) => {
              const cfg = SEVERITY_CONFIG[sev];
              return (
                <motion.div
                  key={sev}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.1 }}
                  className="rounded-xl border p-4"
                  style={{ borderColor: cfg.border, background: cfg.bg }}
                >
                  <div className="text-2xl font-black font-mono" style={{ color: cfg.color }}>
                    {counts[sev]}
                  </div>
                  <div className="text-xs text-slate-400 uppercase tracking-wider mt-1">
                    {sev}
                  </div>
                </motion.div>
              );
            })}
          </div>

          {/* Notification Channels */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h3 className="font-bold text-white mb-4 flex items-center gap-2">
              <Bell className="w-4 h-4 text-blue-400" />
              Notification Channels
            </h3>

            <div className="space-y-4">
              {/* Email */}
              <div className="flex items-center gap-4 p-3 rounded-lg bg-black/20">
                <Mail className="w-5 h-5 text-blue-400 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-white mb-1">
                    Email Notifications
                  </div>
                  <input
                    type="email"
                    value={settings.email}
                    onChange={e => setSettings({ ...settings, email: e.target.value })}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
                    placeholder="maintenance@factory.com"
                  />
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.email_enabled}
                    onChange={e => setSettings({ ...settings, email_enabled: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-5 bg-slate-700 rounded-full peer peer-checked:bg-blue-500 transition-colors relative">
                    <div className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-all ${settings.email_enabled ? "translate-x-5" : ""}`} />
                  </div>
                </label>
              </div>

              {/* SMS */}
              <div className="flex items-center gap-4 p-3 rounded-lg bg-black/20">
                <MessageSquare className="w-5 h-5 text-green-400 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-white mb-1">
                    SMS Notifications
                  </div>
                  <input
                    type="tel"
                    value={settings.phone}
                    onChange={e => setSettings({ ...settings, phone: e.target.value })}
                    className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-white"
                    placeholder="+1-555-0100"
                  />
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.sms_enabled}
                    onChange={e => setSettings({ ...settings, sms_enabled: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-5 bg-slate-700 rounded-full peer peer-checked:bg-green-500 transition-colors relative">
                    <div className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-all ${settings.sms_enabled ? "translate-x-5" : ""}`} />
                  </div>
                </label>
              </div>

              {/* Sound */}
              <div className="flex items-center gap-4 p-3 rounded-lg bg-black/20">
                <Bell className="w-5 h-5 text-yellow-400 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-white">Sound Alerts</div>
                  <div className="text-xs text-slate-400">Play audio tone for new alerts</div>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.sound_enabled}
                    onChange={e => setSettings({ ...settings, sound_enabled: e.target.checked })}
                    className="sr-only peer"
                  />
                  <div className="w-10 h-5 bg-slate-700 rounded-full peer peer-checked:bg-yellow-500 transition-colors relative">
                    <div className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-all ${settings.sound_enabled ? "translate-x-5" : ""}`} />
                  </div>
                </label>
              </div>

              {/* Min severity */}
              <div className="flex items-center gap-4 p-3 rounded-lg bg-black/20">
                <Shield className="w-5 h-5 text-purple-400 flex-shrink-0" />
                <div className="flex-1">
                  <div className="text-sm font-medium text-white mb-1">
                    Minimum Alert Severity
                  </div>
                  <select
                    value={settings.min_severity}
                    onChange={e => setSettings({ ...settings, min_severity: e.target.value as Severity })}
                    className="bg-slate-800 border border-slate-700 text-white text-sm rounded-lg px-3 py-1.5 w-full"
                  >
                    {["LOW", "MEDIUM", "HIGH", "CRITICAL"].map(s => (
                      <option key={s} value={s}>{s} and above</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Escalation Rules */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h3 className="font-bold text-white mb-4 flex items-center gap-2">
              <Users className="w-4 h-4 text-orange-400" />
              Escalation Rules
            </h3>

            <div className="space-y-3">
              {settings.escalation_rules.map((rule, i) => {
                const cfg = SEVERITY_CONFIG[rule.severity];
                return (
                  <div
                    key={rule.severity}
                    className="rounded-lg border p-4"
                    style={{ borderColor: cfg.border, background: cfg.bg + "50" }}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <div
                        className="px-3 py-1 rounded-full text-xs font-black uppercase"
                        style={{ background: cfg.color + "20", color: cfg.color }}
                      >
                        {rule.severity}
                      </div>
                      <ChevronRight className="w-3 h-3 text-slate-500" />
                      <div className="flex gap-1 flex-wrap">
                        {rule.notify.map(person => (
                          <span
                            key={person}
                            className="px-2 py-0.5 rounded text-xs bg-slate-700 text-slate-300"
                          >
                            {person}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-xs text-slate-400 block mb-1">Channel</label>
                        <select
                          value={rule.channel}
                          onChange={e => {
                            const updated = [...settings.escalation_rules];
                            updated[i] = { ...rule, channel: e.target.value as any };
                            setSettings({ ...settings, escalation_rules: updated });
                          }}
                          className="w-full bg-slate-800 border border-slate-700 text-white text-xs rounded-lg px-2 py-1.5"
                        >
                          <option value="email">Email only</option>
                          <option value="sms">SMS only</option>
                          <option value="both">Email + SMS</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-xs text-slate-400 block mb-1">
                          Delay (minutes)
                        </label>
                        <input
                          type="number"
                          min="0"
                          max="60"
                          value={rule.delay_minutes}
                          onChange={e => {
                            const updated = [...settings.escalation_rules];
                            updated[i] = { ...rule, delay_minutes: +e.target.value };
                            setSettings({ ...settings, escalation_rules: updated });
                          }}
                          className="w-full bg-slate-800 border border-slate-700 text-white text-xs rounded-lg px-2 py-1.5"
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <button
              onClick={handleSave}
              className={`mt-4 w-full flex items-center justify-center gap-2 py-3 rounded-xl font-bold text-sm transition-all ${
                saved
                  ? "bg-green-500/20 border border-green-500/30 text-green-400"
                  : "bg-blue-500/20 border border-blue-500/30 text-blue-400 hover:bg-blue-500/30"
              }`}
            >
              <Save className="w-4 h-4" />
              {saved ? "✓ Settings Saved!" : "Save Settings"}
            </button>
          </motion.div>
        </div>

        {/* RIGHT: Alert History ─────────────────────────── */}
        <div className="space-y-4">

          {/* Chart */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="rounded-xl border border-border bg-card p-5"
          >
            <h3 className="font-bold text-white text-sm mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-blue-400" />
              Alert Distribution
            </h3>
            <ResponsiveContainer width="100%" height={160}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
                <XAxis dataKey="name" fontSize={10} stroke="#64748B" />
                <YAxis fontSize={10} stroke="#64748B" />
                <Tooltip
                  contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
                  labelStyle={{ color: "#9CA3AF" }}
                />
                <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                  {chartData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Alert list */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="rounded-xl border border-border bg-card p-4"
          >
            <div className="flex items-center justify-between mb-3">
              <h3 className="font-bold text-white text-sm flex items-center gap-2">
                <Clock className="w-4 h-4 text-slate-400" />
                Recent Alerts
              </h3>
              <div className="flex gap-1">
                <button
                  onClick={() => alertStore.acknowledgeAll()}
                  className="text-xs text-slate-400 hover:text-white flex items-center gap-1 px-2 py-1 rounded hover:bg-white/5"
                >
                  <CheckCheck className="w-3 h-3" />
                  All read
                </button>
                <button
                  onClick={() => alertStore.clearAll()}
                  className="text-xs text-slate-400 hover:text-red-400 flex items-center gap-1 px-2 py-1 rounded hover:bg-red-500/10"
                >
                  <Trash2 className="w-3 h-3" />
                  Clear
                </button>
              </div>
            </div>

            {alerts.length === 0 ? (
              <div className="text-center py-8">
                <Bell className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                <p className="text-xs text-slate-500">
                  No alerts yet. Start Live Monitor to begin.
                </p>
              </div>
            ) : (
              <div className="space-y-2 max-h-96 overflow-y-auto">
                {alerts.slice(0, 20).map(alert => {
                  const cfg = SEVERITY_CONFIG[alert.severity];
                  return (
                    <div
                      key={alert.id}
                      className={`flex items-start gap-2 p-2 rounded-lg transition-all ${
                        !alert.acknowledged ? "bg-white/5" : "opacity-50"
                      }`}
                    >
                      <div
                        className="w-2 h-2 rounded-full mt-1.5 flex-shrink-0"
                        style={{
                          background: cfg.color,
                          boxShadow: alert.acknowledged ? "none" : `0 0 6px ${cfg.color}`,
                        }}
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-black" style={{ color: cfg.color }}>
                            {alert.severity}
                          </span>
                          <span className="text-[10px] text-slate-500">
                            E#{alert.engine_id}
                          </span>
                          {alert.escalated && (
                            <span className="text-[10px] text-purple-400">↑</span>
                          )}
                        </div>
                        <p className="text-xs text-white truncate">{alert.root_cause}</p>
                        <p className="text-[10px] text-slate-500 mt-0.5">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                      {!alert.acknowledged && (
                        <button
                          onClick={() => alertStore.acknowledgeAlert(alert.id)}
                          className="text-slate-600 hover:text-green-400 flex-shrink-0"
                        >
                          <CheckCheck className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}