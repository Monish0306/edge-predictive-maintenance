import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { Bell, ChevronDown, Info, CheckCheck, Trash2 } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DarkTooltip } from "@/components/ChartTooltip";
import { alertStore, Alert, SEVERITY_CONFIG } from "@/lib/alertStore";
import { cn } from "@/lib/utils";

const SEVERITY_COLORS: Record<string, string> = {
  LOW:      "#EAB308",
  MEDIUM:   "#F97316",
  HIGH:     "#EF4444",
  CRITICAL: "#A855F7",
};

const AgentLog = () => {
  const [alerts, setAlerts]   = useState<Alert[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);

  // Subscribe to store updates
  useEffect(() => {
    const unsub = alertStore.subscribe(() => {
      setAlerts([...alertStore.getAlerts()]);
    });
    setAlerts([...alertStore.getAlerts()]);
    return () => { unsub(); };
  }, []);

  const counts = ["LOW", "MEDIUM", "HIGH", "CRITICAL"].map(s => ({
    name: s,
    value: alerts.filter(a => a.severity === s).length,
    color: SEVERITY_COLORS[s],
  }));

  if (!alerts.length) {
    return (
      <div>
        <PageHeader
          title="Maintenance Agent Alert History"
          subtitle="All alerts raised by the autonomous agent"
        />
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-8 flex items-start gap-4">
          <Info className="w-6 h-6 text-primary shrink-0 mt-0.5" />
          <div>
            <div className="font-semibold mb-1">No alerts yet</div>
            <div className="text-sm text-muted-foreground">
              Head to the Live Monitor and switch to "Warning" or "Fault" mode to generate alerts.
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader
        title="Maintenance Agent Alert History"
        subtitle={`${alerts.length} alert${alerts.length !== 1 ? "s" : ""} captured`}
      />

      {/* ── TOP METRICS ───────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <MetricCard
          label="Total Alerts"
          value={alerts.length}
          icon={Bell}
          color="danger"
          delay={0}
        />

        {/* Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-xl border border-border bg-card p-5 lg:col-span-2"
        >
          <h3 className="font-semibold mb-4 text-sm">Alert Severity Breakdown</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={counts}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip
                content={<DarkTooltip />}
                cursor={{ fill: "hsl(var(--muted) / 0.3)" }}
              />
              <Bar dataKey="value" radius={[6, 6, 0, 0]} animationDuration={900}>
                {counts.map((c, i) => (
                  <Cell key={i} fill={c.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* ── ACTIONS ───────────────────────────────────── */}
      <div className="flex items-center gap-3 mb-4">
        <button
          onClick={() => alertStore.acknowledgeAll()}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs bg-green-500/10 border border-green-500/20 text-green-400 hover:bg-green-500/20 transition-all"
        >
          <CheckCheck className="w-3.5 h-3.5" />
          Mark All Read
        </button>
        <button
          onClick={() => { alertStore.clearAll(); setAlerts([]); }}
          className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs bg-red-500/10 border border-red-500/20 text-red-400 hover:bg-red-500/20 transition-all"
        >
          <Trash2 className="w-3.5 h-3.5" />
          Clear All
        </button>
        <span className="text-xs text-muted-foreground ml-auto">
          Showing last {Math.min(alerts.length, 20)} alerts
        </span>
      </div>

      {/* ── ALERT LIST ────────────────────────────────── */}
      <div className="space-y-2">
        {alerts.slice(0, 20).map((a, i) => {
          const cfg   = SEVERITY_CONFIG[a.severity];
          const isExp = expanded === a.id;

          return (
            <motion.div
              key={a.id}
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.03 }}
              className={`rounded-xl border bg-card overflow-hidden transition-all ${
                !a.acknowledged ? "border-l-2" : "border-border opacity-70"
              }`}
              style={!a.acknowledged ? { borderLeftColor: cfg.color } : {}}
            >
              {/* Row */}
              <button
                onClick={() => setExpanded(isExp ? null : a.id)}
                className="w-full flex items-center gap-4 p-4 hover:bg-muted/30 transition-colors text-left"
              >
                {/* Status dot */}
                <span
                  className="w-3 h-3 rounded-full flex-shrink-0"
                  style={{
                    background: cfg.color,
                    boxShadow: a.acknowledged ? "none" : `0 0 6px ${cfg.color}`,
                  }}
                />

                {/* Timestamp */}
                <span className="font-mono text-xs text-muted-foreground w-44 flex-shrink-0">
                  {new Date(a.timestamp).toLocaleString()}
                </span>

                {/* Severity */}
                <span
                  className="font-bold text-xs px-2 py-0.5 rounded-full flex-shrink-0"
                  style={{ background: cfg.bg, color: cfg.color }}
                >
                  {a.severity}
                </span>

                {/* Probability */}
                <span className="text-sm font-mono flex-shrink-0">
                  {(a.anomaly_probability * 100).toFixed(1)}%
                </span>

                {/* Root cause */}
                <span className="text-sm text-foreground flex-1 truncate">
                  {a.root_cause}
                </span>

                {/* Engine */}
                <span className="text-xs text-muted-foreground flex-shrink-0">
                  E-{String(a.engine_id).padStart(3, "0")}
                </span>

                {/* Health */}
                <span
                  className="text-xs font-mono font-bold flex-shrink-0"
                  style={{ color: cfg.color }}
                >
                  {a.health_score.toFixed(0)}%
                </span>

                {/* Escalated */}
                {a.escalated && (
                  <span className="text-xs text-purple-400 flex-shrink-0">↑</span>
                )}

                <ChevronDown
                  className={cn(
                    "w-4 h-4 text-muted-foreground transition-transform flex-shrink-0",
                    isExp && "rotate-180"
                  )}
                />
              </button>

              {/* Expanded detail */}
              {isExp && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  className="border-t border-border bg-background/40 px-4 py-4"
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Alert Details */}
                    <div>
                      <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
                        Alert Details
                      </div>
                      <div className="space-y-1.5 text-sm">
                        <div className="flex gap-2">
                          <span className="text-muted-foreground w-28">Engine:</span>
                          <span className="font-mono">#{a.engine_id}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-muted-foreground w-28">Probability:</span>
                          <span className="font-mono" style={{ color: cfg.color }}>
                            {(a.anomaly_probability * 100).toFixed(2)}%
                          </span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-muted-foreground w-28">Health Score:</span>
                          <span className="font-mono">{a.health_score.toFixed(1)}%</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-muted-foreground w-28">Escalated:</span>
                          <span className={a.escalated ? "text-purple-400" : "text-green-400"}>
                            {a.escalated ? "Yes — Management notified" : "No"}
                          </span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-muted-foreground w-28">Status:</span>
                          <span className={a.acknowledged ? "text-green-400" : "text-yellow-400"}>
                            {a.acknowledged ? "Acknowledged" : "Unread"}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Message */}
                    <div>
                      <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2">
                        Message
                      </div>
                      <p className="text-sm text-foreground bg-black/20 rounded-lg p-3">
                        {a.message}
                      </p>
                      <button
                        onClick={() => alertStore.acknowledgeAlert(a.id)}
                        className="mt-2 text-xs text-green-400 hover:text-green-300 flex items-center gap-1"
                      >
                        <CheckCheck className="w-3 h-3" />
                        Mark as acknowledged
                      </button>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
};

export default AgentLog;