import { motion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { Bell, ChevronDown, Info } from "lucide-react";
import { useState } from "react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DarkTooltip } from "@/components/ChartTooltip";
import { useAlerts } from "@/lib/alertStore";
import { formatNum, severityHex } from "@/lib/api";
import { cn } from "@/lib/utils";

const AgentLog = () => {
  const alerts = useAlerts();
  const [expanded, setExpanded] = useState<string | null>(null);

  const counts = ["LOW", "MEDIUM", "HIGH", "CRITICAL"].map((s) => ({ name: s, value: alerts.filter((a) => a.severity === s).length }));

  if (!alerts.length) {
    return (
      <div>
        <PageHeader title="Maintenance Agent Alert History" subtitle="All alerts raised by the autonomous agent" />
        <div className="rounded-xl border border-primary/30 bg-primary/5 p-8 flex items-start gap-4">
          <Info className="w-6 h-6 text-primary shrink-0" />
          <div>
            <div className="font-semibold mb-1">No alerts yet</div>
            <div className="text-sm text-muted-foreground">Head to the Live Monitor and switch the simulator to "Warning" or "Fault" mode to start generating alerts.</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="Maintenance Agent Alert History" subtitle={`${alerts.length} alerts captured`} />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <MetricCard label="Total Alerts" value={alerts.length} icon={Bell} color="danger" />
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}
          className="rounded-xl border border-border bg-card p-5 lg:col-span-2">
          <h3 className="font-semibold mb-4 text-sm">Alert Severity Breakdown</h3>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={counts}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip content={<DarkTooltip />} cursor={{ fill: "hsl(var(--muted) / 0.3)" }} />
              <Bar dataKey="value" radius={[6, 6, 0, 0]} animationDuration={900}>
                {counts.map((c, i) => <Cell key={i} fill={severityHex(c.name as any)} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      <div className="space-y-2">
        {alerts.slice(0, 20).map((a, i) => (
          <motion.div key={a.id} initial={{ opacity: 0, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.03 }}
            className="rounded-xl border border-border bg-card overflow-hidden">
            <button onClick={() => setExpanded(expanded === a.id ? null : a.id)} className="w-full flex items-center gap-4 p-4 hover:bg-muted/30 transition-colors">
              <span className="w-3 h-3 rounded-full pulse-dot" style={{ background: severityHex(a.severity) }} />
              <span className="font-mono text-xs text-muted-foreground w-44 text-left">{new Date(a.timestamp).toLocaleString()}</span>
              <span className="font-semibold text-sm" style={{ color: severityHex(a.severity) }}>{a.severity}</span>
              <span className="text-sm font-mono">{(a.probability * 100).toFixed(1)}%</span>
              <span className="text-sm text-foreground flex-1 text-left truncate">{a.root_cause}</span>
              <span className="text-sm text-success font-semibold">${formatNum(a.cost_saved)}</span>
              <ChevronDown className={cn("w-4 h-4 transition-transform", expanded === a.id && "rotate-180")} />
            </button>
            {expanded === a.id && (
              <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} className="border-t border-border bg-background/40 px-4 py-3">
                <div className="text-xs uppercase tracking-wider text-muted-foreground mb-2">Recommended Actions</div>
                <ul className="space-y-1 text-sm">
                  {a.actions.map((act, j) => <li key={j} className="flex gap-2"><span className="text-primary">•</span>{act}</li>)}
                </ul>
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
};

export default AgentLog;
