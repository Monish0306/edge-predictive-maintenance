import { useEffect, useRef, useState } from "react";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, Bell, Play, RotateCcw, Shield, Square, Wrench, X } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DarkTooltip } from "@/components/ChartTooltip";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { simulateReading, healthGrade, severityEmoji, SimulateMode, PredictionResult } from "@/lib/api";
import { alertStore } from "@/lib/alertStore";

const MAX_POINTS = 40;

const LiveMonitor = () => {
  const [mode, setMode]           = useState<SimulateMode>("normal");
  const [running, setRunning]     = useState(true);
  const [data, setData]           = useState<PredictionResult | null>(null);
  const [history, setHistory]     = useState<{ t: string; prob: number; health: number }[]>([]);
  const [alertCount, setAlertCount] = useState(0);
  const [showAlert, setShowAlert] = useState(true);
  const intervalRef               = useRef<number | null>(null);

  const tick = async () => {
    try {
      const res = await simulateReading(mode.toLowerCase() as SimulateMode, 1);
      setData(res);

      setHistory((h) => [
        ...h,
        {
          t: new Date().toLocaleTimeString().slice(3, 8),
          prob: res.anomaly_probability,
          health: res.health_score,
        },
      ].slice(-MAX_POINTS));

      // ── ADD TO GLOBAL ALERT STORE ──────────────────────
      if (res.severity !== "NORMAL") {
        setAlertCount((c) => c + 1);
        setShowAlert(true);

        alertStore.addAlert({
          engine_id:           res.engine_id || 1,
          severity:            res.severity,
          anomaly_probability: res.anomaly_probability,
          health_score:        res.health_score,
          root_cause:          res.root_cause || "Anomaly detected",
          message:             `Engine #${res.engine_id} — ${res.severity} alert`,
        });
      }
    } catch (err) {
      console.error("Tick error:", err);
    }
  };

  useEffect(() => {
    if (!running) {
      if (intervalRef.current) window.clearInterval(intervalRef.current);
      return;
    }
    tick();
    intervalRef.current = window.setInterval(tick, 1000);
    return () => { if (intervalRef.current) window.clearInterval(intervalRef.current); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [running, mode]);

  const clear = () => {
    setHistory([]);
    setAlertCount(0);
    setData(null);
    alertStore.clearAll();
  };

  return (
    <div>
      <PageHeader
        title="Live Sensor Monitoring"
        subtitle="Real-time anomaly detection on NASA Turbofan engine #1"
        live
      >
        <Select value={mode} onValueChange={(v) => setMode(v as SimulateMode)}>
          <SelectTrigger className="w-36 bg-card"><SelectValue /></SelectTrigger>
          <SelectContent>
            <SelectItem value="Normal">🟢 Normal</SelectItem>
            <SelectItem value="Warning">🟡 Warning</SelectItem>
            <SelectItem value="Fault">🔴 Fault</SelectItem>
          </SelectContent>
        </Select>

        <Button
          variant={running ? "destructive" : "default"}
          onClick={() => setRunning(!running)}
          className="gap-2"
        >
          {running
            ? <><Square className="w-4 h-4" /> Stop</>
            : <><Play  className="w-4 h-4" /> Start</>
          }
        </Button>

        <Button variant="outline" onClick={clear} className="gap-2">
          <RotateCcw className="w-4 h-4" /> Clear
        </Button>
      </PageHeader>

      {/* ── METRIC CARDS ──────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard
          label="Anomaly Probability"
          value={data?.anomaly_probability ?? 0}
          decimals={3}
          icon={AlertTriangle}
          color="warning"
          delay={0}
        />
        <MetricCard
          label="Health Score"
          value={data?.health_score ?? 0}
          decimals={1}
          icon={Shield}
          color="primary"
          delay={0.1}
          hint={`Grade ${healthGrade(data?.health_score ?? 0)}`}
        />
        <MetricCard
          label="Status"
          value={data ? `${severityEmoji(data.severity)} ${data.severity}` : "—"}
          icon={Wrench}
          color={
            data?.severity === "CRITICAL" ? "critical" :
            data?.severity === "HIGH"     ? "danger"   :
            data?.severity === "MEDIUM"   ? "warning"  : "success"
          }
          delay={0.2}
        />
        <MetricCard
          label="Total Alerts"
          value={alertCount}
          icon={Bell}
          color="danger"
          delay={0.3}
        />
      </div>

      {/* ── ALERT BANNER ──────────────────────────────────── */}
      <AnimatePresence>
        {data && (data.severity === "HIGH" || data.severity === "CRITICAL") && showAlert && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -10 }}
            className="mb-6 rounded-xl border border-danger/40 bg-danger/10 p-4 flex items-center gap-4"
            style={{ boxShadow: "0 0 20px rgba(239,68,68,0.15)" }}
          >
            <div className="w-10 h-10 rounded-lg bg-danger/20 flex items-center justify-center shrink-0">
              <AlertTriangle className="w-5 h-5 text-danger" />
            </div>
            <div className="flex-1">
              <div className="font-semibold text-danger">
                {data.severity} ALERT — Engine E-001
              </div>
              <div className="text-sm text-muted-foreground">
                Anomaly probability {(data.anomaly_probability * 100).toFixed(1)}%
                — {data.root_cause || "Immediate action recommended"}
              </div>
            </div>
            <button
              onClick={() => setShowAlert(false)}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="w-5 h-5" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── CHARTS ────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ChartCard title="Real-time Anomaly Probability">
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="probGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#3B82F6" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="t" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis domain={[0, 1]} stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip content={<DarkTooltip />} />
              <ReferenceLine
                y={0.5}
                stroke="#EF4444"
                strokeDasharray="4 4"
                label={{ value: "Threshold", fill: "#EF4444", fontSize: 10, position: "right" }}
              />
              <Area
                type="monotone"
                dataKey="prob"
                name="Probability"
                stroke="#3B82F6"
                strokeWidth={2}
                fill="url(#probGrad)"
                isAnimationActive
                animationDuration={400}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Engine Health Score">
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={history}>
              <defs>
                <linearGradient id="healthGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%"   stopColor="#22C55E" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="#22C55E" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="t" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis domain={[0, 100]} stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip content={<DarkTooltip />} />
              <Area
                type="monotone"
                dataKey="health"
                name="Health"
                stroke="#22C55E"
                strokeWidth={2}
                fill="url(#healthGrad)"
                isAnimationActive
                animationDuration={400}
              />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* ── AGENT RECOMMENDATION ──────────────────────────── */}
      {data && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="rounded-xl border border-border bg-card p-6"
        >
          <div className="flex items-center gap-2 mb-4">
            <Wrench className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Agent Recommendation</h3>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-5">
            <RecBox label="Next Maintenance" value={data.maintenance_schedule || "—"} />
            <RecBox label="Est. Downtime"    value={data.estimated_downtime    || "—"} />
            <RecBox label="Cost Saved"       value={data.cost_saved            || "—"} accent />
          </div>

          <ol className="space-y-2">
            {data.recommended_actions?.map((a, i) => (
              <motion.li
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.05 }}
                className="flex items-start gap-3 text-sm"
              >
                <span className="w-6 h-6 rounded-full bg-primary/15 text-primary text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">
                  {i + 1}
                </span>
                <span className="text-foreground">{a}</span>
              </motion.li>
            ))}
          </ol>
        </motion.div>
      )}
    </div>
  );
};

// ── SUB-COMPONENTS ─────────────────────────────────────────
const ChartCard = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.5 }}
    className="rounded-xl border border-border bg-card p-5"
  >
    <h3 className="font-semibold mb-4 text-sm">{title}</h3>
    {children}
  </motion.div>
);

const RecBox = ({ label, value, accent }: { label: string; value: string; accent?: boolean }) => (
  <div className="rounded-lg border border-border bg-background/50 p-4">
    <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1">{label}</div>
    <div className={`text-lg font-bold ${accent ? "text-success" : "text-foreground"}`}>{value}</div>
  </div>
);

export default LiveMonitor;