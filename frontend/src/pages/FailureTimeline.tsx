import { useState } from "react";
import { motion } from "framer-motion";
import { Area, CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, Calendar, Clock, Sparkles, Wrench } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DarkTooltip } from "@/components/ChartTooltip";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { formatDate } from "@/lib/api";

const FailureTimeline = () => {
  const [engineId, setEngineId] = useState("1");
  const [rul, setRul] = useState(45);
  const [prob, setProb] = useState(0.65);
  const [cyclesPerDay, setCyclesPerDay] = useState(2);
  const [generated, setGenerated] = useState(false);

  const days = Math.round(rul / cyclesPerDay);
  const failureDate = new Date(Date.now() + days * 86400000);
  const actBefore = new Date(Date.now() + Math.round(days * 0.7) * 86400000);
  const urgency = prob > 0.8 ? "danger" : prob > 0.5 ? "warning" : "success";
  const urgencyText = prob > 0.8 ? "URGENT — Act within 7 days" : prob > 0.5 ? "WARNING — Plan maintenance" : "SAFE — Routine monitoring";

  const data = Array.from({ length: 30 }, (_, i) => {
    const pct = (i / 29) * 100;
    const deg = Math.min(100, prob * 100 * (i / 29) * 1.5 + i * 1.2);
    return { day: i, degradation: +deg.toFixed(1), upper: Math.min(100, deg + 8), lower: Math.max(0, deg - 8) };
  });

  return (
    <div>
      <PageHeader title="Failure Prediction Timeline" subtitle="RUL cycles converted to calendar dates with confidence zones" />

      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-border bg-card p-5 mb-6 grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        <div className="space-y-1">
          <label className="text-xs uppercase text-muted-foreground tracking-wider">Engine ID</label>
          <Input value={engineId} onChange={(e) => setEngineId(e.target.value)} />
        </div>
        <SliderField label={`RUL: ${rul} cycles`} value={rul} max={125} onChange={setRul} />
        <SliderField label={`Anomaly: ${prob.toFixed(2)}`} value={prob * 100} max={100} onChange={(v) => setProb(v / 100)} />
        <SliderField label={`Cycles/day: ${cyclesPerDay}`} value={cyclesPerDay} max={10} min={1} onChange={setCyclesPerDay} />
        <Button onClick={() => setGenerated(true)} className="gap-2 md:col-span-4 md:w-auto"><Sparkles className="w-4 h-4" /> Generate Timeline</Button>
      </motion.div>

      {generated && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard label="RUL Cycles" value={rul} icon={Clock} color="primary" />
            <MetricCard label="RUL Days" value={days} icon={Calendar} color="warning" delay={0.1} />
            <MetricCard label="Predicted Failure" value={formatDate(failureDate)} icon={AlertTriangle} color="danger" delay={0.2} />
            <MetricCard label="Act Before" value={formatDate(actBefore)} icon={Wrench} color="success" delay={0.3} />
          </div>

          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
            className={`rounded-xl border p-4 mb-6 ${urgency === "danger" ? "border-danger/40 bg-danger/10" : urgency === "warning" ? "border-warning/40 bg-warning/10" : "border-success/40 bg-success/10"}`}>
            <div className={`font-semibold ${urgency === "danger" ? "text-danger" : urgency === "warning" ? "text-warning" : "text-success"}`}>{urgencyText}</div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}
            className="rounded-xl border border-border bg-card p-5 mb-6">
            <h3 className="font-semibold mb-4">Degradation Forecast</h3>
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart data={data}>
                <defs>
                  <linearGradient id="confGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#EF4444" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#EF4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="day" stroke="hsl(var(--muted-foreground))" fontSize={11} label={{ value: "Days from today", fill: "hsl(var(--muted-foreground))", fontSize: 11, position: "insideBottom", offset: -5 }} />
                <YAxis domain={[0, 100]} stroke="hsl(var(--muted-foreground))" fontSize={11} />
                <Tooltip content={<DarkTooltip />} />
                <ReferenceArea y1={0} y2={50} fill="#22C55E" fillOpacity={0.08} label={{ value: "Safe", position: "insideTopLeft", fill: "#22C55E", fontSize: 10 }} />
                <ReferenceArea y1={50} y2={80} fill="#EAB308" fillOpacity={0.08} label={{ value: "Warning", position: "insideTopLeft", fill: "#EAB308", fontSize: 10 }} />
                <ReferenceArea y1={80} y2={100} fill="#EF4444" fillOpacity={0.1} label={{ value: "Danger", position: "insideTopLeft", fill: "#EF4444", fontSize: 10 }} />
                <Area dataKey="upper" stroke="none" fill="url(#confGrad)" />
                <Area dataKey="lower" stroke="none" fill="hsl(var(--background))" />
                <Line type="monotone" dataKey="degradation" stroke="#3B82F6" strokeWidth={3} dot={false} animationDuration={1500} />
                <ReferenceLine x={Math.round(days * 0.4)} stroke="#22C55E" strokeDasharray="4 4" label={{ value: "Inspect", fill: "#22C55E", fontSize: 10 }} />
                <ReferenceLine x={Math.round(days * 0.6)} stroke="#EAB308" strokeDasharray="4 4" label={{ value: "Order Parts", fill: "#EAB308", fontSize: 10 }} />
                <ReferenceLine x={Math.round(days * 0.7)} stroke="#F97316" strokeDasharray="4 4" label={{ value: "Maintain", fill: "#F97316", fontSize: 10 }} />
                <ReferenceLine x={days > 29 ? 29 : days} stroke="#EF4444" strokeDasharray="4 4" label={{ value: "Failure", fill: "#EF4444", fontSize: 10 }} />
              </ComposedChart>
            </ResponsiveContainer>
          </motion.div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}
              className="rounded-xl border border-border bg-card p-5">
              <h3 className="font-semibold mb-4">Action Schedule</h3>
              <table className="w-full text-sm">
                <thead className="text-xs uppercase tracking-wider text-muted-foreground border-b border-border"><tr><th className="text-left py-2">Action</th><th>Date</th><th className="text-right">Days</th></tr></thead>
                <tbody>
                  {[
                    ["Inspect sensors", 0.4, "#22C55E"],
                    ["Order replacement parts", 0.6, "#EAB308"],
                    ["Schedule maintenance", 0.7, "#F97316"],
                    ["Estimated failure", 1.0, "#EF4444"],
                  ].map(([label, mult, color]: any) => {
                    const d = Math.round(days * mult);
                    return (
                      <tr key={label} className="border-b border-border/50 last:border-0">
                        <td className="py-3 font-medium" style={{ color }}>{label}</td>
                        <td className="font-mono">{formatDate(new Date(Date.now() + d * 86400000))}</td>
                        <td className="text-right font-mono">{d}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </motion.div>
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}
              className="rounded-xl border border-border bg-card p-5">
              <h3 className="font-semibold mb-4">Degradation Analysis</h3>
              <dl className="space-y-3 text-sm">
                <Row k="Degradation rate" v={`${(prob * 1.5).toFixed(2)} %/day`} />
                <Row k="Model confidence" v={`${(85 + (1 - prob) * 10).toFixed(1)}%`} />
                <Row k="Failure window" v={`±${Math.max(2, Math.round(days * 0.15))} days`} />
                <Row k="Trend slope" v={prob > 0.5 ? "Accelerating ↑" : "Stable →"} />
              </dl>
            </motion.div>
          </div>
        </>
      )}
    </div>
  );
};

const SliderField = ({ label, value, max, min = 0, onChange }: any) => (
  <div className="space-y-2">
    <label className="text-xs uppercase text-muted-foreground tracking-wider">{label}</label>
    <Slider value={[value]} min={min} max={max} step={max > 10 ? 1 : 1} onValueChange={(v) => onChange(v[0])} />
  </div>
);
const Row = ({ k, v }: { k: string; v: string }) => (
  <div className="flex justify-between border-b border-border/50 pb-2">
    <dt className="text-muted-foreground">{k}</dt><dd className="font-mono font-semibold">{v}</dd>
  </div>
);

export default FailureTimeline;
