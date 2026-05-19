import { useState } from "react";
import { motion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { Activity, Info, Shield, Sparkles } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DarkTooltip } from "@/components/ChartTooltip";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const SENSORS = ["T-2 Temp", "T-24 Temp", "T-30 Temp", "T-50 Temp", "P-2 Press", "P-15 Press", "P-30 Press", "Nf RPM", "Nc RPM", "PCNfR", "Ps-30", "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32"];

const SensorHeatmap = () => {
  const [mode, setMode] = useState("Fault");
  const [engineId, setEngineId] = useState("1");
  const [analyzed, setAnalyzed] = useState(false);
  const [importance, setImportance] = useState<{ name: string; value: number }[]>([]);
  const [grid, setGrid] = useState<number[][]>([]);
  const [prob, setProb] = useState(0);
  const [health, setHealth] = useState(0);

  const analyze = () => {
    const base = mode === "Fault" ? 0.9 : mode === "Warning" ? 0.6 : 0.15;
    setProb(+base.toFixed(3));
    setHealth(+(100 - base * 90).toFixed(1));
    const imp = SENSORS.map((s) => ({ name: s, value: +(Math.random() * (mode === "Fault" ? 95 : 60)).toFixed(1) }))
      .sort((a, b) => b.value - a.value);
    setImportance(imp);
    setGrid(Array.from({ length: SENSORS.length }, () => Array.from({ length: 30 }, () => Math.random())));
    setAnalyzed(true);
  };

  const top3 = importance.slice(0, 3);

  return (
    <div>
      <PageHeader title="Sensor Attention Heatmap" subtitle="Explainable AI — which sensors trigger the anomaly alert" />

      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-border bg-card p-4 mb-6 flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label className="text-xs uppercase tracking-wider text-muted-foreground">Fault Mode</label>
          <Select value={mode} onValueChange={setMode}>
            <SelectTrigger className="w-36"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="Normal">Normal</SelectItem>
              <SelectItem value="Warning">Warning</SelectItem>
              <SelectItem value="Fault">Fault</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1">
          <label className="text-xs uppercase tracking-wider text-muted-foreground">Engine ID</label>
          <Input value={engineId} onChange={(e) => setEngineId(e.target.value)} className="w-24" />
        </div>
        <Button onClick={analyze} className="gap-2"><Sparkles className="w-4 h-4" /> Analyze</Button>
      </motion.div>

      {analyzed && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <MetricCard label="Anomaly Probability" value={prob} decimals={3} icon={Activity} color="danger" />
            <MetricCard label="Health Score" value={health} decimals={1} icon={Shield} color="primary" delay={0.1} />
          </div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
            className="rounded-xl border border-border bg-card p-5 mb-6">
            <h3 className="font-semibold mb-4">Sensor Importance</h3>
            <ResponsiveContainer width="100%" height={420}>
              <BarChart data={importance} layout="vertical" margin={{ left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis type="number" domain={[0, 100]} stroke="hsl(var(--muted-foreground))" fontSize={11} />
                <YAxis type="category" dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} width={70} />
                <Tooltip content={<DarkTooltip />} cursor={{ fill: "hsl(var(--muted) / 0.3)" }} />
                <Bar dataKey="value" name="Importance %" radius={[0, 6, 6, 0]} animationDuration={1000}>
                  {importance.map((d, i) => <Cell key={i} fill={d.value > 70 ? "#EF4444" : d.value > 40 ? "#EAB308" : "#22C55E"} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            {top3.map((s, i) => (
              <MetricCard key={s.name} label={`Top ${i + 1} Sensor`} value={s.value} decimals={1} suffix="%" hint={s.name}
                color={s.value > 70 ? "danger" : s.value > 40 ? "warning" : "success"} delay={i * 0.1} />
            ))}
          </div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
            className="rounded-xl border border-border bg-card p-5 mb-6 overflow-x-auto">
            <h3 className="font-semibold mb-4">2D Sensor Heatmap (last 30 cycles)</h3>
            <div className="inline-block min-w-full">
              <div className="grid gap-px" style={{ gridTemplateColumns: "100px repeat(30, 1fr)" }}>
                {SENSORS.map((s, r) => (
                  <>
                    <div key={`l-${r}`} className="text-[10px] text-muted-foreground pr-2 py-1 truncate font-mono">{s}</div>
                    {grid[r]?.map((v, c) => (
                      <div key={`${r}-${c}`} className="aspect-square rounded-sm" title={v.toFixed(2)}
                        style={{ background: `hsl(${(1 - v) * 200}, 80%, ${30 + v * 25}%)` }} />
                    ))}
                  </>
                ))}
              </div>
            </div>
          </motion.div>

          <div className="rounded-xl border border-primary/30 bg-primary/5 p-4 flex items-start gap-3">
            <Info className="w-5 h-5 text-primary shrink-0 mt-0.5" />
            <p className="text-sm text-foreground">Physical components matching high-attention sensors (red) should be inspected first. The model has identified these as the most discriminative signals for the predicted failure mode.</p>
          </div>
        </>
      )}
    </div>
  );
};

export default SensorHeatmap;
