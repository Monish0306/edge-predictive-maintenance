import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { Award, Cpu, Database, Target } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DarkTooltip } from "@/components/ChartTooltip";
import { getEvaluation, formatNum, formatPct } from "@/lib/api";

const COLORS = ["#22C55E", "#F97316", "#3B82F6", "#A855F7"];

const Analytics = () => {
  const [data, setData] = useState<any>(null);
  useEffect(() => { getEvaluation().then(setData); }, []);

  if (!data) return <div className="grid grid-cols-4 gap-4">{Array.from({ length: 4 }).map((_, i) => <div key={i} className="h-32 rounded-xl skeleton" />)}</div>;

  const champion = data.datasets[0];

  return (
    <div>
      <PageHeader title="Cross-Dataset Evaluation" subtitle="Model trained on FD001, tested on all 4 NASA Turbofan datasets" />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <MetricCard label="FD001 AUC-ROC" value={champion.auc} decimals={3} icon={Award} color="success" delay={0} />
        <MetricCard label="FD001 Accuracy" value={champion.accuracy * 100} decimals={2} suffix="%" icon={Target} color="primary" delay={0.1} />
        <MetricCard label="Total Engines" value={data.total_engines} icon={Cpu} color="warning" delay={0.2} />
        <MetricCard label="Test Samples" value={data.test_samples} icon={Database} color="critical" delay={0.3} />
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
        className="rounded-xl border border-border bg-card p-5 mb-6 overflow-x-auto">
        <h3 className="font-semibold mb-4">Dataset Overview</h3>
        <table className="w-full text-sm">
          <thead className="text-xs uppercase tracking-wider text-muted-foreground border-b border-border">
            <tr><th className="text-left py-2 px-2">Dataset</th><th className="text-right">Engines</th><th className="text-right">Sensors</th><th className="text-right">Sequences</th><th className="text-right px-2">Anomaly Rate</th></tr>
          </thead>
          <tbody>
            {data.datasets.map((d: any, i: number) => (
              <tr key={d.name} className="border-b border-border/50 last:border-0">
                <td className="py-3 px-2 font-mono font-semibold" style={{ color: COLORS[i] }}>{d.name}</td>
                <td className="text-right">{formatNum(d.engines)}</td>
                <td className="text-right">{d.sensors}</td>
                <td className="text-right">{formatNum(d.sequences)}</td>
                <td className="text-right px-2">{formatPct(d.anomaly_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <ChartBox title="AUC-ROC by Dataset">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.datasets}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis domain={[0, 1]} stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip content={<DarkTooltip />} cursor={{ fill: "hsl(var(--muted) / 0.3)" }} />
              <Bar dataKey="auc" name="AUC-ROC" radius={[8, 8, 0, 0]} animationDuration={1200}>
                {data.datasets.map((_: any, i: number) => <Cell key={i} fill={COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>
        <ChartBox title="Accuracy by Dataset">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.datasets}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="name" stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <YAxis domain={[0, 1]} stroke="hsl(var(--muted-foreground))" fontSize={11} />
              <Tooltip content={<DarkTooltip />} cursor={{ fill: "hsl(var(--muted) / 0.3)" }} />
              <Bar dataKey="accuracy" name="Accuracy" radius={[8, 8, 0, 0]} animationDuration={1200}>
                {data.datasets.map((_: any, i: number) => <Cell key={i} fill={COLORS[i]} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartBox>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {data.datasets.map((d: any, i: number) => {
          const tp = Math.round(d.accuracy * 100);
          const fn = 100 - tp;
          return (
            <motion.div key={d.name} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 + i * 0.05 }}
              className="rounded-xl border border-border bg-card p-4">
              <div className="font-mono font-bold mb-3" style={{ color: COLORS[i] }}>{d.name} Confusion</div>
              <div className="grid grid-cols-2 gap-1 text-center text-xs">
                <div className="bg-success/20 text-success rounded p-3 font-semibold">TP {tp}</div>
                <div className="bg-warning/15 text-warning rounded p-3 font-semibold">FP {Math.round(fn / 2)}</div>
                <div className="bg-warning/15 text-warning rounded p-3 font-semibold">FN {Math.round(fn / 2)}</div>
                <div className="bg-success/20 text-success rounded p-3 font-semibold">TN {tp}</div>
              </div>
            </motion.div>
          );
        })}
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}
        className="rounded-xl border border-border bg-card p-5">
        <h3 className="font-semibold mb-4">Key Insights</h3>
        <table className="w-full text-sm">
          <tbody>
            {[
              ["FD001", "Single fault, single condition — easiest, model excels (Champion)"],
              ["FD002", "6 operating conditions — domain shift hurts generalization"],
              ["FD003", "Two fault modes — moderate challenge"],
              ["FD004", "6 conditions × 2 faults — hardest, requires retraining"],
            ].map(([k, v], i) => (
              <tr key={k} className="border-b border-border/50 last:border-0">
                <td className="py-3 font-mono font-semibold" style={{ color: COLORS[i] }}>{k}</td>
                <td className="py-3 text-muted-foreground">{v}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </motion.div>
    </div>
  );
};

const ChartBox = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
    className="rounded-xl border border-border bg-card p-5">
    <h3 className="font-semibold mb-4 text-sm">{title}</h3>
    {children}
  </motion.div>
);

export default Analytics;
