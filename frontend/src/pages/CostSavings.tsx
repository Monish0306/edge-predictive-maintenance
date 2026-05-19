import { motion } from "framer-motion";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Cell } from "recharts";
import { Cloud, Cpu, Leaf, Zap } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { DarkTooltip } from "@/components/ChartTooltip";
import { formatNum, severityHex } from "@/lib/api";

const savings = [
  { severity: "LOW", cost: 18000 },
  { severity: "MEDIUM", cost: 65000 },
  { severity: "HIGH", cost: 180000 },
  { severity: "CRITICAL", cost: 420000 },
];

const CostSavings = () => (
  <div>
    <PageHeader title="Cost & Power Savings Analysis" subtitle="Edge AI vs traditional cloud-based monitoring" />

    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
      <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
        className="rounded-xl border border-danger/30 bg-danger/5 p-6 relative overflow-hidden">
        <Cloud className="w-12 h-12 text-danger/30 absolute top-4 right-4" />
        <h3 className="font-bold text-lg text-danger mb-4">☁️ Cloud System — Old Way</h3>
        <ul className="space-y-2 text-sm">
          {["High latency (200-500ms round-trip)", "Constant internet dependency", "$2,500/month per engine cloud cost", "Privacy/IP concerns sending sensor data", "250W+ GPU power per inference node", "Single point of failure"].map((p) => (
            <li key={p} className="flex gap-2"><span className="text-danger">✗</span>{p}</li>
          ))}
        </ul>
      </motion.div>
      <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }}
        className="rounded-xl border border-success/30 bg-success/5 p-6 relative overflow-hidden">
        <Leaf className="w-12 h-12 text-success/30 absolute top-4 right-4" />
        <h3 className="font-bold text-lg text-success mb-4">⚡ Our Edge AI — New Way</h3>
        <ul className="space-y-2 text-sm">
          {["Sub-millisecond inference (0.20ms)", "Runs offline at the edge", "$0/month cloud cost after deploy", "Sensor data never leaves the floor", "5-15W on Raspberry Pi / Jetson Nano", "Distributed — no single failure point"].map((p) => (
            <li key={p} className="flex gap-2"><span className="text-success">✓</span>{p}</li>
          ))}
        </ul>
      </motion.div>
    </div>

    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}
      className="rounded-xl border border-border bg-card p-5 mb-6 overflow-x-auto">
      <h3 className="font-semibold mb-4">Financial Impact per Caught Failure</h3>
      <table className="w-full text-sm">
        <thead className="text-xs uppercase text-muted-foreground border-b border-border">
          <tr><th className="text-left py-2 px-2">Severity</th><th className="text-right">Cost Saved</th><th className="text-right">Downtime Avoided</th><th className="text-right px-2">Action Window</th></tr>
        </thead>
        <tbody>
          {[
            ["LOW", 18000, "2 hrs", "30 days"],
            ["MEDIUM", 65000, "8 hrs", "10 days"],
            ["HIGH", 180000, "24 hrs", "3 days"],
            ["CRITICAL", 420000, "72 hrs", "24 hrs"],
          ].map(([s, c, d, w]: any) => (
            <tr key={s} className="border-b border-border/50 last:border-0">
              <td className="py-3 px-2 font-bold" style={{ color: severityHex(s) }}>{s}</td>
              <td className="text-right text-success font-semibold">${formatNum(c)}</td>
              <td className="text-right">{d}</td>
              <td className="text-right px-2">{w}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </motion.div>

    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}
      className="rounded-xl border border-border bg-card p-5 mb-6">
      <h3 className="font-semibold mb-4">Cost Saved by Severity</h3>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={savings}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis dataKey="severity" stroke="hsl(var(--muted-foreground))" fontSize={11} />
          <YAxis stroke="hsl(var(--muted-foreground))" fontSize={11} tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
          <Tooltip content={<DarkTooltip />} cursor={{ fill: "hsl(var(--muted) / 0.3)" }} />
          <Bar dataKey="cost" radius={[8, 8, 0, 0]} animationDuration={1200}>
            {savings.map((s, i) => <Cell key={i} fill={severityHex(s.severity as any)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </motion.div>

    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
      <MetricCard label="Cloud GPU Power" value={250} suffix=" W" icon={Cloud} color="danger" />
      <MetricCard label="Edge Device Power" value={10} suffix=" W" icon={Cpu} color="success" delay={0.1} hint="5-15W typical range" />
      <MetricCard label="Power Savings" value={95} suffix="%" icon={Zap} color="warning" delay={0.2} />
    </div>

    <motion.pre initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}
      className="rounded-xl border border-border bg-background/60 p-6 font-mono text-xs whitespace-pre overflow-x-auto">
{`╔══════════════════════════════════════════════════════════╗
║  EXECUTIVE SUMMARY — EDGE AI PREDICTIVE MAINTENANCE       ║
╠══════════════════════════════════════════════════════════╣
║  • $683,000 average savings per critical failure caught   ║
║  • 95% reduction in inference power (250W → 10W)          ║
║  • 0.20ms inference latency vs 350ms cloud                ║
║  • $30,000/year saved per engine in cloud fees            ║
║  • Zero data leaves the factory floor                     ║
║  • ROI: 4.2 months for 50-engine fleet                    ║
╚══════════════════════════════════════════════════════════╝`}
    </motion.pre>
  </div>
);

export default CostSavings;
