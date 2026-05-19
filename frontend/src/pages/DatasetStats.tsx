import { motion } from "framer-motion";
import { Trophy } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { formatNum } from "@/lib/api";

const datasets = [
  { name: "FD001", color: "success", hex: "#22C55E", engines: 100, conditions: 1, faults: 1, train: 17731, test: 13096, badge: "Champion" },
  { name: "FD002", color: "primary", hex: "#3B82F6", engines: 260, conditions: 6, faults: 1, train: 48819, test: 33991 },
  { name: "FD003", color: "warning", hex: "#EAB308", engines: 100, conditions: 1, faults: 2, train: 21820, test: 16596 },
  { name: "FD004", color: "danger", hex: "#EF4444", engines: 249, conditions: 6, faults: 2, train: 49991, test: 41214 },
];

const DatasetStats = () => {
  const totalEngines = datasets.reduce((s, d) => s + d.engines, 0);
  const totalTrain = datasets.reduce((s, d) => s + d.train, 0);

  return (
    <div>
      <PageHeader title="NASA Turbofan Dataset Statistics" subtitle={`${formatNum(totalEngines)} engines · ${formatNum(totalTrain)} training sequences`} />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {datasets.map((d, i) => (
          <motion.div key={d.name} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.1 }}
            whileHover={{ y: -6 }}
            className="rounded-xl border bg-card p-5 relative overflow-hidden"
            style={{ borderColor: `${d.hex}55`, boxShadow: `0 0 30px ${d.hex}15` }}>
            <div className="absolute inset-0 opacity-10" style={{ background: `radial-gradient(circle at top right, ${d.hex}, transparent 60%)` }} />
            <div className="relative">
              <div className="flex items-center justify-between mb-3">
                <span className="font-mono font-bold text-xl" style={{ color: d.hex }}>{d.name}</span>
                {d.badge && <span className="flex items-center gap-1 text-[10px] uppercase font-bold px-2 py-1 rounded-full" style={{ background: `${d.hex}20`, color: d.hex }}><Trophy className="w-3 h-3" />{d.badge}</span>}
              </div>
              <dl className="space-y-2 text-sm">
                <Row k="Engines" v={d.engines.toString()} />
                <Row k="Conditions" v={d.conditions.toString()} />
                <Row k="Fault Modes" v={d.faults.toString()} />
                <Row k="Train Seqs" v={formatNum(d.train)} />
                <Row k="Test Seqs" v={formatNum(d.test)} />
              </dl>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}
        className="rounded-xl border border-border bg-card p-5 overflow-x-auto">
        <h3 className="font-semibold mb-4">Dataset Details</h3>
        <table className="w-full text-sm">
          <thead className="text-xs uppercase text-muted-foreground border-b border-border">
            <tr><th className="text-left py-2 px-2">Dataset</th><th className="text-right">Engines</th><th className="text-right">Conditions</th><th className="text-right">Faults</th><th className="text-right">Train</th><th className="text-right px-2">Test</th></tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.name} className="border-b border-border/50 last:border-0">
                <td className="py-3 px-2 font-mono font-bold" style={{ color: d.hex }}>{d.name}</td>
                <td className="text-right">{d.engines}</td>
                <td className="text-right">{d.conditions}</td>
                <td className="text-right">{d.faults}</td>
                <td className="text-right">{formatNum(d.train)}</td>
                <td className="text-right px-2">{formatNum(d.test)}</td>
              </tr>
            ))}
            <tr className="font-bold">
              <td className="py-3 px-2">TOTAL</td>
              <td className="text-right">{formatNum(totalEngines)}</td>
              <td colSpan={2}></td>
              <td className="text-right">{formatNum(totalTrain)}</td>
              <td className="text-right px-2">{formatNum(datasets.reduce((s, d) => s + d.test, 0))}</td>
            </tr>
          </tbody>
        </table>
      </motion.div>
    </div>
  );
};

const Row = ({ k, v }: { k: string; v: string }) => (
  <div className="flex justify-between"><dt className="text-muted-foreground text-xs">{k}</dt><dd className="font-mono font-semibold">{v}</dd></div>
);

export default DatasetStats;
