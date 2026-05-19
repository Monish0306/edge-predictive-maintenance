import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { Button } from "@/components/ui/button";
import { getFleet, FleetEngine, severityHex } from "@/lib/api";
import { cn } from "@/lib/utils";

const SEVERITIES = ["NORMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"] as const;

const FleetOverview = () => {
  const [engines, setEngines] = useState<FleetEngine[]>([]);
  const [filter, setFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = async () => {
    setLoading(true);
    const data = await getFleet(50);
    setEngines(data.engines || []);
    setLoading(false);
  };
  useEffect(() => { load(); }, []);

  const counts = SEVERITIES.reduce((acc, s) => ({ ...acc, [s]: engines.filter((e) => e.severity === s).length }), {} as Record<string, number>);
  const filtered = filter ? engines.filter((e) => e.severity === filter) : engines;

  return (
    <div>
      <PageHeader title="Fleet Overview" subtitle={`Monitoring ${engines.length} turbofan engines in real-time`}>
        <Button onClick={load} variant="outline" className="gap-2">
          <RefreshCw className={cn("w-4 h-4", loading && "animate-spin")} /> Refresh
        </Button>
      </PageHeader>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
        {SEVERITIES.map((s, i) => (
          <motion.button
            key={s}
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05 }}
            onClick={() => setFilter(filter === s ? null : s)}
            className={cn(
              "rounded-xl border p-4 text-left transition-all hover:scale-[1.02]",
              filter === s ? "border-primary bg-primary/10 shadow-glow" : "border-border bg-card hover:border-primary/40"
            )}
          >
            <div className="text-xs uppercase tracking-wider text-muted-foreground mb-1">{s}</div>
            <div className="text-2xl font-bold" style={{ color: severityHex(s) }}>{counts[s] || 0}</div>
          </motion.button>
        ))}
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {loading
          ? Array.from({ length: 10 }).map((_, i) => <div key={i} className="h-40 rounded-xl skeleton" />)
          : filtered.map((e, i) => (
              <motion.div
                key={e.engine_id}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ delay: Math.min(i * 0.02, 0.4), duration: 0.3 }}
                whileHover={{ scale: 1.05, y: -4 }}
                className={cn(
                  "rounded-xl border bg-card p-4 cursor-pointer relative overflow-hidden",
                  e.severity === "CRITICAL" && "animate-glow-pulse border-critical/50",
                  e.severity === "HIGH" && "border-danger/40"
                )}
                style={{ boxShadow: e.severity === "CRITICAL" ? `0 0 20px ${severityHex(e.severity)}40` : undefined }}
              >
                <div className="flex items-start justify-between mb-3">
                  <span className="font-mono font-bold text-sm">{`E-${String(e.engine_id).padStart(3, "0")}`}</span>
                  {(e.severity === "HIGH" || e.severity === "CRITICAL") && <AlertTriangle className="w-4 h-4" style={{ color: severityHex(e.severity) }} />}
                </div>
                <div className="text-3xl font-bold mb-2" style={{ color: severityHex(e.severity) }}>{e.health_score.toFixed(0)}</div>
                <div className="h-2 rounded-full bg-secondary overflow-hidden mb-3">
                  <motion.div
                    initial={{ width: 0 }} animate={{ width: `${e.health_score}%` }} transition={{ duration: 0.8, delay: 0.2 }}
                    className="h-full rounded-full"
                    style={{ background: e.health_score > 60 ? "#22C55E" : e.health_score > 40 ? "#EAB308" : "#EF4444" }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{e.rul_cycles} cyc</span>
                  <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold" style={{ background: `${severityHex(e.severity)}25`, color: severityHex(e.severity) }}>{e.severity}</span>
                </div>
              </motion.div>
            ))}
      </div>
    </div>
  );
};

export default FleetOverview;
