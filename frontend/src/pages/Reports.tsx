import { useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, Download, FileText, Sparkles } from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { MetricCard } from "@/components/MetricCard";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Slider } from "@/components/ui/slider";
import { formatDate, formatNum } from "@/lib/api";
import { toast } from "sonner";

const Reports = () => {
  const [engineId, setEngineId] = useState("E-001");
  const [prob, setProb] = useState(0.72);
  const [rul, setRul] = useState(35);
  const [report, setReport] = useState<string | null>(null);

  const generate = () => {
    const sev = prob > 0.85 ? "CRITICAL" : prob > 0.65 ? "HIGH" : prob > 0.45 ? "MEDIUM" : "LOW";
    const days = Math.round(rul / 2);
    const cost = Math.round(rul < 30 ? 180000 : rul < 60 ? 95000 : 32000);
    const ignored = cost * 5;
    const text = `================================================================
  EDGE AI — MAINTENANCE REPORT
================================================================
Engine ID:           ${engineId}
Report Date:         ${formatDate(new Date())}
Severity:            ${sev}
Anomaly Probability: ${(prob * 100).toFixed(1)}%
RUL (cycles):        ${rul}
Days Until Action:   ${days}

----------------------------------------------------------------
  PLAIN-ENGLISH SUMMARY
----------------------------------------------------------------
The AI model predicts engine ${engineId} will require
maintenance within the next ${days} days. The high-pressure
compressor temperature sensors (T-30, T-50) are showing
patterns consistent with bearing wear.

----------------------------------------------------------------
  RECOMMENDED ACTIONS
----------------------------------------------------------------
1. Schedule inspection within ${Math.max(1, Math.round(days / 2))} days.
2. Order spare HPC bearing assembly (Part #HPC-2847-B).
3. Notify shift supervisor and update CMMS.
4. Reduce engine load by 15% until inspection.

----------------------------------------------------------------
  FINANCIAL IMPACT
----------------------------------------------------------------
Repair cost (planned):     $${formatNum(cost)}
Cost if ignored:           $${formatNum(ignored)}
Estimated savings:         $${formatNum(ignored - cost)}

----------------------------------------------------------------
  PREPARED BY: Edge AI Predictive Maintenance Agent v2.1
================================================================`;
    setReport(text);
    toast.success("Report ready! Factory workers can read and act on this immediately.");
  };

  const download = () => {
    if (!report) return;
    const blob = new Blob([report], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = `${engineId}-report-${formatDate(new Date())}.txt`; a.click();
    URL.revokeObjectURL(url);
  };

  const sev = prob > 0.85 ? "CRITICAL" : prob > 0.65 ? "HIGH" : prob > 0.45 ? "MEDIUM" : "LOW";
  const days = Math.round(rul / 2);
  const cost = Math.round(rul < 30 ? 180000 : rul < 60 ? 95000 : 32000);

  return (
    <div>
      <PageHeader title="Auto-Generated Maintenance Report" subtitle="Professional plain-English report for factory floor workers" />

      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-border bg-card p-5 mb-6 grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
        <div className="space-y-1">
          <label className="text-xs uppercase text-muted-foreground tracking-wider">Engine ID</label>
          <Input value={engineId} onChange={(e) => setEngineId(e.target.value)} />
        </div>
        <div className="space-y-2">
          <label className="text-xs uppercase text-muted-foreground tracking-wider">Anomaly Probability: {prob.toFixed(2)}</label>
          <Slider value={[prob * 100]} max={100} onValueChange={(v) => setProb(v[0] / 100)} />
        </div>
        <div className="space-y-2">
          <label className="text-xs uppercase text-muted-foreground tracking-wider">RUL: {rul} cycles</label>
          <Slider value={[rul]} max={125} onValueChange={(v) => setRul(v[0])} />
        </div>
        <Button onClick={generate} className="gap-2 md:col-span-3 md:w-fit"><Sparkles className="w-4 h-4" /> Generate Report</Button>
      </motion.div>

      {report && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <MetricCard label="Severity" value={sev} icon={FileText} color={sev === "CRITICAL" ? "critical" : sev === "HIGH" ? "danger" : sev === "MEDIUM" ? "warning" : "success"} />
            <MetricCard label="Days Until Action" value={days} delay={0.1} color="primary" />
            <MetricCard label="Repair Cost" value={cost} prefix="$" delay={0.2} color="warning" />
            <MetricCard label="Cost if Ignored" value={cost * 5} prefix="$" delay={0.3} color="danger" />
          </div>

          <div className="flex items-center gap-2 mb-3 text-success">
            <CheckCircle2 className="w-5 h-5" />
            <span className="text-sm font-medium">Report ready! Factory workers can read and act on this immediately.</span>
          </div>

          <motion.pre initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border border-border bg-background/60 p-6 font-mono text-xs overflow-x-auto whitespace-pre text-foreground">
{report}
          </motion.pre>

          <Button onClick={download} className="mt-4 gap-2"><Download className="w-4 h-4" /> Download as .txt</Button>
        </>
      )}
    </div>
  );
};

export default Reports;
