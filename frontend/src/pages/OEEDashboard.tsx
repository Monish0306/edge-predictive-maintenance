import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  RadialBarChart, RadialBar, ResponsiveContainer,
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, BarChart, Bar, Cell,
  AreaChart, Area, ReferenceLine
} from "recharts";
import {
  Activity, TrendingUp, TrendingDown, AlertTriangle,
  CheckCircle, Clock, Zap, Target, Award, BarChart3,
  RefreshCw, Info
} from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import { simulateReading } from "@/lib/api";

// ── TYPES ───────────────────────────────────────────────
interface OEEData {
  availability: number;
  performance: number;
  quality: number;
  oee: number;
  timestamp: string;
}

interface WeeklyData {
  day: string;
  oee: number;
  availability: number;
  performance: number;
  quality: number;
  target: number;
}

interface LossData {
  category: string;
  loss: number;
  color: string;
  icon: string;
  description: string;
}

// ── CONSTANTS ────────────────────────────────────────────
const WORLD_CLASS_OEE = 85;
const INDUSTRY_AVG = 60;

const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

// ── GAUGE COMPONENT ──────────────────────────────────────
function OEEGauge({ value, label, color }: {
  value: number; label: string; color: string
}) {
  const circumference = 2 * Math.PI * 54;
  const strokeDash = (value / 100) * circumference;

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-36 h-36">
        <svg viewBox="0 0 120 120" className="w-full h-full -rotate-90">
          {/* Background track */}
          <circle
            cx="60" cy="60" r="54"
            fill="none"
            stroke="#1F2937"
            strokeWidth="12"
          />
          {/* Value arc */}
          <motion.circle
            cx="60" cy="60" r="54"
            fill="none"
            stroke={color}
            strokeWidth="12"
            strokeLinecap="round"
            strokeDasharray={`${circumference}`}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: circumference - strokeDash }}
            transition={{ duration: 1.5, ease: "easeOut" }}
            style={{ filter: `drop-shadow(0 0 8px ${color})` }}
          />
          {/* Tick marks */}
          {[0, 25, 50, 75, 100].map((tick) => {
            const angle = (tick / 100) * 360 - 90;
            const rad = (angle * Math.PI) / 180;
            const x1 = 60 + 48 * Math.cos(rad);
            const y1 = 60 + 48 * Math.sin(rad);
            const x2 = 60 + 42 * Math.cos(rad);
            const y2 = 60 + 42 * Math.sin(rad);
            return (
              <line
                key={tick}
                x1={x1} y1={y1} x2={x2} y2={y2}
                stroke="#374151"
                strokeWidth="2"
              />
            );
          })}
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <motion.span
            className="text-2xl font-black font-mono"
            style={{ color }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.5 }}
          >
            {value.toFixed(1)}%
          </motion.span>
        </div>
      </div>
      <span className="text-xs font-bold uppercase tracking-widest text-slate-400 mt-2">
        {label}
      </span>
    </div>
  );
}

// ── OEE STATUS BADGE ─────────────────────────────────────
function OEEBadge({ oee }: { oee: number }) {
  if (oee >= 85) return (
    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-green-500/20 border border-green-500/30 text-green-400 text-xs font-bold">
      <Award className="w-3 h-3" /> WORLD CLASS
    </span>
  );
  if (oee >= 70) return (
    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-blue-500/20 border border-blue-500/30 text-blue-400 text-xs font-bold">
      <TrendingUp className="w-3 h-3" /> GOOD
    </span>
  );
  if (oee >= 60) return (
    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-yellow-500/20 border border-yellow-500/30 text-yellow-400 text-xs font-bold">
      <Target className="w-3 h-3" /> AVERAGE
    </span>
  );
  return (
    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-red-500/20 border border-red-500/30 text-red-400 text-xs font-bold">
      <AlertTriangle className="w-3 h-3" /> NEEDS IMPROVEMENT
    </span>
  );
}

// ── CALCULATE OEE FROM HEALTH ─────────────────────────────
function calculateOEE(healthScore: number, anomalyProb: number): OEEData {
  // Map health score to OEE components
  const baseAvailability = 0.85 + (healthScore / 100) * 0.15;
  const basePerformance = 0.80 + (healthScore / 100) * 0.20;
  const baseQuality = 0.90 + (healthScore / 100) * 0.10;

  // Add realistic noise
  const noise = () => (Math.random() - 0.5) * 0.05;

  const availability = Math.min(100, Math.max(40,
    (baseAvailability + noise()) * 100 * (1 - anomalyProb * 0.3)
  ));
  const performance = Math.min(100, Math.max(40,
    (basePerformance + noise()) * 100 * (1 - anomalyProb * 0.2)
  ));
  const quality = Math.min(100, Math.max(50,
    (baseQuality + noise()) * 100 * (1 - anomalyProb * 0.1)
  ));

  const oee = (availability / 100) * (performance / 100) * (quality / 100) * 100;

  return {
    availability,
    performance,
    quality,
    oee: Math.min(100, Math.max(0, oee)),
    timestamp: new Date().toLocaleTimeString()
  };
}

// Generate weekly trend data
function generateWeeklyData(): WeeklyData[] {
  return WEEKDAYS.map((day, i) => {
    const base = 65 + Math.random() * 20;
    return {
      day,
      oee: parseFloat(base.toFixed(1)),
      availability: parseFloat((base + 5 + Math.random() * 5).toFixed(1)),
      performance: parseFloat((base - 2 + Math.random() * 8).toFixed(1)),
      quality: parseFloat((base + 8 + Math.random() * 5).toFixed(1)),
      target: 85,
    };
  });
}

// ── CUSTOM TOOLTIP ────────────────────────────────────────
const DarkTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#111827] border border-slate-700 rounded-xl p-3 shadow-2xl">
      <p className="text-xs text-slate-400 mb-2">{label}</p>
      {payload.map((p: any) => (
        <p key={p.name} className="text-xs font-semibold" style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(1)}%
        </p>
      ))}
    </div>
  );
};

// ── MAIN OEE DASHBOARD ────────────────────────────────────
export default function OEEDashboard() {
  const [current, setCurrent]     = useState<OEEData | null>(null);
  const [history, setHistory]     = useState<OEEData[]>([]);
  const [weekly, setWeekly]       = useState<WeeklyData[]>([]);
  const [mode, setMode]           = useState<"normal"|"warning"|"fault">("normal");
  const [running, setRunning]     = useState(false);
  const [loading, setLoading]     = useState(false);
  const intervalRef               = useRef<NodeJS.Timeout | null>(null);

  // Generate weekly data on mount
  useEffect(() => {
    setWeekly(generateWeeklyData());
  }, []);

  const fetchOEE = async () => {
    try {
      const res = await simulateReading(mode, 1);
      const oee = calculateOEE(res.health_score, res.anomaly_probability);
      setCurrent(oee);
      setHistory(prev => [...prev.slice(-30), oee]);
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (running) {
      fetchOEE();
      intervalRef.current = setInterval(fetchOEE, 2000);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, mode]);

  // Loss categories
  const losses: LossData[] = current ? [
    {
      category: "Planned Downtime",
      loss: parseFloat((100 - current.availability).toFixed(1)),
      color: "#3B82F6",
      icon: "🔧",
      description: "Scheduled maintenance and changeovers"
    },
    {
      category: "Unplanned Downtime",
      loss: parseFloat((Math.random() * 5).toFixed(1)),
      color: "#EF4444",
      icon: "🚨",
      description: "Equipment failures and breakdowns"
    },
    {
      category: "Speed Loss",
      loss: parseFloat((100 - current.performance).toFixed(1)),
      color: "#F97316",
      icon: "🐢",
      description: "Running below ideal cycle time"
    },
    {
      category: "Quality Defects",
      loss: parseFloat((100 - current.quality).toFixed(1)),
      color: "#A855F7",
      icon: "❌",
      description: "Scrap, rework and rejected parts"
    }
  ] : [];

  const prevWeekOEE = 68.5;
  const weekChange = current ? current.oee - prevWeekOEE : 0;

  return (
    <div>
      <PageHeader
        title="OEE Dashboard"
        subtitle="Overall Equipment Effectiveness — The #1 KPI for factory managers"
      />

      {/* Controls */}
      <div className="flex items-center gap-3 mb-6 flex-wrap">
        <select
          value={mode}
          onChange={e => setMode(e.target.value as any)}
          className="bg-[#111827] border border-slate-700 text-sm rounded-lg px-3 py-2 text-slate-200"
        >
          <option value="normal">🟢 Normal Operation</option>
          <option value="warning">🟡 Warning Mode</option>
          <option value="fault">🔴 Fault Simulation</option>
        </select>

        <button
          onClick={() => setRunning(!running)}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-semibold border transition-all ${
            running
              ? "bg-red-500/20 border-red-500/30 text-red-400"
              : "bg-green-500/20 border-green-500/30 text-green-400"
          }`}
        >
          <Activity className="w-4 h-4" />
          {running ? "Stop Live Feed" : "Start Live Feed"}
        </button>

        <button
          onClick={() => { fetchOEE(); setLoading(true); setTimeout(() => setLoading(false), 500); }}
          className="flex items-center gap-2 px-3 py-2 rounded-lg border border-slate-700 text-slate-400 hover:text-white text-sm transition-all"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>

        {/* OEE Formula Info */}
        <div className="ml-auto flex items-center gap-2 px-4 py-2 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <Info className="w-4 h-4 text-blue-400" />
          <span className="text-xs text-blue-300 font-mono">
            OEE = Availability × Performance × Quality
          </span>
        </div>
      </div>

      {/* No data state */}
      {!current && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="rounded-xl border border-slate-700 bg-[#111827] p-12 text-center mb-6"
        >
          <BarChart3 className="w-12 h-12 text-slate-600 mx-auto mb-3" />
          <p className="text-slate-400 font-medium">Click "Start Live Feed" to begin OEE monitoring</p>
          <p className="text-slate-500 text-sm mt-1">
            Or click "Refresh" for a single reading
          </p>
        </motion.div>
      )}

      {current && (
        <>
          {/* ── MAIN OEE GAUGES ─────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="rounded-xl border border-slate-700 bg-[#111827] p-6 mb-6"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-bold text-white flex items-center gap-2">
                <Target className="w-5 h-5 text-blue-400" />
                Live OEE Metrics
              </h3>
              <OEEBadge oee={current.oee} />
            </div>

            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6">
              {/* Main OEE */}
              <div className="flex flex-col items-center p-4 rounded-xl bg-black/30 border border-slate-700/50">
                <OEEGauge
                  value={current.oee}
                  label="Overall OEE"
                  color={current.oee >= 85 ? "#22C55E" : current.oee >= 70 ? "#3B82F6" : current.oee >= 60 ? "#EAB308" : "#EF4444"}
                />
                <div className="mt-3 flex items-center gap-1.5 text-xs">
                  {weekChange >= 0
                    ? <TrendingUp className="w-3.5 h-3.5 text-green-400" />
                    : <TrendingDown className="w-3.5 h-3.5 text-red-400" />
                  }
                  <span className={weekChange >= 0 ? "text-green-400" : "text-red-400"}>
                    {weekChange >= 0 ? "+" : ""}{weekChange.toFixed(1)}% vs last week
                  </span>
                </div>
              </div>

              {/* Availability */}
              <div className="flex flex-col items-center p-4 rounded-xl bg-black/30 border border-slate-700/50">
                <OEEGauge
                  value={current.availability}
                  label="Availability"
                  color="#3B82F6"
                />
                <p className="text-xs text-slate-500 text-center mt-2">
                  Equipment uptime ratio
                </p>
              </div>

              {/* Performance */}
              <div className="flex flex-col items-center p-4 rounded-xl bg-black/30 border border-slate-700/50">
                <OEEGauge
                  value={current.performance}
                  label="Performance"
                  color="#F97316"
                />
                <p className="text-xs text-slate-500 text-center mt-2">
                  Speed vs ideal rate
                </p>
              </div>

              {/* Quality */}
              <div className="flex flex-col items-center p-4 rounded-xl bg-black/30 border border-slate-700/50">
                <OEEGauge
                  value={current.quality}
                  label="Quality"
                  color="#A855F7"
                />
                <p className="text-xs text-slate-500 text-center mt-2">
                  Good parts ratio
                </p>
              </div>
            </div>

            {/* Benchmark bar */}
            <div className="mt-6 p-4 rounded-lg bg-black/20 border border-slate-800">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-slate-400">Industry Benchmarks</span>
                <span className="text-xs font-mono" style={{
                  color: current.oee >= 85 ? "#22C55E" : current.oee >= 60 ? "#EAB308" : "#EF4444"
                }}>
                  Current: {current.oee.toFixed(1)}%
                </span>
              </div>
              <div className="relative h-4 bg-slate-800 rounded-full overflow-hidden">
                {/* Zones */}
                <div className="absolute left-0 top-0 h-full w-[60%] bg-red-500/20" />
                <div className="absolute left-[60%] top-0 h-full w-[25%] bg-yellow-500/20" />
                <div className="absolute left-[85%] top-0 h-full w-[15%] bg-green-500/20" />

                {/* Current value */}
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${current.oee}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                  className="absolute left-0 top-0 h-full rounded-full"
                  style={{
                    background: current.oee >= 85
                      ? "linear-gradient(90deg, #3B82F6, #22C55E)"
                      : current.oee >= 60
                      ? "linear-gradient(90deg, #3B82F6, #EAB308)"
                      : "linear-gradient(90deg, #3B82F6, #EF4444)"
                  }}
                />

                {/* Markers */}
                <div className="absolute left-[60%] top-0 h-full w-px bg-yellow-400 opacity-60" />
                <div className="absolute left-[85%] top-0 h-full w-px bg-green-400 opacity-60" />
              </div>
              <div className="flex justify-between mt-1">
                <span className="text-[10px] text-slate-500">0%</span>
                <span className="text-[10px] text-yellow-400">60% Industry Avg</span>
                <span className="text-[10px] text-green-400">85% World Class</span>
                <span className="text-[10px] text-slate-500">100%</span>
              </div>
            </div>
          </motion.div>

          {/* ── TWO CHARTS ──────────────────────────────── */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">

            {/* Live OEE trend */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="rounded-xl border border-slate-700 bg-[#111827] p-5"
            >
              <h3 className="font-semibold text-white text-sm mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 text-blue-400" />
                Live OEE Components
              </h3>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={history.slice(-20)}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
                  <XAxis dataKey="timestamp" stroke="#4B5563" fontSize={9} />
                  <YAxis domain={[0, 100]} stroke="#4B5563" fontSize={10} />
                  <Tooltip content={<DarkTooltip />} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine y={85} stroke="#22C55E" strokeDasharray="4 4" label={{ value: "World Class", fill: "#22C55E", fontSize: 9 }} />
                  <ReferenceLine y={60} stroke="#EAB308" strokeDasharray="4 4" label={{ value: "Avg", fill: "#EAB308", fontSize: 9 }} />
                  <Line type="monotone" dataKey="oee" name="OEE" stroke="#3B82F6" strokeWidth={2.5} dot={false} />
                  <Line type="monotone" dataKey="availability" name="Avail" stroke="#22C55E" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="performance" name="Perf" stroke="#F97316" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="quality" name="Quality" stroke="#A855F7" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </motion.div>

            {/* Weekly trend */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="rounded-xl border border-slate-700 bg-[#111827] p-5"
            >
              <h3 className="font-semibold text-white text-sm mb-4 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-400" />
                Week-over-Week OEE Trend
              </h3>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={weekly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
                  <XAxis dataKey="day" stroke="#4B5563" fontSize={11} />
                  <YAxis domain={[0, 100]} stroke="#4B5563" fontSize={10} />
                  <Tooltip content={<DarkTooltip />} />
                  <ReferenceLine y={85} stroke="#22C55E" strokeDasharray="3 3" />
                  <ReferenceLine y={60} stroke="#EAB308" strokeDasharray="3 3" />
                  <Bar dataKey="oee" name="OEE %" radius={[6, 6, 0, 0]} animationDuration={1000}>
                    {weekly.map((d, i) => (
                      <Cell
                        key={i}
                        fill={d.oee >= 85 ? "#22C55E" : d.oee >= 70 ? "#3B82F6" : d.oee >= 60 ? "#EAB308" : "#EF4444"}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </motion.div>
          </div>

          {/* ── LOSS BREAKDOWN ───────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="rounded-xl border border-slate-700 bg-[#111827] p-5 mb-6"
          >
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-orange-400" />
              Six Big Losses Breakdown
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {losses.map((loss, i) => (
                <motion.div
                  key={loss.category}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.3 + i * 0.1 }}
                  className="rounded-xl border p-4"
                  style={{ borderColor: loss.color + "30", background: loss.color + "08" }}
                >
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-xl">{loss.icon}</span>
                    <span className="text-xs font-bold" style={{ color: loss.color }}>
                      {loss.category}
                    </span>
                  </div>

                  <div className="text-3xl font-black font-mono mb-2" style={{ color: loss.color }}>
                    {loss.loss.toFixed(1)}%
                  </div>

                  <div className="w-full h-1.5 bg-slate-800 rounded-full mb-2">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(loss.loss * 5, 100)}%` }}
                      transition={{ duration: 0.8, delay: i * 0.1 }}
                      className="h-full rounded-full"
                      style={{ background: loss.color }}
                    />
                  </div>

                  <p className="text-xs text-slate-500">{loss.description}</p>
                </motion.div>
              ))}
            </div>
          </motion.div>

          {/* ── SUMMARY TABLE ────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="rounded-xl border border-slate-700 bg-[#111827] p-5"
          >
            <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-400" />
              OEE Benchmark Comparison
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-3 px-3 text-xs uppercase tracking-wider text-slate-400">Metric</th>
                    <th className="text-right py-3 px-3 text-xs uppercase tracking-wider text-slate-400">Current</th>
                    <th className="text-right py-3 px-3 text-xs uppercase tracking-wider text-slate-400">Industry Avg</th>
                    <th className="text-right py-3 px-3 text-xs uppercase tracking-wider text-slate-400">World Class</th>
                    <th className="text-right py-3 px-3 text-xs uppercase tracking-wider text-slate-400">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {[
                    { name: "Overall OEE", current: current.oee, avg: 60, wc: 85 },
                    { name: "Availability", current: current.availability, avg: 90, wc: 97 },
                    { name: "Performance", current: current.performance, avg: 80, wc: 95 },
                    { name: "Quality", current: current.quality, avg: 98, wc: 99.9 },
                  ].map((row, i) => {
                    const isGood = row.current >= row.avg;
                    const isWC = row.current >= row.wc;
                    return (
                      <tr key={row.name} className="border-b border-slate-800/50 hover:bg-slate-800/30 transition-colors">
                        <td className="py-3 px-3 font-medium text-white">{row.name}</td>
                        <td className="py-3 px-3 text-right font-mono font-bold" style={{
                          color: isWC ? "#22C55E" : isGood ? "#3B82F6" : "#EF4444"
                        }}>
                          {row.current.toFixed(1)}%
                        </td>
                        <td className="py-3 px-3 text-right text-slate-400 font-mono">{row.avg}%</td>
                        <td className="py-3 px-3 text-right text-green-400 font-mono">{row.wc}%</td>
                        <td className="py-3 px-3 text-right">
                          {isWC
                            ? <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded-full">✓ World Class</span>
                            : isGood
                            ? <span className="text-xs bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded-full">↑ Above Avg</span>
                            : <span className="text-xs bg-red-500/20 text-red-400 px-2 py-0.5 rounded-full">↓ Below Avg</span>
                          }
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </motion.div>
        </>
      )}
    </div>
  );
}