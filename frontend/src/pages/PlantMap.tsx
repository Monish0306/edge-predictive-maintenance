import { useState, useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from "react-leaflet";
import { motion, AnimatePresence } from "framer-motion";
import {
  Globe, AlertTriangle, CheckCircle, Factory,
  Cpu, TrendingUp, X, Activity, RefreshCw
} from "lucide-react";
import { PageHeader } from "@/components/PageHeader";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from "recharts";

// ── TYPES ───────────────────────────────────────────────
interface Plant {
  id: string;
  name: string;
  city: string;
  country: string;
  lat: number;
  lng: number;
  engines: number;
  healthScore: number;
  criticalAlerts: number;
  oee: number;
  status: "normal" | "warning" | "critical";
  region: string;
  production: string;
}

// ── MOCK PLANT DATA ──────────────────────────────────────
const PLANTS: Plant[] = [
  { id: "P001", name: "Detroit Auto Plant", city: "Detroit", country: "USA", lat: 42.33, lng: -83.04, engines: 48, healthScore: 87, criticalAlerts: 0, oee: 88, status: "normal", region: "North America", production: "Automotive" },
  { id: "P002", name: "Chicago Aerospace", city: "Chicago", country: "USA", lat: 41.87, lng: -87.62, engines: 32, healthScore: 62, criticalAlerts: 2, oee: 71, status: "warning", region: "North America", production: "Aerospace" },
  { id: "P003", name: "Houston Oil & Gas", city: "Houston", country: "USA", lat: 29.76, lng: -95.36, engines: 67, healthScore: 34, criticalAlerts: 5, oee: 54, status: "critical", region: "North America", production: "Oil & Gas" },
  { id: "P004", name: "Stuttgart Automotive", city: "Stuttgart", country: "Germany", lat: 48.77, lng: 9.18, engines: 55, healthScore: 92, criticalAlerts: 0, oee: 91, status: "normal", region: "Europe", production: "Automotive" },
  { id: "P005", name: "Munich Semiconductor", city: "Munich", country: "Germany", lat: 48.13, lng: 11.57, engines: 28, healthScore: 78, criticalAlerts: 1, oee: 79, status: "warning", region: "Europe", production: "Semiconductor" },
  { id: "P006", name: "Tokyo Electronics", city: "Tokyo", country: "Japan", lat: 35.68, lng: 139.69, engines: 41, healthScore: 95, criticalAlerts: 0, oee: 94, status: "normal", region: "Asia Pacific", production: "Electronics" },
  { id: "P007", name: "Shanghai Manufacturing", city: "Shanghai", country: "China", lat: 31.23, lng: 121.47, engines: 89, healthScore: 71, criticalAlerts: 3, oee: 72, status: "warning", region: "Asia Pacific", production: "Manufacturing" },
  { id: "P008", name: "Seoul Semiconductor", city: "Seoul", country: "S. Korea", lat: 37.56, lng: 126.97, engines: 36, healthScore: 88, criticalAlerts: 0, oee: 87, status: "normal", region: "Asia Pacific", production: "Semiconductor" },
  { id: "P009", name: "Bangalore Tech Park", city: "Bangalore", country: "India", lat: 12.97, lng: 77.59, engines: 22, healthScore: 55, criticalAlerts: 4, oee: 61, status: "critical", region: "Asia Pacific", production: "Technology" },
  { id: "P010", name: "London Pharma", city: "London", country: "UK", lat: 51.50, lng: -0.12, engines: 19, healthScore: 84, criticalAlerts: 0, oee: 83, status: "normal", region: "Europe", production: "Pharmaceutical" },
  { id: "P011", name: "São Paulo Heavy Ind.", city: "São Paulo", country: "Brazil", lat: -23.55, lng: -46.63, engines: 44, healthScore: 67, criticalAlerts: 2, oee: 68, status: "warning", region: "South America", production: "Heavy Industry" },
  { id: "P012", name: "Dubai Energy", city: "Dubai", country: "UAE", lat: 25.20, lng: 55.27, engines: 31, healthScore: 79, criticalAlerts: 1, oee: 76, status: "warning", region: "Middle East", production: "Energy" },
];

// ── STATUS CONFIG ────────────────────────────────────────
const STATUS = {
  normal:   { color: "#22C55E", bg: "rgba(34,197,94,0.15)",   border: "#22C55E40", label: "Normal",   radius: 14 },
  warning:  { color: "#EAB308", bg: "rgba(234,179,8,0.15)",   border: "#EAB30840", label: "Warning",  radius: 18 },
  critical: { color: "#EF4444", bg: "rgba(239,68,68,0.15)",   border: "#EF444440", label: "Critical", radius: 22 },
};

// ── PLANT CARD ───────────────────────────────────────────
function PlantCard({ plant, onClose }: { plant: Plant; onClose: () => void }) {
  const cfg = STATUS[plant.status];

  return (
    <motion.div
      initial={{ opacity: 0, x: 20, scale: 0.95 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 20, scale: 0.95 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
      className="absolute top-4 right-4 w-80 rounded-2xl border shadow-2xl z-[1000] overflow-hidden"
      style={{
        background: "#111827",
        borderColor: cfg.color + "40",
        boxShadow: `0 0 40px ${cfg.color}20`,
      }}
    >
      {/* Header */}
      <div className="p-4 border-b border-slate-700" style={{ background: cfg.color + "10" }}>
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <Factory className="w-4 h-4" style={{ color: cfg.color }} />
              <span className="font-bold text-white text-sm">{plant.name}</span>
            </div>
            <div className="text-xs text-slate-400">{plant.city}, {plant.country}</div>
            <div className="text-xs text-slate-500 mt-0.5">{plant.production} • {plant.region}</div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white transition-colors">
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Status badge */}
        <div
          className="mt-3 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold uppercase"
          style={{ background: cfg.color + "20", color: cfg.color }}
        >
          <div className="w-1.5 h-1.5 rounded-full" style={{ background: cfg.color }} />
          {cfg.label}
          {plant.criticalAlerts > 0 && ` — ${plant.criticalAlerts} Alert${plant.criticalAlerts > 1 ? "s" : ""}`}
        </div>
      </div>

      {/* Metrics */}
      <div className="p-4 grid grid-cols-2 gap-3">
        {[
          { label: "Engines", value: plant.engines.toString(), icon: "⚙️" },
          { label: "Health", value: `${plant.healthScore}%`, icon: "❤️" },
          { label: "OEE", value: `${plant.oee}%`, icon: "📊" },
          { label: "Alerts", value: plant.criticalAlerts.toString(), icon: "🚨" },
        ].map((m) => (
          <div key={m.label} className="bg-black/30 rounded-xl p-3 text-center">
            <div className="text-lg mb-0.5">{m.icon}</div>
            <div className="text-lg font-black font-mono text-white">{m.value}</div>
            <div className="text-[10px] text-slate-500 uppercase tracking-wide">{m.label}</div>
          </div>
        ))}
      </div>

      {/* Health bar */}
      <div className="px-4 pb-4">
        <div className="flex justify-between text-xs mb-1">
          <span className="text-slate-400">Plant Health</span>
          <span className="font-mono" style={{ color: cfg.color }}>
            {plant.healthScore}%
          </span>
        </div>
        <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${plant.healthScore}%` }}
            transition={{ duration: 0.8 }}
            className="h-full rounded-full"
            style={{ background: cfg.color }}
          />
        </div>

        {/* Engine fleet preview */}
        <div className="mt-3 flex gap-1 flex-wrap">
          {Array.from({ length: Math.min(plant.engines, 24) }).map((_, i) => (
            <div
              key={i}
              className="w-2.5 h-2.5 rounded-sm"
              style={{
                background: i < Math.floor(plant.engines * (plant.healthScore / 100))
                  ? cfg.color
                  : "#374151",
                opacity: 0.8,
              }}
            />
          ))}
          {plant.engines > 24 && (
            <span className="text-[10px] text-slate-500 ml-1">+{plant.engines - 24}</span>
          )}
        </div>
        <div className="text-[10px] text-slate-500 mt-1">
          Engine fleet health overview
        </div>
      </div>
    </motion.div>
  );
}

// ── MAP FIT BOUNDS ────────────────────────────────────────
function MapController() {
  const map = useMap();
  useEffect(() => {
    map.setView([20, 10], 2);
  }, [map]);
  return null;
}

// ── MAIN PLANT MAP ────────────────────────────────────────
export default function PlantMap() {
  const [selected, setSelected]   = useState<Plant | null>(null);
  const [filter, setFilter]       = useState<"all" | "normal" | "warning" | "critical">("all");
  const [region, setRegion]       = useState<string>("All Regions");

  const regions = ["All Regions", ...Array.from(new Set(PLANTS.map(p => p.region)))];

  const filtered = PLANTS.filter(p => {
    if (filter !== "all" && p.status !== filter) return false;
    if (region !== "All Regions" && p.region !== region) return false;
    return true;
  });

  const totals = {
    engines: PLANTS.reduce((s, p) => s + p.engines, 0),
    alerts:  PLANTS.reduce((s, p) => s + p.criticalAlerts, 0),
    avgHealth: Math.round(PLANTS.reduce((s, p) => s + p.healthScore, 0) / PLANTS.length),
    avgOEE:   Math.round(PLANTS.reduce((s, p) => s + p.oee, 0) / PLANTS.length),
  };

  const statusCounts = {
    normal:   PLANTS.filter(p => p.status === "normal").length,
    warning:  PLANTS.filter(p => p.status === "warning").length,
    critical: PLANTS.filter(p => p.status === "critical").length,
  };

  const regionData = regions.slice(1).map(r => ({
    name: r.replace(" ", "\n"),
    plants: PLANTS.filter(p => p.region === r).length,
    health: Math.round(PLANTS.filter(p => p.region === r).reduce((s, p) => s + p.healthScore, 0) / PLANTS.filter(p => p.region === r).length),
  }));

  return (
    <div>
      <PageHeader
        title="Multi-Plant Overview Map"
        subtitle="Global factory network — 12 plants, 12 countries, real-time health monitoring"
      />

      {/* Global Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {[
          { label: "Total Plants",    value: PLANTS.length.toString(), icon: Factory,  color: "#3B82F6" },
          { label: "Total Engines",   value: totals.engines.toString(), icon: Cpu,    color: "#A855F7" },
          { label: "Active Alerts",   value: totals.alerts.toString(),  icon: AlertTriangle, color: totals.alerts > 0 ? "#EF4444" : "#22C55E" },
          { label: "Avg Plant Health",value: `${totals.avgHealth}%`,    icon: Activity, color: "#22C55E" },
        ].map((stat, i) => {
          const Icon = stat.icon;
          return (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.1 }}
              className="rounded-xl border border-slate-700 bg-[#111827] p-4"
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4" style={{ color: stat.color }} />
                <span className="text-xs text-slate-400">{stat.label}</span>
              </div>
              <div className="text-2xl font-black font-mono" style={{ color: stat.color }}>
                {stat.value}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        {(["all", "normal", "warning", "critical"] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold uppercase border transition-all ${
              filter === f
                ? f === "all"
                  ? "bg-blue-500/20 border-blue-500/30 text-blue-400"
                  : f === "normal"
                  ? "bg-green-500/20 border-green-500/30 text-green-400"
                  : f === "warning"
                  ? "bg-yellow-500/20 border-yellow-500/30 text-yellow-400"
                  : "bg-red-500/20 border-red-500/30 text-red-400"
                : "border-slate-700 text-slate-400 hover:text-white"
            }`}
          >
            {f === "all" ? `All (${PLANTS.length})` :
             f === "normal" ? `🟢 Normal (${statusCounts.normal})` :
             f === "warning" ? `🟡 Warning (${statusCounts.warning})` :
             `🔴 Critical (${statusCounts.critical})`}
          </button>
        ))}

        <select
          value={region}
          onChange={e => setRegion(e.target.value)}
          className="bg-[#111827] border border-slate-700 text-slate-200 text-xs rounded-lg px-3 py-1.5 ml-auto"
        >
          {regions.map(r => <option key={r}>{r}</option>)}
        </select>
      </div>

      {/* Map + Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">

        {/* Map */}
        <div className="lg:col-span-2">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="rounded-xl border border-slate-700 overflow-hidden relative"
            style={{ height: "480px" }}
          >
            <MapContainer
              center={[20, 10]}
              zoom={2}
              style={{ height: "100%", width: "100%", background: "#0A0F1E" }}
              zoomControl={true}
              scrollWheelZoom={true}
            >
              <MapController />
              <TileLayer
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                attribution='© OpenStreetMap © CartoDB'
              />

              {filtered.map(plant => {
                const cfg = STATUS[plant.status];
                return (
                  <CircleMarker
                    key={plant.id}
                    center={[plant.lat, plant.lng]}
                    radius={cfg.radius}
                    fillColor={cfg.color}
                    color={cfg.color}
                    weight={2}
                    opacity={0.9}
                    fillOpacity={0.4}
                    eventHandlers={{ click: () => setSelected(plant) }}
                  >
                    <Popup>
                      <div style={{ background: "#111827", color: "white", padding: "8px", borderRadius: "8px", minWidth: "160px" }}>
                        <strong style={{ color: cfg.color }}>{plant.name}</strong>
                        <br />
                        <span style={{ fontSize: "11px", color: "#9CA3AF" }}>
                          {plant.city}, {plant.country}
                        </span>
                        <br />
                        <span style={{ fontSize: "11px" }}>
                          Health: {plant.healthScore}% | OEE: {plant.oee}%
                        </span>
                        <br />
                        <span style={{ fontSize: "11px" }}>
                          {plant.engines} engines | {plant.criticalAlerts} alerts
                        </span>
                      </div>
                    </Popup>
                  </CircleMarker>
                );
              })}
            </MapContainer>

            {/* Legend */}
            <div className="absolute bottom-4 left-4 bg-black/80 backdrop-blur rounded-xl p-3 z-[500] border border-slate-700">
              <p className="text-[10px] text-slate-400 mb-2 uppercase tracking-wider">Status Legend</p>
              {Object.entries(STATUS).map(([key, cfg]) => (
                <div key={key} className="flex items-center gap-2 mb-1">
                  <div className="w-3 h-3 rounded-full" style={{ background: cfg.color }} />
                  <span className="text-[10px] text-slate-300 capitalize">{key}</span>
                </div>
              ))}
              <p className="text-[10px] text-slate-500 mt-2">Click marker for details</p>
            </div>

            {/* Selected plant panel */}
            <AnimatePresence>
              {selected && (
                <PlantCard
                  plant={selected}
                  onClose={() => setSelected(null)}
                />
              )}
            </AnimatePresence>
          </motion.div>
        </div>

        {/* Plant List */}
        <div className="rounded-xl border border-slate-700 bg-[#111827] overflow-hidden">
          <div className="p-4 border-b border-slate-700">
            <h3 className="font-semibold text-white text-sm flex items-center gap-2">
              <Globe className="w-4 h-4 text-blue-400" />
              All Plants ({filtered.length})
            </h3>
          </div>
          <div className="overflow-y-auto" style={{ maxHeight: "432px" }}>
            {filtered
              .sort((a, b) => b.criticalAlerts - a.criticalAlerts)
              .map((plant, i) => {
                const cfg = STATUS[plant.status];
                return (
                  <motion.button
                    key={plant.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: i * 0.03 }}
                    onClick={() => setSelected(plant)}
                    className={`w-full flex items-center gap-3 p-3 border-b border-slate-800/50 hover:bg-slate-800/40 transition-all text-left ${
                      selected?.id === plant.id ? "bg-slate-800/60" : ""
                    }`}
                  >
                    {/* Status dot */}
                    <div className="relative flex-shrink-0">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ background: cfg.color, boxShadow: `0 0 6px ${cfg.color}` }}
                      />
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <div className="text-xs font-semibold text-white truncate">{plant.name}</div>
                      <div className="text-[10px] text-slate-500">{plant.city} • {plant.engines} engines</div>
                    </div>

                    {/* Health */}
                    <div className="flex-shrink-0 text-right">
                      <div className="text-xs font-mono font-bold" style={{ color: cfg.color }}>
                        {plant.healthScore}%
                      </div>
                      {plant.criticalAlerts > 0 && (
                        <div className="text-[10px] text-red-400">
                          {plant.criticalAlerts} alert{plant.criticalAlerts > 1 ? "s" : ""}
                        </div>
                      )}
                    </div>
                  </motion.button>
                );
              })}
          </div>
        </div>
      </div>

      {/* Region Chart */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="rounded-xl border border-slate-700 bg-[#111827] p-5"
      >
        <h3 className="font-semibold text-white text-sm mb-4 flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-green-400" />
          Regional Plant Health Comparison
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={regionData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1F2937" />
            <XAxis dataKey="name" stroke="#4B5563" fontSize={10} />
            <YAxis domain={[0, 100]} stroke="#4B5563" fontSize={10} />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
              labelStyle={{ color: "#9CA3AF" }}
            />
            <Bar dataKey="health" name="Avg Health %" radius={[6, 6, 0, 0]} animationDuration={1000}>
              {regionData.map((d, i) => (
                <Cell
                  key={i}
                  fill={d.health >= 80 ? "#22C55E" : d.health >= 65 ? "#3B82F6" : d.health >= 50 ? "#EAB308" : "#EF4444"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
}