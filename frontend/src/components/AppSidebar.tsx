import { NavLink, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity, Cpu, BarChart3, Map, Clock, FileText,
  Bot, Database, DollarSign, Settings,
  ChevronLeft, ChevronRight, Bell,
  Globe,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";
const items = [
  { title: "Live Monitor",     url: "/",          icon: Activity,   color: "#EF4444" },
  { title: "Fleet Overview",   url: "/fleet",     icon: Cpu,        color: "#3B82F6" },
  { title: "Analytics",        url: "/analytics", icon: BarChart3,  color: "#A855F7" },
  { title: "Sensor Heatmap",   url: "/heatmap",   icon: Map,        color: "#06B6D4" },
  { title: "Failure Timeline", url: "/timeline",  icon: Clock,      color: "#EAB308" },
  { title: "Reports",          url: "/reports",   icon: FileText,   color: "#22C55E" },
  { title: "Agent Log",        url: "/agent-log", icon: Bot,        color: "#F97316" },
  { title: "Dataset Stats",    url: "/datasets",  icon: Database,   color: "#6366F1" },
  { title: "Cost Savings",     url: "/savings",   icon: DollarSign, color: "#10B981" },
  { title: "Model Info",       url: "/model",     icon: Settings,   color: "#94A3B8" },
  { title: "Digital Twin", url: "/digital-twin", icon: Cpu, color: "#06B6D4" },
  { title: "Notifications", url: "/notifications", icon: Bell, color: "#F59E0B" },
  { title: "OEE Dashboard",  url: "/oee",       icon: BarChart3, color: "#10B981" },
  { title: "Plant Map",      url: "/plant-map",  icon: Globe,     color: "#06B6D4" },
];

export const AppSidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const { pathname } = useLocation();

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 256 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="sticky top-0 h-screen bg-sidebar border-r border-sidebar-border flex flex-col z-40 overflow-hidden"
    >

      {/* ── LOGO AREA ─────────────────────────────── */}
      <div className="p-4 border-b border-sidebar-border flex items-center gap-3 h-[72px] shrink-0">

        {/* Gear Icon with pulse ring */}
        <div className="relative shrink-0">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-lg">
            <svg
              className="w-5 h-5 text-white"
              fill="none" stroke="currentColor" viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"
              />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
              />
            </svg>
          </div>
          {/* Live pulse ring */}
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500 border-2 border-[#0A0F1E]"></span>
          </span>
        </div>

        {/* Brand Text */}
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0 }}
            className="overflow-hidden"
          >
            <div className="font-bold text-white text-sm leading-tight tracking-tight">
              Edge AI
            </div>
            <div className="text-[10px] text-slate-400 font-medium uppercase tracking-widest mt-0.5">
              Predictive Maintenance
            </div>
          </motion.div>
        )}
      </div>

      {/* ── SECTION LABEL ─────────────────────────── */}
      {!collapsed && (
        <div className="px-4 pt-4 pb-1">
          <p className="text-[10px] font-semibold text-slate-500 uppercase tracking-widest">
            Navigation
          </p>
        </div>
      )}

      {/* ── NAV ITEMS ─────────────────────────────── */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-0.5">
        {items.map((item, i) => {
          const active = pathname === item.url;
          const Icon = item.icon;
          return (
            <motion.div
              key={item.url}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.04 }}
            >
              <NavLink
                to={item.url}
                title={collapsed ? item.title : undefined}
                className={cn(
                  "group flex items-center gap-3 px-3 py-2.5 rounded-lg relative transition-all duration-200 cursor-pointer",
                  "hover:bg-white/5 hover:translate-x-1",
                  active
                    ? "bg-blue-500/15 border border-blue-500/20"
                    : "border border-transparent"
                )}
              >
                {/* Active left bar */}
                {active && (
                  <motion.div
                    layoutId="active-bar"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-blue-500 rounded-r-full"
                    style={{ boxShadow: "0 0 8px #3B82F6" }}
                  />
                )}

                {/* Icon */}
                <Icon
                  className="w-[18px] h-[18px] shrink-0 transition-transform duration-200 group-hover:scale-110"
                  style={{ color: active ? "#60A5FA" : item.color }}
                />

                {/* Label */}
                {!collapsed && (
                  <span className={cn(
                    "text-sm font-medium truncate transition-colors",
                    active ? "text-blue-300" : "text-slate-300 group-hover:text-white"
                  )}>
                    {item.title}
                  </span>
                )}

                {/* Active dot */}
                {active && !collapsed && (
                  <motion.div
                    layoutId="active-dot"
                    className="ml-auto w-1.5 h-1.5 rounded-full bg-blue-400"
                  />
                )}
              </NavLink>
            </motion.div>
          );
        })}
      </nav>

      {/* ── BOTTOM STATUS ─────────────────────────── */}
      <div className="p-3 border-t border-sidebar-border space-y-2 shrink-0">

        {/* Model Status Card */}
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="rounded-lg border border-green-500/20 bg-green-500/5 p-3"
          >
            {/* Header row */}
            <div className="flex items-center gap-2 mb-2">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
              </span>
              <span className="text-xs font-bold text-green-400 tracking-wide">
                MODEL ACTIVE
              </span>
            </div>

            {/* Stats row */}
            <div className="grid grid-cols-2 gap-1.5">
              <div className="bg-black/20 rounded px-2 py-1">
                <div className="text-[9px] text-slate-500 uppercase tracking-wide">Latency</div>
                <div className="text-xs font-bold text-white font-mono">0.20ms</div>
              </div>
              <div className="bg-black/20 rounded px-2 py-1">
                <div className="text-[9px] text-slate-500 uppercase tracking-wide">Accuracy</div>
                <div className="text-xs font-bold text-white font-mono">98.82%</div>
              </div>
            </div>

            {/* ONNX badge */}
            <div className="mt-2 flex items-center gap-1">
              <span className="text-[9px] bg-blue-500/20 text-blue-400 border border-blue-500/20 px-1.5 py-0.5 rounded font-mono font-semibold">
                ONNX
              </span>
              <span className="text-[9px] text-slate-500">Edge Runtime</span>
            </div>
          </motion.div>
        )}

        {/* Collapsed status dot */}
        {collapsed && (
          <div className="flex justify-center">
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-3 w-3 bg-green-500"></span>
            </span>
          </div>
        )}

        {/* Collapse button */}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-slate-400 hover:text-white transition-all duration-200 border border-white/5"
        >
          {collapsed
            ? <ChevronRight className="w-4 h-4" />
            : <>
                <ChevronLeft className="w-4 h-4" />
                <span className="text-xs font-medium">Collapse</span>
              </>
          }
        </button>

        {/* Footer branding */}
        {!collapsed && (
          <div className="text-center">
            <p className="text-[9px] text-slate-600 uppercase tracking-widest">
              NASA Turbofan • 709 Engines
            </p>
          </div>
        )}
      </div>
    </motion.aside>
  );
};