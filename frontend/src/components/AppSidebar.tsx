import { NavLink, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Activity, Cpu, BarChart3, Map, Clock, FileText, Bot, Database,
  DollarSign, Settings, ChevronLeft, ChevronRight, Cog,
} from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const items = [
  { title: "Live Monitor", url: "/", icon: Activity, color: "#EF4444" },
  { title: "Fleet Overview", url: "/fleet", icon: Cpu, color: "#3B82F6" },
  { title: "Analytics", url: "/analytics", icon: BarChart3, color: "#A855F7" },
  { title: "Sensor Heatmap", url: "/heatmap", icon: Map, color: "#06B6D4" },
  { title: "Failure Timeline", url: "/timeline", icon: Clock, color: "#EAB308" },
  { title: "Reports", url: "/reports", icon: FileText, color: "#22C55E" },
  { title: "Agent Log", url: "/agent-log", icon: Bot, color: "#F97316" },
  { title: "Dataset Stats", url: "/datasets", icon: Database, color: "#6366F1" },
  { title: "Cost Savings", url: "/savings", icon: DollarSign, color: "#10B981" },
  { title: "Model Info", url: "/model", icon: Settings, color: "#94A3B8" },
];

export const AppSidebar = () => {
  const [collapsed, setCollapsed] = useState(false);
  const { pathname } = useLocation();

  return (
    <motion.aside
      animate={{ width: collapsed ? 72 : 240 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="sticky top-0 h-screen bg-sidebar border-r border-sidebar-border flex flex-col z-40"
    >
      {/* Logo */}
      <div className="p-4 border-b border-sidebar-border flex items-center gap-3 h-[72px]">
        <motion.div
          animate={{ rotate: collapsed ? 0 : 360 }}
          transition={{ duration: 1.5, ease: "easeInOut" }}
          className="w-10 h-10 rounded-xl bg-gradient-primary flex items-center justify-center shrink-0 shadow-glow"
        >
          <Cog className="w-5 h-5 text-primary-foreground" />
        </motion.div>
        {!collapsed && (
          <motion.div initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} className="overflow-hidden">
            <div className="font-bold text-sidebar-accent-foreground text-base leading-tight">Edge AI</div>
            <div className="text-[10px] text-sidebar-foreground uppercase tracking-wider">Pred. Maintenance</div>
          </motion.div>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
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
                className={cn(
                  "group flex items-center gap-3 px-3 py-2.5 rounded-lg relative transition-all duration-200",
                  "hover:bg-sidebar-accent hover:translate-x-1",
                  active && "bg-primary/15 text-primary"
                )}
              >
                {active && (
                  <motion.div
                    layoutId="active-indicator"
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-1 h-6 bg-primary rounded-r-full shadow-glow"
                  />
                )}
                <Icon
                  className="w-5 h-5 shrink-0 transition-transform group-hover:scale-110"
                  style={{ color: active ? "hsl(var(--primary))" : item.color }}
                />
                {!collapsed && (
                  <span className={cn("text-sm font-medium truncate", active ? "text-primary" : "text-sidebar-foreground")}>
                    {item.title}
                  </span>
                )}
              </NavLink>
            </motion.div>
          );
        })}
      </nav>

      {/* Status */}
      <div className="p-3 border-t border-sidebar-border space-y-2">
        {!collapsed && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="flex items-center gap-2 px-3 py-2 rounded-lg bg-success/10 border border-success/20"
          >
            <span className="w-2 h-2 rounded-full bg-success pulse-dot" />
            <div className="flex-1 min-w-0">
              <div className="text-xs font-semibold text-success">Model Active</div>
              <div className="text-[10px] text-muted-foreground">0.20ms latency</div>
            </div>
          </motion.div>
        )}
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="w-full flex items-center justify-center gap-2 py-2 rounded-lg bg-sidebar-accent hover:bg-sidebar-accent/70 text-sidebar-foreground transition-colors"
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <><ChevronLeft className="w-4 h-4" /><span className="text-xs">Collapse</span></>}
        </button>
      </div>
    </motion.aside>
  );
};
