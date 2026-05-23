import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Bell, X, Check, CheckCheck, Trash2, Settings } from "lucide-react";
import { alertStore, Alert, SEVERITY_CONFIG, Severity } from "@/lib/alertStore";
import { useNavigate } from "react-router-dom";

function TimeAgo({ timestamp }: { timestamp: string }) {
  const diff = Date.now() - new Date(timestamp).getTime();
  const mins = Math.floor(diff / 60000);
  const secs = Math.floor(diff / 1000);
  if (secs < 60) return <span>{secs}s ago</span>;
  if (mins < 60) return <span>{mins}m ago</span>;
  return <span>{Math.floor(mins / 60)}h ago</span>;
}

export default function NotificationBell() {
  const [count, setCount]   = useState(0);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [open, setOpen]     = useState(false);
  const [filter, setFilter] = useState<Severity | "ALL">("ALL");
  const panelRef            = useRef<HTMLDivElement>(null);
  const navigate            = useNavigate();

  useEffect(() => {
    const unsub = alertStore.subscribe(() => {
      setCount(alertStore.getUnreadCount());
      setAlerts(alertStore.getAlerts());
    });
    return () => { unsub(); };
  }, []);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  const filtered = filter === "ALL"
    ? alerts
    : alerts.filter(a => a.severity === filter);

  const counts = alertStore.getSeverityCounts();

  return (
    <div ref={panelRef} className="relative">
      {/* Bell Button */}
      <button
        onClick={() => setOpen(!open)}
        className="relative p-2 rounded-lg hover:bg-white/10 transition-colors"
      >
        <motion.div
          animate={count > 0 ? { rotate: [0, -15, 15, -10, 10, 0] } : {}}
          transition={{ duration: 0.6, repeat: count > 0 ? Infinity : 0, repeatDelay: 3 }}
        >
          <Bell className="w-5 h-5 text-slate-300" />
        </motion.div>

        {/* Badge */}
        <AnimatePresence>
          {count > 0 && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0 }}
              className="absolute -top-1 -right-1 min-w-[18px] h-[18px] rounded-full bg-red-500 flex items-center justify-center"
            >
              <span className="text-[10px] font-black text-white px-1">
                {count > 99 ? "99+" : count}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </button>

      {/* Dropdown Panel */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute right-0 top-12 w-96 rounded-xl border border-slate-700 bg-[#111827] shadow-2xl overflow-hidden z-[9999]"
            style={{ boxShadow: "0 25px 50px rgba(0,0,0,0.5)" }}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-700">
              <div className="flex items-center gap-2">
                <Bell className="w-4 h-4 text-blue-400" />
                <span className="font-bold text-white text-sm">
                  Alerts
                </span>
                {count > 0 && (
                  <span className="bg-red-500 text-white text-xs px-2 py-0.5 rounded-full font-bold">
                    {count} new
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => alertStore.acknowledgeAll()}
                  className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                  title="Mark all read"
                >
                  <CheckCheck className="w-4 h-4" />
                </button>
                <button
                  onClick={() => alertStore.clearAll()}
                  className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-red-400 transition-colors"
                  title="Clear all"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
                <button
                  onClick={() => { setOpen(false); navigate("/notifications"); }}
                  className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                  title="Settings"
                >
                  <Settings className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setOpen(false)}
                  className="p-1.5 rounded-lg hover:bg-white/10 text-slate-400 hover:text-white transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Severity Filter */}
            <div className="flex gap-1 p-2 border-b border-slate-700 overflow-x-auto">
              {(["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"] as const).map(sev => (
                <button
                  key={sev}
                  onClick={() => setFilter(sev)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-bold whitespace-nowrap transition-all ${
                    filter === sev
                      ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                      : "text-slate-400 hover:text-white hover:bg-white/5"
                  }`}
                >
                  {sev}
                  {sev !== "ALL" && counts[sev] > 0 && (
                    <span
                      className="ml-1 px-1 rounded text-[10px]"
                      style={{
                        background: SEVERITY_CONFIG[sev].color + "30",
                        color: SEVERITY_CONFIG[sev].color
                      }}
                    >
                      {counts[sev]}
                    </span>
                  )}
                </button>
              ))}
            </div>

            {/* Alert List */}
            <div className="max-h-96 overflow-y-auto">
              {filtered.length === 0 ? (
                <div className="p-8 text-center">
                  <Bell className="w-8 h-8 text-slate-600 mx-auto mb-2" />
                  <p className="text-sm text-slate-500">No alerts</p>
                  <p className="text-xs text-slate-600 mt-1">
                    Start Live Monitor to detect anomalies
                  </p>
                </div>
              ) : (
                filtered.map(alert => {
                  const cfg = SEVERITY_CONFIG[alert.severity];
                  return (
                    <motion.div
                      key={alert.id}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      className={`flex gap-3 p-3 border-b border-slate-800/50 hover:bg-white/5 transition-colors ${
                        !alert.acknowledged ? "border-l-2" : ""
                      }`}
                      style={!alert.acknowledged ? { borderLeftColor: cfg.color } : {}}
                    >
                      {/* Severity dot */}
                      <div
                        className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
                        style={{ background: cfg.bg }}
                      >
                        <div
                          className="w-2.5 h-2.5 rounded-full"
                          style={{
                            background: cfg.color,
                            boxShadow: `0 0 6px ${cfg.color}`,
                          }}
                        />
                      </div>

                      {/* Content */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span
                            className="text-[10px] font-black uppercase"
                            style={{ color: cfg.color }}
                          >
                            {alert.severity}
                          </span>
                          <span className="text-[10px] text-slate-500">
                            Engine #{alert.engine_id}
                          </span>
                          {alert.escalated && (
                            <span className="text-[10px] text-purple-400">
                              ↑ Escalated
                            </span>
                          )}
                        </div>
                        <p className="text-xs text-white font-medium truncate mt-0.5">
                          {alert.root_cause}
                        </p>
                        <div className="flex items-center gap-3 mt-1">
                          <span className="text-[10px] text-slate-500">
                            <TimeAgo timestamp={alert.timestamp} />
                          </span>
                          <span className="text-[10px] text-slate-500">
                            {(alert.anomaly_probability * 100).toFixed(1)}% prob
                          </span>
                        </div>
                      </div>

                      {/* Acknowledge */}
                      {!alert.acknowledged && (
                        <button
                          onClick={() => alertStore.acknowledgeAlert(alert.id)}
                          className="text-slate-500 hover:text-green-400 transition-colors flex-shrink-0"
                          title="Acknowledge"
                        >
                          <Check className="w-4 h-4" />
                        </button>
                      )}
                    </motion.div>
                  );
                })
              )}
            </div>

            {/* Footer */}
            {alerts.length > 0 && (
              <div className="p-3 border-t border-slate-700 text-center">
                <button
                  onClick={() => { setOpen(false); navigate("/notifications"); }}
                  className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
                >
                  View all alerts & settings →
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}