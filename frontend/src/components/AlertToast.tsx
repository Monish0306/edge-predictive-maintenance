import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, AlertTriangle, Bell } from "lucide-react";
import { alertStore, Alert, SEVERITY_CONFIG } from "@/lib/alertStore";

// Individual Toast
function Toast({ alert, onDismiss }: { alert: Alert; onDismiss: () => void }) {
  const cfg = SEVERITY_CONFIG[alert.severity];

  useEffect(() => {
    const timer = setTimeout(onDismiss,
      alert.severity === "CRITICAL" ? 8000 :
      alert.severity === "HIGH" ? 6000 : 4000
    );
    return () => clearTimeout(timer);
  }, [alert.severity]);

  return (
    <motion.div
      initial={{ opacity: 0, x: 400, scale: 0.9 }}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      exit={{ opacity: 0, x: 400, scale: 0.9 }}
      transition={{ type: "spring", stiffness: 300, damping: 25 }}
      className="w-80 rounded-xl border p-4 shadow-2xl backdrop-blur-xl"
      style={{
        background: cfg.bg,
        borderColor: cfg.border,
        boxShadow: `0 0 30px ${cfg.color}20`,
      }}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div
          className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0"
          style={{ background: cfg.color + "20" }}
        >
          <AlertTriangle className="w-5 h-5" style={{ color: cfg.color }} />
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span
              className="text-xs font-black uppercase tracking-wider px-2 py-0.5 rounded-full"
              style={{ background: cfg.color + "20", color: cfg.color }}
            >
              {alert.severity}
            </span>
            <span className="text-xs text-slate-400">
              Engine #{alert.engine_id}
            </span>
          </div>

          <p className="text-sm font-semibold text-white leading-tight mb-1">
            {alert.root_cause}
          </p>

          <p className="text-xs text-slate-400">
            Anomaly: {(alert.anomaly_probability * 100).toFixed(1)}% •
            Health: {alert.health_score.toFixed(1)}%
          </p>

          {/* Escalation notice */}
          {alert.escalated && (
            <div className="mt-2 text-xs text-purple-400 flex items-center gap-1">
              <Bell className="w-3 h-3" />
              Escalated to management
            </div>
          )}
        </div>

        {/* Dismiss */}
        <button
          onClick={onDismiss}
          className="text-slate-500 hover:text-white transition-colors flex-shrink-0"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Progress bar */}
      <motion.div
        className="mt-3 h-0.5 rounded-full"
        style={{ background: cfg.color }}
        initial={{ scaleX: 1, transformOrigin: "left" }}
        animate={{ scaleX: 0 }}
        transition={{
          duration: alert.severity === "CRITICAL" ? 8 :
                    alert.severity === "HIGH" ? 6 : 4,
          ease: "linear"
        }}
      />
    </motion.div>
  );
}

// Toast Container
export default function AlertToastContainer() {
  const [toasts, setToasts] = useState<Alert[]>([]);

  useEffect(() => {
    const unsub = alertStore.onNewAlert((alert) => {
      setToasts(prev => [alert, ...prev].slice(0, 4));

      // Sound notification
      if (alertStore.getSettings().sound_enabled) {
        try {
          const ctx = new AudioContext();
          const osc = ctx.createOscillator();
          const gain = ctx.createGain();
          osc.connect(gain);
          gain.connect(ctx.destination);
          osc.frequency.value = alert.severity === "CRITICAL" ? 880 :
                                 alert.severity === "HIGH" ? 660 : 440;
          gain.gain.setValueAtTime(0.1, ctx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.5);
          osc.start(ctx.currentTime);
          osc.stop(ctx.currentTime + 0.5);
        } catch {}
      }
    });
    return () => {
      unsub();
    };
  }, []);

  const dismiss = (id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  };

  return (
    <div className="fixed top-4 right-4 z-[99995] space-y-3 pointer-events-none">
      <AnimatePresence mode="popLayout">
        {toasts.map(toast => (
          <div key={toast.id} className="pointer-events-auto">
            <Toast
              alert={toast}
              onDismiss={() => dismiss(toast.id)}
            />
          </div>
        ))}
      </AnimatePresence>
    </div>
  );
}