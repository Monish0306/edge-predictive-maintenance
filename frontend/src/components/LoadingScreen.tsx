/**
 * LoadingScreen.tsx
 * =================
 * Shows a professional loading screen with a live status message
 * while the backend wakes up on first load.
 * Replaces the blank/broken charts that confused interviewers.
 *
 * Usage: wrap your routes in App.tsx with <AppLoader>
 */

import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, Activity, CheckCircle2 } from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface Props {
  children: React.ReactNode;
}

type Status = "waking" | "ready" | "timeout";

export function AppLoader({ children }: Props) {
  const [status,  setStatus]  = useState<Status>("waking");
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    let attempts = 0;
    const MAX_ATTEMPTS = 20;  // 20 × 2s = 40s max wait

    // Tick elapsed counter every second for the loading message
    const ticker = setInterval(() => setElapsed(s => s + 1), 1000);

    const tryConnect = async () => {
      try {
        const res = await fetch(`${API}/health`, {
          signal: AbortSignal.timeout(4000),
        });
        if (res.ok) {
          setStatus("ready");
          clearInterval(ticker);
          clearInterval(poller);
          return;
        }
      } catch {
        // Backend not ready yet — keep polling
      }

      attempts++;
      if (attempts >= MAX_ATTEMPTS) {
        // After 40s show the app anyway — don't block forever
        setStatus("timeout");
        clearInterval(ticker);
        clearInterval(poller);
      }
    };

    // Poll every 2 seconds
    const poller = setInterval(tryConnect, 2000);
    tryConnect(); // immediate first attempt

    return () => {
      clearInterval(ticker);
      clearInterval(poller);
    };
  }, []);

  // Once ready or timed out, show the actual app
  if (status === "ready" || status === "timeout") {
    return <>{children}</>;
  }

  // Show professional loading screen while backend wakes up
  return (
    <div className="min-h-screen bg-[#0A0F1E] flex items-center justify-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col items-center gap-6 text-center px-6"
      >
        {/* Animated logo */}
        <motion.div
          animate={{ scale: [1, 1.08, 1] }}
          transition={{ duration: 1.6, repeat: Infinity }}
          className="w-20 h-20 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-2xl"
          style={{ boxShadow: "0 0 40px rgba(59,130,246,0.4)" }}
        >
          <Zap className="w-10 h-10 text-white" />
        </motion.div>

        {/* Title */}
        <div>
          <h1 className="text-2xl font-bold text-white tracking-tight">
            Edge AI Predictive Maintenance
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Industry 4.0 · NASA Turbofan · Dual-Head Transformer
          </p>
        </div>

        {/* Status row */}
        <div className="flex items-center gap-2 text-blue-400">
          <Activity className="w-4 h-4 animate-pulse" />
          <span className="text-sm">
            {elapsed < 5
              ? "Connecting to AI backend..."
              : elapsed < 15
              ? "Waking up edge inference engine..."
              : "Almost ready — starting ONNX runtime..."}
          </span>
        </div>

        {/* Progress bar */}
        <div className="w-64 h-1.5 bg-slate-800 rounded-full overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-blue-500 to-blue-400 rounded-full"
            animate={{ width: [`${Math.min(elapsed * 3, 90)}%`] }}
            transition={{ duration: 0.5 }}
          />
        </div>

        {/* Stats preview — shows what's coming */}
        <div className="grid grid-cols-3 gap-4 mt-2">
          {[
            { label: "Accuracy",  value: "98.82%" },
            { label: "Inference", value: "0.20ms" },
            { label: "AUC-ROC",   value: "0.997"  },
          ].map(({ label, value }) => (
            <div key={label} className="bg-slate-800/60 border border-slate-700/50 rounded-xl px-4 py-3">
              <p className="text-blue-400 font-bold text-lg">{value}</p>
              <p className="text-slate-500 text-xs">{label}</p>
            </div>
          ))}
        </div>

        <p className="text-slate-600 text-xs">
          Free tier backend — first load takes ~15 seconds
        </p>
      </motion.div>
    </div>
  );
}