import { useState, useEffect } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppLayout } from "./components/AppLayout";
import CustomCursor from "./components/CustomCursor";
import { HeroGeometric } from "./components/ui/shape-landing-hero";
import { motion, AnimatePresence } from "framer-motion";
import { useKeepAlive } from "./hooks/useKeepAlive";
import { AppLoader } from "./components/LoadingScreen";

// ── Page imports ──────────────────────────────────────────────────────────────
import LiveMonitor     from "./pages/LiveMonitor";
import FleetOverview   from "./pages/FleetOverview";
import Analytics       from "./pages/Analytics";
import SensorHeatmap   from "./pages/SensorHeatmap";
import FailureTimeline from "./pages/FailureTimeline";
import Reports         from "./pages/Reports";
import AgentLog        from "./pages/AgentLog";
import DatasetStats    from "./pages/DatasetStats";
import CostSavings     from "./pages/CostSavings";
import ModelInfo       from "./pages/ModelInfo";
import NotFound        from "./pages/NotFound";
import DigitalTwin     from "./pages/DigitalTwin";
import Notifications   from "./pages/Notifications";
import OEEDashboard    from "./pages/OEEDashboard";
import PlantMap        from "./pages/PlantMap";

// ── React Query client with optimized defaults ────────────────────────────────
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Cache for 30s — reduces redundant API calls to Render backend
      staleTime: 30 * 1000,
      // Retry failed requests twice before showing error
      retry: 2,
      // Don't refetch on window focus — prevents chart flicker when switching tabs
      refetchOnWindowFocus: false,
    },
  },
});

// ── App ───────────────────────────────────────────────────────────────────────
const App = () => {
  const [entered, setEntered] = useState(false);

  // ── Keep Render backend warm ──────────────────────────────────────────────
  // Fires immediately on app load while user is still reading landing page.
  // Backend is fully warm before they click "Launch Dashboard".
  // Re-pings every 10 minutes to prevent Render free-tier 15-min spin-down.
  useKeepAlive();

  // ── Preload critical chunks while user is on landing page ─────────────────
  // Silently imports the 3 most-visited pages in background.
  // When user clicks Launch Dashboard, LiveMonitor renders instantly
  // instead of waiting for JS chunk download.
  useEffect(() => {
    if (!entered) {
      const preload = async () => {
        await Promise.allSettled([
          import("./pages/LiveMonitor"),
          import("./pages/FleetOverview"),
          import("./pages/Analytics"),
        ]);
      };
      // 2s delay — let landing animation finish first
      const timer = setTimeout(preload, 2000);
      return () => clearTimeout(timer);
    }
  }, [entered]);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <CustomCursor />

        <AnimatePresence mode="wait">
          {!entered ? (

            // ── LANDING PAGE ────────────────────────────────────────────────
            // AppLoader polls GET /health every 2s.
            // Shows professional loading screen with model stats while
            // Render backend wakes from cold start (~15s first visit).
            // Once /health returns OK → shows HeroGeometric landing page.
            // User never sees blank charts or broken API calls.
            <motion.div
              key="landing"
              initial={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              transition={{ duration: 0.6, ease: "easeInOut" }}
            >
              <AppLoader>
                <HeroGeometric onEnter={() => setEntered(true)} />
              </AppLoader>
            </motion.div>

          ) : (

            // ── MAIN DASHBOARD ──────────────────────────────────────────────
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, ease: "easeInOut" }}
            >
              <BrowserRouter>
                <Routes>
                  <Route element={<AppLayout />}>

                    {/* ── Core Monitoring ── */}
                    <Route path="/"              element={<LiveMonitor />}     />
                    <Route path="/fleet"         element={<FleetOverview />}   />
                    <Route path="/digital-twin"  element={<DigitalTwin />}     />
                    <Route path="/plant-map"     element={<PlantMap />}        />

                    {/* ── AI & Analytics ── */}
                    <Route path="/analytics"     element={<Analytics />}       />
                    <Route path="/heatmap"       element={<SensorHeatmap />}   />
                    <Route path="/timeline"      element={<FailureTimeline />} />
                    <Route path="/model"         element={<ModelInfo />}       />
                    <Route path="/datasets"      element={<DatasetStats />}    />

                    {/* ── Operations ── */}
                    <Route path="/oee"           element={<OEEDashboard />}    />
                    <Route path="/reports"       element={<Reports />}         />
                    <Route path="/savings"       element={<CostSavings />}     />

                    {/* ── Alerts & Logs ── */}
                    <Route path="/notifications" element={<Notifications />}   />
                    <Route path="/agent-log"     element={<AgentLog />}        />

                  </Route>

                  {/* 404 catch-all */}
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </BrowserRouter>
            </motion.div>

          )}
        </AnimatePresence>

      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;