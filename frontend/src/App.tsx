import { useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppLayout } from "./components/AppLayout";
import CustomCursor from "./components/CustomCursor";
import { HeroGeometric } from "./components/ui/shape-landing-hero";
import { motion, AnimatePresence } from "framer-motion";
import LiveMonitor from "./pages/LiveMonitor";
import FleetOverview from "./pages/FleetOverview";
import Analytics from "./pages/Analytics";
import SensorHeatmap from "./pages/SensorHeatmap";
import FailureTimeline from "./pages/FailureTimeline";
import Reports from "./pages/Reports";
import AgentLog from "./pages/AgentLog";
import DatasetStats from "./pages/DatasetStats";
import CostSavings from "./pages/CostSavings";
import ModelInfo from "./pages/ModelInfo";
import NotFound from "./pages/NotFound";
import DigitalTwin from "./pages/DigitalTwin";
import Notifications from "./pages/Notifications";

const queryClient = new QueryClient();

const App = () => {
  const [entered, setEntered] = useState(false);

  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <CustomCursor />

        <AnimatePresence mode="wait">
          {!entered ? (
            // ── LANDING PAGE ──────────────────────────
            <motion.div
              key="landing"
              initial={{ opacity: 1 }}
              exit={{ opacity: 0, scale: 1.05 }}
              transition={{ duration: 0.6, ease: "easeInOut" }}
            >
              <HeroGeometric onEnter={() => setEntered(true)} />
            </motion.div>

          ) : (
            // ── MAIN DASHBOARD ────────────────────────
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, ease: "easeInOut" }}
            >
              <BrowserRouter>
                <Routes>
                  <Route element={<AppLayout />}>
                    <Route path="/" element={<LiveMonitor />} />
                    <Route path="/notifications" element={<Notifications />} />
                    <Route path="/digital-twin" element={<DigitalTwin />} />
                    <Route path="/fleet" element={<FleetOverview />} />
                    <Route path="/analytics" element={<Analytics />} />
                    <Route path="/heatmap" element={<SensorHeatmap />} />
                    <Route path="/timeline" element={<FailureTimeline />} />
                    <Route path="/reports" element={<Reports />} />
                    <Route path="/agent-log" element={<AgentLog />} />
                    <Route path="/datasets" element={<DatasetStats />} />
                    <Route path="/savings" element={<CostSavings />} />
                    <Route path="/model" element={<ModelInfo />} />
                  </Route>
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