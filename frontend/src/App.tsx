import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { AppLayout } from "./components/AppLayout";
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

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout />}>
            <Route path="/" element={<LiveMonitor />} />
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
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
