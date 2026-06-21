import { Outlet, useLocation } from "react-router-dom";
import { AppSidebar } from "./AppSidebar";
import { AnimatePresence, motion } from "framer-motion";
import NotificationBell from "./NotificationBell";
import AlertToastContainer from "./AlertToast";
import ChatbotWidget from "./ChatbotWidget";

export const AppLayout = () => {
  const location = useLocation();
  return (
    <div className="min-h-screen flex w-full bg-background text-foreground">
      <AppSidebar />
      <div className="flex-1 flex flex-col min-w-0">

        {/* Top bar with bell */}
        <div className="h-12 border-b border-border flex items-center justify-end px-6 bg-background/80 backdrop-blur-xl sticky top-0 z-40">
          <NotificationBell />
        </div>

        <main className="flex-1 relative">
          <div className="absolute inset-0 grid-bg opacity-20 pointer-events-none" />
          <div className="relative p-6 lg:p-8 max-w-[1600px] mx-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: 0.25 }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>

      {/* Toast notifications */}
      <AlertToastContainer />

      {/* ⭐ RAG Chatbot - Floating bottom-right */}
      <ChatbotWidget engineId={1} mode="normal" />
    </div>
  );
};