"use client";
import { motion } from "framer-motion";
import { Circle, Zap, Activity, Cpu } from "lucide-react";
import { cn } from "@/lib/utils";

function ElegantShape({
  className, delay = 0, width = 400, height = 100,
  rotate = 0, gradient = "from-white/[0.08]",
}: {
  className?: string; delay?: number; width?: number;
  height?: number; rotate?: number; gradient?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -150, rotate: rotate - 15 }}
      animate={{ opacity: 1, y: 0, rotate: rotate }}
      transition={{
        duration: 2.4, delay,
        ease: [0.23, 0.86, 0.39, 0.96],
        opacity: { duration: 1.2 },
      }}
      className={cn("absolute", className)}
    >
      <motion.div
        animate={{ y: [0, 15, 0] }}
        transition={{ duration: 12, repeat: Infinity, ease: "easeInOut" }}
        style={{ width, height }}
        className="relative"
      >
        <div className={cn(
          "absolute inset-0 rounded-full",
          "bg-gradient-to-r to-transparent", gradient,
          "backdrop-blur-[2px] border-2 border-white/[0.15]",
          "shadow-[0_8px_32px_0_rgba(255,255,255,0.1)]",
          "after:absolute after:inset-0 after:rounded-full",
          "after:bg-[radial-gradient(circle_at_50%_50%,rgba(255,255,255,0.2),transparent_70%)]"
        )} />
      </motion.div>
    </motion.div>
  );
}

export function HeroGeometric({ onEnter }: { onEnter: () => void }) {
  const fadeUpVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: (i: number) => ({
      opacity: 1, y: 0,
      transition: {
        duration: 1, delay: 0.5 + i * 0.2,
        ease: [0.25, 0.4, 0.25, 1],
      },
    }),
  };

  return (
    <div className="relative min-h-screen w-full flex items-center justify-center overflow-hidden bg-[#030303]">

      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-br from-blue-500/[0.05] via-transparent to-amber-500/[0.05] blur-3xl" />

      {/* Animated shapes */}
      <div className="absolute inset-0 overflow-hidden">
        <ElegantShape delay={0.3} width={600} height={140} rotate={12}
          gradient="from-blue-500/[0.15]"
          className="left-[-10%] md:left-[-5%] top-[15%] md:top-[20%]"
        />
        <ElegantShape delay={0.5} width={500} height={120} rotate={-15}
          gradient="from-amber-500/[0.15]"
          className="right-[-5%] md:right-[0%] top-[70%] md:top-[75%]"
        />
        <ElegantShape delay={0.4} width={300} height={80} rotate={-8}
          gradient="from-cyan-500/[0.15]"
          className="left-[5%] md:left-[10%] bottom-[5%] md:bottom-[10%]"
        />
        <ElegantShape delay={0.6} width={200} height={60} rotate={20}
          gradient="from-yellow-500/[0.15]"
          className="right-[15%] md:right-[20%] top-[10%] md:top-[15%]"
        />
        <ElegantShape delay={0.7} width={150} height={40} rotate={-25}
          gradient="from-blue-400/[0.15]"
          className="left-[20%] md:left-[25%] top-[5%] md:top-[10%]"
        />
      </div>

      {/* Main content */}
      <div className="relative z-10 container mx-auto px-4 md:px-6">
        <div className="max-w-4xl mx-auto text-center">

          {/* Badge */}
          <motion.div
            custom={0}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/[0.03] border border-white/[0.08] mb-8 md:mb-12"
          >
            <Circle className="h-2 w-2 fill-amber-500/80" />
            <span className="text-sm text-white/60 tracking-wide font-medium">
              Industry 4.0 — Edge AI System
            </span>
          </motion.div>

          {/* Title */}
          <motion.div
            custom={1}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
          >
            <h1 className="text-4xl sm:text-6xl md:text-7xl font-bold mb-6 tracking-tight">
              <span className="bg-clip-text text-transparent bg-gradient-to-b from-white to-white/80">
                Edge AI Predictive
              </span>
              <br />
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-300 via-white/90 to-amber-300">
                Maintenance System
              </span>
            </h1>
          </motion.div>

          {/* Description */}
          <motion.div
            custom={2}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
          >
            <p className="text-base sm:text-lg md:text-xl text-white/40 mb-4 leading-relaxed font-light tracking-wide max-w-2xl mx-auto">
              Predicts equipment failures <span className="text-amber-400/80">days in advance</span> using
              Transformer AI. Deployed on edge devices with <span className="text-blue-400/80">0.20ms inference</span> — no cloud required.
            </p>
          </motion.div>

          {/* Stats row */}
          <motion.div
            custom={3}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
            className="flex items-center justify-center gap-8 mb-10"
          >
            {[
              { label: "Accuracy", value: "98.82%", color: "text-green-400" },
              { label: "AUC-ROC", value: "0.997", color: "text-blue-400" },
              { label: "Inference", value: "0.20ms", color: "text-amber-400" },
              { label: "Engines", value: "709", color: "text-purple-400" },
            ].map((stat, i) => (
              <div key={i} className="text-center">
                <div className={`text-2xl font-bold font-mono ${stat.color}`}>
                  {stat.value}
                </div>
                <div className="text-xs text-white/30 uppercase tracking-widest mt-1">
                  {stat.label}
                </div>
              </div>
            ))}
          </motion.div>

          {/* Enter Button */}
          <motion.div
            custom={4}
            variants={fadeUpVariants}
            initial="hidden"
            animate="visible"
          >
            <motion.button
              onClick={onEnter}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.97 }}
              className="group relative inline-flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-white text-lg overflow-hidden"
              style={{
                background: "linear-gradient(135deg, #3B82F6, #2563EB)",
                boxShadow: "0 0 30px #3B82F640, 0 4px 15px #0000004D",
              }}
            >
              {/* Animated shimmer */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent"
                animate={{ x: ["-100%", "100%"] }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              />

              <Zap className="w-5 h-5 text-amber-300" />
              <span>Launch Dashboard</span>
              <Activity className="w-5 h-5 opacity-60 group-hover:opacity-100 transition-opacity" />
            </motion.button>

            <p className="text-xs text-white/20 mt-4 tracking-wide">
              NASA Turbofan Dataset • PyTorch Transformer • ONNX Runtime
            </p>
          </motion.div>

        </div>
      </div>

      {/* Bottom/top fade */}
      <div className="absolute inset-0 bg-gradient-to-t from-[#030303] via-transparent to-[#030303]/80 pointer-events-none" />

      {/* Floating tech badges */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 2 }}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 flex items-center gap-3"
      >
        {["PyTorch", "ONNX", "FastAPI", "React", "MLflow"].map((tech, i) => (
          <motion.span
            key={tech}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 2 + i * 0.1 }}
            className="px-3 py-1 rounded-full text-xs text-white/30 border border-white/[0.06] bg-white/[0.02] font-mono"
          >
            {tech}
          </motion.span>
        ))}
      </motion.div>

    </div>
  );
}