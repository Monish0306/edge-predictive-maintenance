import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { AnimatedNumber } from "./AnimatedNumber";

interface Props {
  label: string;
  value: number | string;
  decimals?: number;
  suffix?: string;
  prefix?: string;
  icon?: LucideIcon;
  color?: "primary" | "success" | "warning" | "danger" | "critical";
  delay?: number;
  hint?: string;
  children?: React.ReactNode;
}

const colorMap = {
  primary: { text: "text-primary", bg: "bg-primary/10", border: "border-primary/30", glow: "shadow-[0_0_25px_hsl(217_91%_60%/0.2)]" },
  success: { text: "text-success", bg: "bg-success/10", border: "border-success/30", glow: "shadow-[0_0_25px_hsl(142_71%_45%/0.2)]" },
  warning: { text: "text-warning", bg: "bg-warning/10", border: "border-warning/30", glow: "shadow-[0_0_25px_hsl(45_93%_47%/0.2)]" },
  danger: { text: "text-danger", bg: "bg-danger/10", border: "border-danger/30", glow: "shadow-[0_0_25px_hsl(0_84%_60%/0.25)]" },
  critical: { text: "text-critical", bg: "bg-critical/10", border: "border-critical/30", glow: "shadow-[0_0_25px_hsl(271_91%_65%/0.25)]" },
};

export const MetricCard = ({ label, value, decimals = 0, suffix, prefix, icon: Icon, color = "primary", delay = 0, hint, children }: Props) => {
  const c = colorMap[color];
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4, ease: "easeOut" }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className={cn("relative overflow-hidden rounded-xl border bg-card p-5 transition-shadow", c.border, "hover:" + c.glow)}
    >
      <div className="absolute inset-0 bg-gradient-card opacity-50 pointer-events-none" />
      <div className="relative flex items-start justify-between mb-3">
        <span className="text-xs uppercase tracking-wider text-muted-foreground font-medium">{label}</span>
        {Icon && (
          <div className={cn("w-9 h-9 rounded-lg flex items-center justify-center", c.bg)}>
            <Icon className={cn("w-4 h-4", c.text)} />
          </div>
        )}
      </div>
      <div className="relative">
        <div className={cn("text-3xl font-bold tracking-tight", c.text)}>
          {typeof value === "number" ? (
            <AnimatedNumber value={value} decimals={decimals} suffix={suffix} prefix={prefix} />
          ) : (
            <span>{prefix}{value}{suffix}</span>
          )}
        </div>
        {hint && <div className="text-xs text-muted-foreground mt-1">{hint}</div>}
        {children}
      </div>
    </motion.div>
  );
};
