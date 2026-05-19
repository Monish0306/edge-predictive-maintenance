import { motion } from "framer-motion";
import { ReactNode } from "react";

interface Props { title: string; subtitle?: string; live?: boolean; children?: ReactNode; }

export const PageHeader = ({ title, subtitle, live, children }: Props) => (
  <motion.div
    initial={{ opacity: 0, y: -10 }}
    animate={{ opacity: 1, y: 0 }}
    className="flex flex-wrap items-center justify-between gap-4 pb-6 border-b border-border mb-6"
  >
    <div>
      <h1 className="text-3xl font-bold tracking-tight flex items-center gap-3">
        {live && <span className="inline-flex items-center gap-1.5 text-sm font-semibold text-danger">
          <span className="w-2.5 h-2.5 rounded-full bg-danger pulse-dot" />LIVE
        </span>}
        <span>{title}</span>
      </h1>
      {subtitle && <p className="text-sm text-muted-foreground mt-1.5">{subtitle}</p>}
    </div>
    {children && <div className="flex items-center gap-2 flex-wrap">{children}</div>}
  </motion.div>
);
