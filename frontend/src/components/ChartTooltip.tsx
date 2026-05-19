interface Props { active?: boolean; payload?: any[]; label?: any; }
export const DarkTooltip = ({ active, payload, label }: Props) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-lg border border-border bg-card/95 backdrop-blur px-3 py-2 shadow-card-soft">
      {label !== undefined && <div className="text-xs text-muted-foreground mb-1">{label}</div>}
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color || p.fill }} />
          <span className="text-foreground font-medium">{p.name}:</span>
          <span className="text-foreground font-mono">{typeof p.value === "number" ? p.value.toFixed(p.value < 1 ? 3 : 1) : p.value}</span>
        </div>
      ))}
    </div>
  );
};
