import { useEffect, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

export default function CustomCursor() {
  const [clicking, setClicking] = useState(false);
  const [hovering, setHovering] = useState(false);
  const [visible, setVisible] = useState(false);

  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  // Main cursor - fast
  const springX = useSpring(mouseX, { stiffness: 900, damping: 30 });
  const springY = useSpring(mouseY, { stiffness: 900, damping: 30 });

  // Trail - slower
  const trailX = useSpring(mouseX, { stiffness: 120, damping: 18 });
  const trailY = useSpring(mouseY, { stiffness: 120, damping: 18 });

  useEffect(() => {
    // Hide default cursor
    document.body.style.cursor = "none";

    const move = (e: MouseEvent) => {
      mouseX.set(e.clientX);
      mouseY.set(e.clientY);
      setVisible(true);
    };

    const down  = () => setClicking(true);
    const up    = () => setClicking(false);
    const leave = () => setVisible(false);
    const enter = () => setVisible(true);

    const addHover    = () => setHovering(true);
    const removeHover = () => setHovering(false);

    window.addEventListener("mousemove", move);
    window.addEventListener("mousedown", down);
    window.addEventListener("mouseup", up);
    document.addEventListener("mouseleave", leave);
    document.addEventListener("mouseenter", enter);

    // Detect hover on interactive elements
    const els = document.querySelectorAll(
      "button, a, input, select, [role='button']"
    );
    els.forEach(el => {
      el.addEventListener("mouseenter", addHover);
      el.addEventListener("mouseleave", removeHover);
    });

    return () => {
      document.body.style.cursor = "auto";
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mousedown", down);
      window.removeEventListener("mouseup", up);
      document.removeEventListener("mouseleave", leave);
      document.removeEventListener("mouseenter", enter);
      els.forEach(el => {
        el.removeEventListener("mouseenter", addHover);
        el.removeEventListener("mouseleave", removeHover);
      });
    };
  }, [mouseX, mouseY]);

  if (!visible) return null;

  return (
    <>
      {/* ── TRAILING GLOW RING ──────────────────── */}
      <motion.div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          x: trailX,
          y: trailY,
          translateX: "-50%",
          translateY: "-50%",
          pointerEvents: "none",
          zIndex: 99997,
        }}
        animate={{
          scale: hovering ? 2.8 : clicking ? 0.6 : 1,
          opacity: hovering ? 0.5 : 0.2,
        }}
        transition={{ duration: 0.2 }}
      >
        <div style={{
          width: 48,
          height: 48,
          borderRadius: "50%",
          background: "radial-gradient(circle, #F59E0B40 0%, transparent 70%)",
          border: "1px solid #F59E0B40",
        }} />
      </motion.div>

      {/* ── OUTER RING ──────────────────────────── */}
      <motion.div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          x: trailX,
          y: trailY,
          translateX: "-50%",
          translateY: "-50%",
          pointerEvents: "none",
          zIndex: 99998,
        }}
        animate={{
          scale: hovering ? 2 : clicking ? 0.7 : 1,
          opacity: hovering ? 0.8 : 0.5,
          rotate: hovering ? 45 : 0,
        }}
        transition={{ duration: 0.15 }}
      >
        <div style={{
          width: 32,
          height: 32,
          borderRadius: "50%",
          border: `1.5px solid ${hovering ? "#22C55E" : clicking ? "#EF4444" : "#F59E0B"}`,
          boxShadow: hovering
            ? "0 0 12px #22C55E"
            : clicking
            ? "0 0 12px #EF4444"
            : "0 0 8px #F59E0B",
        }} />
      </motion.div>

      {/* ── LIGHTNING BOLT CURSOR ───────────────── */}
      <motion.div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          x: springX,
          y: springY,
          translateX: "-50%",
          translateY: "-50%",
          pointerEvents: "none",
          zIndex: 99999,
          userSelect: "none",
        }}
        animate={{
          scale: clicking ? 0.7 : hovering ? 1.4 : 1,
          rotate: clicking ? -20 : 0,
          filter: hovering
            ? "drop-shadow(0 0 8px #22C55E) drop-shadow(0 0 16px #22C55E80)"
            : clicking
            ? "drop-shadow(0 0 10px #EF4444) drop-shadow(0 0 20px #EF444480)"
            : "drop-shadow(0 0 6px #F59E0B) drop-shadow(0 0 12px #F59E0B80)",
        }}
        transition={{ duration: 0.1 }}
      >
        {/* Lightning Bolt SVG */}
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <motion.path
            d="M13 2L4.5 13.5H11L10 22L19.5 10.5H13L13 2Z"
            fill={
              hovering ? "#22C55E"
              : clicking ? "#EF4444"
              : "#F59E0B"
            }
            stroke={
              hovering ? "#22C55E"
              : clicking ? "#EF4444"
              : "#F59E0B"
            }
            strokeWidth="1"
            strokeLinejoin="round"
            animate={{
              fill: hovering ? "#22C55E" : clicking ? "#EF4444" : "#F59E0B",
            }}
            transition={{ duration: 0.1 }}
          />
        </svg>
      </motion.div>

      {/* ── CLICK SPARK EFFECT ──────────────────── */}
      {clicking && (
        <motion.div
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            x: springX,
            y: springY,
            translateX: "-50%",
            translateY: "-50%",
            pointerEvents: "none",
            zIndex: 99996,
          }}
          initial={{ scale: 0, opacity: 1 }}
          animate={{ scale: 3, opacity: 0 }}
          transition={{ duration: 0.4, ease: "easeOut" }}
        >
          <div style={{
            width: 20,
            height: 20,
            borderRadius: "50%",
            background: "radial-gradient(circle, #F59E0B 0%, transparent 70%)",
          }} />
        </motion.div>
      )}
    </>
  );
}