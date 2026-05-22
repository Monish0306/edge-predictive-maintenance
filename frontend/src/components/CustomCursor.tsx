import { useEffect, useState } from "react";
import { motion, useMotionValue, useSpring } from "framer-motion";

export default function CustomCursor() {
  const [clicking, setClicking]   = useState(false);
  const [hovering, setHovering]   = useState(false);
  const [visible, setVisible]     = useState(false);

  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);

  const springX = useSpring(mouseX, { stiffness: 900, damping: 30 });
  const springY = useSpring(mouseY, { stiffness: 900, damping: 30 });

  const trailX = useSpring(mouseX, { stiffness: 120, damping: 18 });
  const trailY = useSpring(mouseY, { stiffness: 120, damping: 18 });

  useEffect(() => {
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
      {/* ── TRAILING GLOW ───────────────────────── */}
      <motion.div
        style={{
          position: "fixed",
          top: 0, left: 0,
          x: trailX, y: trailY,
          translateX: "-50%",
          translateY: "-50%",
          pointerEvents: "none",
          zIndex: 99996,
        }}
        animate={{
          scale: hovering ? 3 : clicking ? 0.5 : 1,
          opacity: hovering ? 0.4 : 0.15,
        }}
        transition={{ duration: 0.2 }}
      >
        <div style={{
          width: 60,
          height: 60,
          borderRadius: "50%",
          background: hovering
            ? "radial-gradient(circle, #22C55E50 0%, transparent 70%)"
            : clicking
            ? "radial-gradient(circle, #EF444450 0%, transparent 70%)"
            : "radial-gradient(circle, #F59E0B40 0%, transparent 70%)",
        }} />
      </motion.div>

      {/* ── OUTER RING ──────────────────────────── */}
      <motion.div
        style={{
          position: "fixed",
          top: 0, left: 0,
          x: trailX, y: trailY,
          translateX: "-50%",
          translateY: "-50%",
          pointerEvents: "none",
          zIndex: 99997,
        }}
        animate={{
          scale: hovering ? 2.2 : clicking ? 0.6 : 1,
          opacity: hovering ? 0.9 : 0.4,
          rotate: clicking ? 45 : 0,
        }}
        transition={{ duration: 0.15 }}
      >
        <div style={{
          width: 36,
          height: 36,
          borderRadius: "50%",
          border: `1.5px solid ${
            hovering ? "#22C55E"
            : clicking ? "#EF4444"
            : "#F59E0B"
          }`,
          boxShadow: hovering
            ? "0 0 15px #22C55E, inset 0 0 8px #22C55E30"
            : clicking
            ? "0 0 15px #EF4444, inset 0 0 8px #EF444430"
            : "0 0 10px #F59E0B, inset 0 0 6px #F59E0B20",
        }} />
      </motion.div>

      {/* ── ⚡ EMOJI CURSOR ──────────────────────── */}
      <motion.div
        style={{
          position: "fixed",
          top: 0, left: 0,
          x: springX, y: springY,
          translateX: "-20%",
          translateY: "-20%",
          pointerEvents: "none",
          zIndex: 99999,
          userSelect: "none",
          lineHeight: 1,
        }}
        animate={{
          scale: clicking ? 0.6 : hovering ? 1.5 : 1,
          rotate: clicking ? -20 : hovering ? 10 : 0,
          filter: hovering
            ? "drop-shadow(0 0 8px #22C55E) brightness(1.3)"
            : clicking
            ? "drop-shadow(0 0 10px #EF4444) brightness(1.5)"
            : "drop-shadow(0 0 6px #F59E0B) brightness(1.1)",
        }}
        transition={{ duration: 0.1 }}
      >
        <span style={{
          fontSize: clicking ? "20px" : hovering ? "28px" : "22px",
          display: "block",
          transition: "font-size 0.1s ease",
        }}>
          ⚡
        </span>
      </motion.div>

      {/* ── CLICK SPARK ─────────────────────────── */}
      {clicking && (
        <motion.div
          style={{
            position: "fixed",
            top: 0, left: 0,
            x: springX, y: springY,
            translateX: "-50%",
            translateY: "-50%",
            pointerEvents: "none",
            zIndex: 99998,
          }}
          initial={{ scale: 0, opacity: 1 }}
          animate={{ scale: 4, opacity: 0 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <div style={{
            width: 16,
            height: 16,
            borderRadius: "50%",
            background: "radial-gradient(circle, #FCD34D 0%, #F59E0B 50%, transparent 70%)",
          }} />
        </motion.div>
      )}
    </>
  );
}