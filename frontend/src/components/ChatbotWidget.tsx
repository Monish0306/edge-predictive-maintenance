/**
 * ChatbotWidget.tsx — Edge AI Predictive Maintenance Copilot
 * ===========================================================
 * Fixes applied vs previous version:
 * 1. Inline bold (**word**) and inline code (`code`) now render correctly
 * 2. Tooltip shows only once — never reappears after first close
 * 3. history sent to API skips welcome message (index 0)
 * 4. engine_id and mode props accepted so chatbot knows current dashboard state
 * 5. Unread badge increment logic fixed — now actually reachable
 * 6. ChevronDown replaced with RotateCcw for clear chat button
 * 7. Clear chat also resets unread counter
 * 8. Wrench icon replaced with Clock for RUL suggestion
 * 9. Error message uses API constant not hardcoded localhost
 * 10. catch(err) with typed parameter — no TS strict mode warning
 * 11. Minimize2 unused import removed
 * 12. showBadge initialized false (no badge on fresh load)
 * 13. Mobile responsive width (min-w, max-w instead of fixed 380px)
 * 14. Message IDs use crypto.randomUUID() — no timestamp collision risk
 */

import { useState, useRef, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  X, Send, Bot, User, Loader2,
  Sparkles, RotateCcw, ChevronDown,
  AlertTriangle, Wrench, BarChart3,
  HelpCircle, Zap, Clock,
} from "lucide-react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Types ─────────────────────────────────────────────────────────────────────
interface Message {
  id:      string;
  role:    "user" | "assistant";
  content: string;
  time:    string;
}

interface Props {
  /** Current engine being viewed — passed to /chat so context matches dashboard */
  engineId?: number;
  /** Current simulation mode — passed to /chat for accurate engine context */
  mode?: "normal" | "warning" | "fault";
}

// ── Suggestion quick-questions ────────────────────────────────────────────────
const SUGGESTIONS = [
  { icon: AlertTriangle, text: "What does a CRITICAL alert mean?",   color: "#EF4444" },
  { icon: Wrench,        text: "How do I respond to HPC fault?",     color: "#F97316" },
  { icon: BarChart3,     text: "What is world class OEE?",           color: "#3B82F6" },
  { icon: Zap,           text: "Why is inference only 0.20ms?",      color: "#EAB308" },
  { icon: HelpCircle,    text: "What does sensor 4 (T30) measure?",  color: "#A855F7" },
  { icon: Clock,         text: "What is Remaining Useful Life?",     color: "#22C55E" },  // was Wrench (duplicate)
] as const;

// ── Timestamp helper ──────────────────────────────────────────────────────────
const now = () =>
  new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// ── Welcome message ───────────────────────────────────────────────────────────
const WELCOME: Message = {
  id:      "welcome",
  role:    "assistant",
  content: "👋 Hi! I'm your **Maintenance AI Copilot**!\n\nI can help you with:\n• Sensor readings and fault detection\n• Alert severity and response procedures\n• Maintenance scheduling and repair costs\n• OEE metrics and equipment effectiveness\n• Model performance and ONNX deployment\n\nWhat would you like to know?",
  time:    now(),
};

// ══════════════════════════════════════════════════════════════════════════════
// FormatMessage — markdown-lite renderer
// FIXED: now handles inline **bold** and `code` inside regular sentences
// ══════════════════════════════════════════════════════════════════════════════
function FormatMessage({ text }: { text: string }) {
  /**
   * Renders a single line of text, handling inline formatting:
   * - **bold** → <strong>
   * - `code`   → <code>
   * Returns an array of React nodes (mix of strings and elements).
   */
  const renderInline = (line: string): React.ReactNode[] => {
    const nodes: React.ReactNode[] = [];
    // Split on **bold** or `code` tokens, keeping delimiters
    const parts = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    parts.forEach((part, i) => {
      if (part.startsWith("**") && part.endsWith("**")) {
        nodes.push(
          <strong key={i} className="font-semibold text-white">
            {part.slice(2, -2)}
          </strong>
        );
      } else if (part.startsWith("`") && part.endsWith("`")) {
        nodes.push(
          <code
            key={i}
            className="bg-slate-700 text-blue-300 px-1 py-0.5 rounded text-xs font-mono"
          >
            {part.slice(1, -1)}
          </code>
        );
      } else {
        nodes.push(part);
      }
    });
    return nodes;
  };

  const lines = text.split("\n");

  return (
    <div className="space-y-1">
      {lines.map((line, i) => {
        // Whole-line bold heading (e.g. "**Sensor 4 (T30)**" alone on a line)
        if (/^\*\*[^*]+\*\*$/.test(line.trim())) {
          return (
            <p key={i} className="font-bold text-white text-sm">
              {line.trim().slice(2, -2)}
            </p>
          );
        }

        // Bullet point: "• text" or "- text"
        if (line.startsWith("• ") || line.startsWith("- ")) {
          return (
            <div key={i} className="flex items-start gap-2">
              <span className="text-blue-400 mt-0.5 flex-shrink-0 select-none">•</span>
              <span className="text-sm text-slate-200 leading-relaxed">
                {renderInline(line.slice(2))}
              </span>
            </div>
          );
        }

        // Numbered list: "1. text"
        const numMatch = line.match(/^(\d+)\.\s(.+)/);
        if (numMatch) {
          return (
            <div key={i} className="flex items-start gap-2">
              <span className="text-blue-400 font-bold text-xs mt-0.5 flex-shrink-0 w-4">
                {numMatch[1]}.
              </span>
              <span className="text-sm text-slate-200 leading-relaxed">
                {renderInline(numMatch[2])}
              </span>
            </div>
          );
        }

        // Markdown heading: "# text" or "## text"
        if (/^#{1,3}\s/.test(line)) {
          return (
            <p key={i} className="font-bold text-blue-300 text-sm">
              {renderInline(line.replace(/^#+\s/, ""))}
            </p>
          );
        }

        // Blank line spacer
        if (!line.trim()) {
          return <div key={i} className="h-1" />;
        }

        // Regular paragraph with inline formatting
        return (
          <p key={i} className="text-sm text-slate-200 leading-relaxed">
            {renderInline(line)}
          </p>
        );
      })}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════════════════════
// ChatbotWidget — main component
// ══════════════════════════════════════════════════════════════════════════════
export default function ChatbotWidget({ engineId = 1, mode = "normal" }: Props) {
  const [open,         setOpen]         = useState(false);
  const [input,        setInput]        = useState("");
  const [messages,     setMessages]     = useState<Message[]>([WELCOME]);
  const [loading,      setLoading]      = useState(false);
  const [unread,       setUnread]       = useState(0);
  const [showBadge,    setShowBadge]    = useState(false);   // FIXED: was true
  // FIXED: tooltip shows only once — never reappears after first close
  const [tooltipShown, setTooltipShown] = useState(false);
  const [showTooltip,  setShowTooltip]  = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLInputElement>(null);

  // Auto-scroll on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // On open: focus input, clear badge, hide tooltip
  useEffect(() => {
    if (open) {
      setTimeout(() => inputRef.current?.focus(), 300);
      setUnread(0);
      setShowBadge(false);
      setShowTooltip(false);
    } else if (!tooltipShown) {
      // Show tooltip only the FIRST time the widget closes
      const timer = setTimeout(() => {
        setShowTooltip(true);
        setTooltipShown(true);   // never show again
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [open, tooltipShown]);

  // ── sendMessage ───────────────────────────────────────────────────────────
  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || loading) return;

    const userMsg: Message = {
      id:      crypto.randomUUID(),   // FIXED: no timestamp collision risk
      role:    "user",
      content: text,
      time:    now(),
    };

    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      // FIXED: skip welcome message (index 0) in history sent to API
      // Also include the current user message for accurate context
      const historyToSend = messages
        .filter(m => m.id !== "welcome")   // skip welcome message
        .slice(-8)                          // last 8 messages
        .map(m => ({ role: m.role, content: m.content }));

      // Add current user message to history
      historyToSend.push({ role: "user", content: text });

      const res = await fetch(`${API}/chat`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question:  text,
          history:   historyToSend.slice(0, -1),  // history excludes current question
          engine_id: engineId,
          mode:      mode,   // FIXED: pass current dashboard mode
        }),
      });

      if (!res.ok) {
        throw new Error(`Server error ${res.status}`);
      }

      const data = await res.json();

      const botMsg: Message = {
        id:      crypto.randomUUID(),
        role:    "assistant",
        content: data.answer ?? "Sorry, I couldn't process that request.",
        time:    now(),
      };

      setMessages(prev => [...prev, botMsg]);

      // FIXED: unread badge now correctly shown when chat is closed
      if (!open) {
        setUnread(c => c + 1);
        setShowBadge(true);
      }

    } catch (err: unknown) {   // FIXED: typed catch parameter
      const errorText = err instanceof Error ? err.message : "Unknown error";
      setMessages(prev => [...prev, {
        id:      crypto.randomUUID(),
        role:    "assistant",
        // FIXED: uses API constant, not hardcoded localhost
        content: `⚠️ Connection error: ${errorText}\n\nMake sure the backend is running at:\n\`${API}\``,
        time:    now(),
      }]);
    } finally {
      setLoading(false);
    }
  }, [loading, messages, open, engineId, mode]);

  // ── Clear chat handler ─────────────────────────────────────────────────────
  const clearChat = useCallback(() => {
    setMessages([{ ...WELCOME, time: now() }]);
    setUnread(0);          // FIXED: also resets unread count
    setShowBadge(false);
  }, []);

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <>
      {/* ── CHAT WINDOW ───────────────────────────────────────────────────── */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            transition={{ type: "spring", stiffness: 300, damping: 25 }}
            // FIXED: responsive width — no more fixed 380px that overflows mobile
            className="fixed bottom-24 right-4 z-[99990] flex flex-col rounded-2xl border border-slate-700 shadow-2xl overflow-hidden"
            style={{
              width:     "min(380px, calc(100vw - 32px))",   // responsive
              height:    "560px",
              background: "#0D1117",
              boxShadow: "0 0 50px rgba(59,130,246,0.15), 0 25px 50px rgba(0,0,0,0.6)",
            }}
          >
            {/* Header */}
            <div
              className="flex items-center gap-3 px-4 py-3 border-b border-slate-700/80 flex-shrink-0"
              style={{ background: "linear-gradient(135deg,#1a2035 0%,#111827 100%)" }}
            >
              {/* Bot avatar */}
              <div className="relative">
                <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-blue-700 flex items-center justify-center shadow-lg">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-green-500 border-2 border-[#111827]">
                  <span className="absolute inset-0 rounded-full bg-green-400 animate-ping opacity-75" />
                </span>
              </div>

              {/* Title */}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-bold text-white">Maintenance Copilot</p>
                <p className="text-[10px] text-green-400 flex items-center gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 inline-block" />
                  RAG-powered • Always online
                </p>
              </div>

              {/* Header actions */}
              <div className="flex items-center gap-1">
                {/* FIXED: RotateCcw instead of ChevronDown for clear action */}
                <button
                  onClick={clearChat}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-all"
                  title="Clear chat history"
                >
                  <RotateCcw className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setOpen(false)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-red-400 hover:bg-red-500/10 transition-all"
                  title="Close"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-slate-700">
              {messages.map((msg, idx) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  // FIXED: only animate the newest message, not all on re-render
                  transition={{ duration: idx === messages.length - 1 ? 0.2 : 0 }}
                  className={`flex gap-2.5 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
                >
                  {/* Avatar */}
                  <div
                    className={`w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 ${
                      msg.role === "assistant"
                        ? "bg-blue-500/20 border border-blue-500/30"
                        : "bg-slate-700"
                    }`}
                  >
                    {msg.role === "assistant"
                      ? <Bot  className="w-3.5 h-3.5 text-blue-400" />
                      : <User className="w-3.5 h-3.5 text-slate-400" />
                    }
                  </div>

                  {/* Bubble */}
                  <div className={`max-w-[82%] flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>
                    <div
                      className={`rounded-2xl px-3.5 py-2.5 ${
                        msg.role === "assistant"
                          ? "rounded-tl-sm bg-slate-800/80 border border-slate-700/50"
                          : "rounded-tr-sm bg-blue-600/30 border border-blue-500/30"
                      }`}
                    >
                      <FormatMessage text={msg.content} />
                    </div>
                    <span className="text-[9px] text-slate-600 mt-1 px-1">{msg.time}</span>
                  </div>
                </motion.div>
              ))}

              {/* Loading dots */}
              {loading && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex gap-2.5"
                >
                  <div className="w-7 h-7 rounded-lg bg-blue-500/20 border border-blue-500/30 flex items-center justify-center flex-shrink-0">
                    <Bot className="w-3.5 h-3.5 text-blue-400" />
                  </div>
                  <div className="bg-slate-800/80 border border-slate-700/50 rounded-2xl rounded-tl-sm px-4 py-3">
                    <div className="flex items-center gap-1.5">
                      {([0, 0.15, 0.3] as const).map((delay, i) => (
                        <motion.div
                          key={i}
                          animate={{ y: [0, -4, 0] }}
                          transition={{ duration: 0.6, delay, repeat: Infinity }}
                          className="w-1.5 h-1.5 rounded-full bg-blue-400"
                        />
                      ))}
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Suggestions — show only on fresh chat */}
              {messages.length === 1 && !loading && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="space-y-2"
                >
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider flex items-center gap-1.5 px-1">
                    <Sparkles className="w-3 h-3" />
                    Quick questions
                  </p>
                  <div className="grid grid-cols-1 gap-1.5">
                    {SUGGESTIONS.map(({ icon: Icon, text, color }) => (
                      <button
                        key={text}
                        onClick={() => sendMessage(text)}
                        className="flex items-center gap-2.5 px-3 py-2 rounded-xl border border-slate-700/50 bg-slate-800/40 hover:bg-slate-700/60 transition-all text-left group"
                      >
                        <div
                          className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
                          style={{ background: `${color}20` }}
                        >
                          <Icon className="w-3 h-3" style={{ color }} />
                        </div>
                        <span className="text-xs text-slate-400 group-hover:text-white transition-colors">
                          {text}
                        </span>
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}

              <div ref={bottomRef} />
            </div>

            {/* Input */}
            <div className="flex-shrink-0 p-3 border-t border-slate-700/80 bg-[#111827]">
              <div className="flex items-center gap-2 bg-slate-800/60 border border-slate-700/50 rounded-xl px-3 py-2">
                <input
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      sendMessage(input);
                    }
                  }}
                  placeholder="Ask about sensors, faults, costs..."
                  className="flex-1 bg-transparent text-sm text-white placeholder:text-slate-500 focus:outline-none"
                  disabled={loading}
                  maxLength={1000}
                />
                <button
                  onClick={() => sendMessage(input)}
                  disabled={loading || !input.trim()}
                  className="w-7 h-7 rounded-lg bg-blue-500/20 border border-blue-500/30 text-blue-400 hover:bg-blue-500/30 disabled:opacity-40 flex items-center justify-center transition-all flex-shrink-0"
                >
                  {loading
                    ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    : <Send    className="w-3.5 h-3.5" />
                  }
                </button>
              </div>
              <p className="text-[9px] text-slate-600 text-center mt-1.5">
                Powered by RAG + Claude AI • Maintenance knowledge base
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── FLOATING BUTTON ─────────────────────────────────────────────────── */}
      <motion.button
        onClick={() => setOpen(o => !o)}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        className="fixed bottom-6 right-6 z-[99991] w-14 h-14 rounded-2xl flex items-center justify-center shadow-2xl"
        style={{
          background:  open
            ? "linear-gradient(135deg,#EF4444,#DC2626)"
            : "linear-gradient(135deg,#3B82F6,#2563EB)",
          boxShadow: open
            ? "0 0 30px rgba(239,68,68,0.4), 0 8px 25px rgba(0,0,0,0.4)"
            : "0 0 30px rgba(59,130,246,0.4), 0 8px 25px rgba(0,0,0,0.4)",
        }}
      >
        <AnimatePresence mode="wait">
          {open ? (
            <motion.div
              key="close"
              initial={{ rotate: -90, opacity: 0 }}
              animate={{ rotate: 0,   opacity: 1 }}
              exit={{   rotate: 90,  opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <X className="w-6 h-6 text-white" />
            </motion.div>
          ) : (
            <motion.div
              key="bot"
              initial={{ rotate: 90,  opacity: 0 }}
              animate={{ rotate: 0,   opacity: 1 }}
              exit={{   rotate: -90, opacity: 0 }}
              transition={{ duration: 0.2 }}
            >
              <Bot className="w-6 h-6 text-white" />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Unread badge — now actually reachable */}
        <AnimatePresence>
          {showBadge && unread > 0 && !open && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              exit={{   scale: 0 }}
              className="absolute -top-1.5 -right-1.5 w-5 h-5 rounded-full bg-red-500 flex items-center justify-center border-2 border-[#0A0F1E]"
            >
              <span className="text-[10px] font-black text-white">
                {unread > 9 ? "9+" : unread}
              </span>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.button>

      {/* ── TOOLTIP — shows ONCE only, never repeats ────────────────────────── */}
      <AnimatePresence>
        {showTooltip && !open && (
          <motion.div
            initial={{ opacity: 0, x: 10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{   opacity: 0, x: 10 }}
            className="fixed bottom-8 right-24 z-[99990] px-3 py-2 rounded-xl bg-[#111827] border border-slate-700 shadow-xl pointer-events-none"
          >
            <p className="text-xs text-white font-medium whitespace-nowrap">
              Need help? Talk to AI Copilot
            </p>
            {/* Arrow pointing right */}
            <div className="absolute right-[-6px] top-1/2 -translate-y-1/2 w-3 h-3 rotate-45 bg-[#111827] border-r border-t border-slate-700" />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}