"""
Edge AI Predictive Maintenance — Advanced RAG Query Engine
===========================================================
Updated: Groq API (LLaMA 3.3 70B) instead of Anthropic Claude
- Groq is free tier, faster (300+ tokens/sec), perfect for demos
- All RAG logic unchanged — only LLM provider swapped
- Falls back to smart keyword responses if no API key set
"""

import os
import re

# ── Safe Groq import — server never crashes if package missing ────────────────
try:
    from groq import Groq as _GroqClient
    _GROQ_AVAILABLE = True
except ImportError:
    _GroqClient = None
    _GROQ_AVAILABLE = False
    print("[RAG] WARNING: 'groq' package not installed.")
    print("[RAG] Fix: pip install groq  +  add 'groq' to requirements.txt")
    print("[RAG] Chatbot will use built-in fallback responses until then.")

from .knowledge_base import hybrid_search

# ── Groq client singleton — created once, reused on every call ───────────────
_groq_client = None


def _get_groq_client():
    """Return cached Groq client. Returns None if package missing or no API key."""
    global _groq_client

    if not _GROQ_AVAILABLE:
        return None

    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if api_key:
            _groq_client = _GroqClient(api_key=api_key)
            print("[RAG] Groq client initialized — LLaMA 3.3 70B ready.")
        else:
            print("[RAG] No GROQ_API_KEY set — using fallback responses.")

    return _groq_client


# ══════════════════════════════════════════════════════════════════════════════
# DOMAIN SCOPE GUARD
# ══════════════════════════════════════════════════════════════════════════════

_DOMAIN_PHRASES: list = [
    # Hardware / system
    r"turbofan", r"jet engine", r"compressor", r"hpc", r"lpc", r"lpt",
    r"combustion", r"bypass duct", r"fan blade", r"fan speed", r"fan inlet",
    r"bearing", r"skf", r"rul", r"remaining useful life",
    r"anomaly probability", r"health score", r"edge ai", r"edge deployment",
    r"predictive maintenance", r"condition monitoring",

    # Sensors
    r"sensor\s*\d+", r"\bt2\b", r"\bt24\b", r"\bt30\b", r"\bt50\b",
    r"\bp2\b", r"\bp15\b", r"\bp30\b", r"\bnf\b", r"\bnc\b",
    r"\bepr\b", r"\bps30\b", r"\bphi\b", r"\bnrf\b", r"\bnrc\b",
    r"\bbpr\b", r"hpc temperature", r"hpc pressure", r"hpc outlet",

    # Fault and alerts
    r"hpc degradation", r"fan degradation", r"fault mode",
    r"severity", r"alert", r"anomaly", r"failure",
    r"\bcritical\b.*alert", r"\bhigh\b.*alert", r"\bmedium\b.*alert",
    r"\blow\b.*alert", r"escalation", r"notification bell",

    # ML / model
    r"transformer model", r"dual.head", r"attention head",
    r"onnx", r"pytorch", r"\bauc\b", r"\broc\b", r"auc.roc",
    r"accuracy.*model", r"model.*accuracy", r"inference speed",
    r"inference.*ms", r"18.?690.*param", r"param.*18.?690",
    r"positional encoding", r"sliding window", r"class imbalance",
    r"pos_weight", r"bce.*loss", r"mse.*loss",

    # Dataset
    r"nasa", r"cmapss", r"fd001", r"fd002", r"fd003", r"fd004",
    r"domain shift", r"138.?361", r"709.*engine", r"engine.*709",

    # MLOps
    r"mlflow", r"mlops", r"drift detection", r"model.*drift",
    r"retraining", r"experiment.*tracking",

    # OEE
    r"\boee\b", r"overall equipment", r"availability.*performance",
    r"six big loss", r"unplanned downtime", r"planned downtime",

    # Dashboard / app
    r"streamlit", r"react.*dashboard", r"digital twin", r"three\.js",
    r"plant map", r"fleet overview", r"framer motion", r"recharts",
    r"vite", r"tailwind", r"fastapi", r"uvicorn",
    r"localhost:8000", r"localhost:8080", r"localhost:8501",

    # RAG chatbot itself
    r"chromadb", r"bm25", r"langchain", r"rag", r"knowledge base",
    r"maintenance.*copilot", r"chatbot.*maintenance",

    # Cost / business
    r"cost.*saved", r"\$.*failure", r"roi.*maintenance",
    r"downtime.*cost", r"repair.*cost",
]

# Pre-compiled at import time — fast on every query
_COMPILED_PATTERNS: list = [
    re.compile(p, re.IGNORECASE) for p in _DOMAIN_PHRASES
]

_GREETINGS: list = [
    "hi", "hello", "hey", "good morning", "good afternoon",
    "good evening", "how are you", "what can you do",
    "help me", "what is this system", "who made this",
    "who built this", "tell me about this",
]

OUT_OF_SCOPE_RESPONSE = (
    "I'm the Edge AI Predictive Maintenance Assistant, "
    "trained exclusively to answer questions about this project.\n\n"
    "I can help you with:\n"
    "• 🌡️ Sensor readings and what they mean (T30, P30, Nf, etc.)\n"
    "• ⚠️ Alert severity levels and exact response procedures\n"
    "• 🔧 Maintenance scheduling, repair costs, and part numbers\n"
    "• 📊 OEE metrics and equipment effectiveness improvement\n"
    "• 🧠 AI model architecture, accuracy, and ONNX deployment\n"
    "• 📈 NASA CMAPSS dataset details and cross-dataset results\n"
    "• 🌍 Digital Twin, fleet overview, and dashboard navigation\n"
    "• 🔔 Notification system and escalation rules\n\n"
    "Please ask me something related to the Edge AI Predictive Maintenance System!"
)


def is_project_related(question: str) -> bool:
    """Check if question belongs to this project's domain."""
    q = question.lower().strip()

    if any(q.startswith(g) for g in _GREETINGS):
        return True

    # Allow short follow-up questions (under 6 words)
    if len(q.split()) <= 6:
        return True

    return any(pat.search(q) for pat in _COMPILED_PATTERNS)


# ══════════════════════════════════════════════════════════════════════════════
# QUERY EXPANSION
# ══════════════════════════════════════════════════════════════════════════════

_QUERY_EXPANSIONS: dict = {
    "how fast":     "inference speed latency milliseconds ONNX",
    "how accurate": "model accuracy AUC-ROC test validation",
    "how much":     "cost repair maintenance price dollars",
    "how long":     "time duration hours cycles days RUL",
    "what happens": "alert response action procedure",
    "is it good":   "accuracy performance benchmark world class",
    "why is":       "explanation reason cause analysis",
    "what to do":   "response procedure action steps",
    "broken":       "fault failure severity critical alert",
    "failing":      "anomaly probability critical high severity RUL",
    "shut down":    "critical alert emergency shutdown procedure",
    "best":         "world class benchmark optimal performance",
}


def _expand_query(question: str) -> str:
    """Expand vague questions with technical terms for better retrieval."""
    q_lower = question.lower()
    expansions = [exp for trigger, exp in _QUERY_EXPANSIONS.items() if trigger in q_lower]
    return f"{question} {' '.join(expansions)}" if expansions else question


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def get_chat_response(
    question: str,
    chat_history: list = None,
    engine_data: dict = None,
) -> str:
    """Generate RAG-powered response scoped to this project."""

    # 1. Scope guard
    if not is_project_related(question):
        return OUT_OF_SCOPE_RESPONSE

    # 2. Expand query
    search_query = _expand_query(question)

    # 3. Hybrid search
    context = ""
    try:
        chunks  = hybrid_search(search_query, k=5)
        context = "\n\n---\n\n".join(chunks[:4])
    except Exception as e:
        print(f"[RAG] Search failed: {e}")

    # 4. Format history
    history_text = ""
    if chat_history:
        meaningful = [
            m for m in chat_history
            if not (m.get("role") == "assistant"
                    and "Maintenance Copilot" in m.get("content", ""))
        ]
        for msg in meaningful[-6:]:
            role    = "Engineer" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:400].replace("\n", " ")
            history_text += f"{role}: {content}\n"

    # 5. Engine context (only when non-NORMAL)
    engine_ctx = ""
    if engine_data and engine_data.get("severity", "NORMAL") != "NORMAL":
        prob = engine_data.get("anomaly_probability", 0)
        engine_ctx = (
            f"\n⚡ LIVE ENGINE STATUS:\n"
            f"Engine #{engine_data.get('engine_id', 1)} — "
            f"Severity: {engine_data.get('severity', 'NORMAL')}\n"
            f"Anomaly Probability: {prob * 100:.1f}%\n"
            f"Health Score: {engine_data.get('health_score', 100):.1f}%\n"
            f"Root Cause: {engine_data.get('root_cause', 'Analysis in progress')}\n"
            f"RUL Remaining: {engine_data.get('rul_cycles', 100):.0f} cycles\n"
        )

    # 6. Try Groq API first
    client = _get_groq_client()
    if client:
        return _groq_response(
            question=question,
            context=context,
            history=history_text,
            engine_ctx=engine_ctx,
            client=client,
            fallback_context=context,
        )

    # 7. Smart fallback if no API key
    return _smart_fallback(question, context, engine_data)


# ══════════════════════════════════════════════════════════════════════════════
# GROQ API RESPONSE — LLaMA 3.3 70B
# ══════════════════════════════════════════════════════════════════════════════

def _groq_response(
    question: str,
    context: str,
    history: str,
    engine_ctx: str,
    client,
    fallback_context: str,
) -> str:
    """Generate answer using Groq LLaMA 3.3 70B with RAG context."""
    try:
        system_prompt = (
            "You are the Edge AI Predictive Maintenance Assistant — an expert AI "
            "exclusively for the Edge AI Predictive Maintenance System built with "
            "NASA turbofan data, PyTorch Transformer, ONNX Runtime, and FastAPI.\n\n"
            "STRICT RULES:\n"
            "- Answer ONLY questions about this specific project\n"
            "- Always use the retrieved knowledge base context as your primary source\n"
            "- Include specific numbers: accuracy percentages, costs, sensor ranges, times\n"
            "- Format with bullet points for lists, bold for key terms using **word**\n"
            "- Keep answers under 220 words — concise and actionable\n"
            "- For greetings, introduce yourself warmly and list what you can help with\n"
            "- Never make up information not in the context\n"
            "- Use emojis sparingly (max 2 per response) for readability"
        )

        user_message = (
            f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{context}\n\n"
            + (f"LIVE ENGINE STATUS:\n{engine_ctx}\n\n" if engine_ctx else "")
            + (f"RECENT CONVERSATION:\n{history}\n\n" if history else "")
            + f"ENGINEER QUESTION: {question}"
        )

        # ── Groq API call ─────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # Best free Groq model
            max_tokens=600,
            temperature=0.3,                    # Low temp = factual, consistent
            messages=[
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"[RAG] Groq API error: {type(e).__name__}: {e}")
        return _smart_fallback(question, fallback_context, None)


# ══════════════════════════════════════════════════════════════════════════════
# SMART FALLBACK — when Groq API unavailable
# ══════════════════════════════════════════════════════════════════════════════

def _smart_fallback(
    question: str,
    context: str,
    engine_data: dict,
) -> str:
    """Keyword-based fallback when Groq API is unavailable."""
    q = question.lower().strip()

    if re.search(r"\b(hi|hello|hey)\b", q):
        return (
            "👋 **Hello! I'm your Edge AI Maintenance Assistant!**\n\n"
            "I can help you with:\n"
            "• Sensor readings and fault detection (T30, P30, Nf...)\n"
            "• Alert severity levels and exact response steps\n"
            "• Maintenance costs and scheduling by RUL\n"
            "• OEE metrics and equipment effectiveness\n"
            "• Model accuracy, ONNX deployment, and benchmarks\n"
            "• Dashboard features and API endpoints\n\n"
            "What would you like to know?"
        )

    if "how are you" in q:
        return (
            "Running at **0.20ms** and feeling great! 😄\n\n"
            "I'm your Edge AI Maintenance Assistant, ready to help. "
            "What's your question?"
        )

    if re.search(r"\bcritical\b|\bemergency\b|\bshut.?down\b", q):
        return (
            "🚨 **CRITICAL Alert — Immediate Action Required**\n\n"
            "**Anomaly probability: 90-100%**\n\n"
            "Steps RIGHT NOW:\n"
            "1. SHUTDOWN engine immediately\n"
            "2. Notify CEO, Safety Officer, Plant Manager\n"
            "3. Emergency maintenance within 24 hours\n"
            "4. Expedite parts order (24-48h, 3-5x cost)\n\n"
            "**Cost if ignored: $350,000 – $500,000**"
        )

    if re.search(r"\bsensor\s*4\b|\bt30\b|\bhpc.*temp", q):
        return (
            "🌡️ **Sensor 4 (T30) — HPC Outlet Temperature**\n\n"
            "• **Normal:** 1589–1591°F\n"
            "• **Warning:** above 1600°F\n"
            "• **Critical:** above 1620°F\n\n"
            "Most critical sensor — rising T30 + dropping P30 = **HPC degradation**."
        )

    if re.search(r"\bsensor\s*9\b|\bp30\b|\bhpc.*pressure", q):
        return (
            "⚙️ **Sensor 9 (P30) — HPC Outlet Pressure**\n\n"
            "• **Normal:** 552–554 PSI\n"
            "• Dropping = compressor blade wear\n\n"
            "**Key pair:** Rising T30 + Dropping P30 = HPC degradation confirmed."
        )

    if re.search(r"\bhpc\b|\bcompressor\b", q):
        return (
            "⚙️ **HPC Degradation Fault Mode**\n\n"
            "Early warning (50–80 cycles before failure):\n"
            "• Sensor 4 (T30): rising above 1591°F\n"
            "• Sensor 9 (P30): dropping below 552 PSI\n\n"
            "**Planned repair:** $11,700–$19,800\n"
            "**If ignored:** $150,000–$500,000"
        )

    if re.search(r"\brul\b|\bremaining.*life\b|\bcycles.*left\b", q):
        return (
            "⏱️ **Remaining Useful Life (RUL)**\n\n"
            "• **RUL 60+:** Plan quarterly shutdown\n"
            "• **RUL 30–60:** Order parts now\n"
            "• **RUL 15–30:** URGENT — schedule this week\n"
            "• **RUL < 15:** CRITICAL — consider shutdown\n\n"
            "1 cycle ≈ 1 day of operation."
        )

    if re.search(r"\boee\b|\boverall equipment\b|\beffectiveness\b", q):
        return (
            "📊 **OEE = Availability × Performance × Quality**\n\n"
            "• World Class: **85%+**\n"
            "• Industry Average: **60%**\n"
            "• Each 1% ≈ **$100,000/year savings**\n\n"
            "Our AI improves OEE by 8–15 points = $1.5M–$3M annually."
        )

    if re.search(r"\baccuracy\b|\bauc.?roc\b|\bmodel.*performance\b", q):
        return (
            "🎯 **Model Performance**\n\n"
            "• Accuracy: **98.82%**\n"
            "• AUC-ROC: **0.997**\n"
            "• False Alarm Rate: **0.7%**\n"
            "• Parameters: **18,690**\n"
            "• Inference: **0.20ms** (250× faster than limit)"
        )

    if re.search(r"\bonnx\b|\bedge.*deploy\b|\binference\b|\b0\.20\b", q):
        return (
            "⚡ **ONNX Edge Deployment**\n\n"
            "• Inference: **0.20ms** (250× faster than 50ms limit)\n"
            "• No internet needed — fully offline\n"
            "• **$0/month** vs $2,000/month cloud\n"
            "• **95% power reduction** — 5W vs 250W GPU"
        )

    if re.search(r"\bcost\b|\bsaving\b|\broi\b|\bmoney\b", q):
        return (
            "💰 **Cost Savings**\n\n"
            "• CRITICAL prevented: **$350K–$500K saved**\n"
            "• Cloud eliminated: **$24,000/year**\n"
            "• Best ROI: **35,600%** per critical failure prevented"
        )

    # Use retrieved context if available
    if context and len(context) > 150:
        lines = [
            line.strip() for line in context.split("\n")
            if line.strip()
            and len(line.strip()) > 30
            and not line.strip().startswith("[Section")
            and not line.strip().startswith("---")
        ]
        if lines:
            return "Based on the maintenance knowledge base:\n\n" + \
                   "\n".join(f"• {l}" for l in lines[:7])

    return (
        "I can help with:\n\n"
        "• **Sensors:** T30, P30, Nf, T2 readings\n"
        "• **Faults:** HPC and fan degradation\n"
        "• **Alerts:** severity levels and response steps\n"
        "• **RUL:** scheduling and maintenance windows\n"
        "• **OEE:** equipment effectiveness metrics\n"
        "• **Model:** 98.82% accuracy, 0.20ms ONNX\n\n"
        "Please ask a specific question!"
    )