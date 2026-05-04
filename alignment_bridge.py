"""
Product–Dev Alignment Bridge Agent
====================================
Cross-layer coherence validator that sits between:
  - Product MCP Final Output (approved PRD)
  - Dev MCP Finalization (architecture + scaffolding)

NOT a builder. NOT a redesigner.
A reasoning agent that detects alignment drift between product intent
and technical implementation.

Pipeline (5 specialised sub-agents + QG loop):
  Step 1 — Intent Extractor          : distil PRD into verifiable intent dimensions
  Step 2 — Implementation Analyzer   : distil Dev output into implementation claims
  Step 3 — Deviation Detector        : cross-compare intent vs implementation
  Step 4 — Risk Assessor             : classify & score each deviation
  Step 5 — Alignment Synthesiser     : produce final alignment report + score

If alignment score < threshold → triggers iteration recommendation.
"""

import os
import sys
import json
import time
import re as _re

# -- Venv packages (local dev only) ------------------------------------------
VENV_SITE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "crewai-project", "venv", "Lib", "site-packages",
)
if os.path.isdir(VENV_SITE) and VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

import litellm

# -- Configuration -----------------------------------------------------------
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
os.environ["CEREBRAS_API_KEY"] = CEREBRAS_API_KEY

MODEL = "cerebras/llama3.1-8b"
MAX_TOKENS = 4096
MAX_TOKENS_LONG = 8192
TEMPERATURE = 0.4          # lower temp for analytical precision
API_KEY = CEREBRAS_API_KEY

ALIGNMENT_PASS_THRESHOLD = 70   # 0-100 scale; below = trigger iteration
MAX_ALIGNMENT_ITERATIONS = 2    # max re-evaluation cycles
ALIGNMENT_TOTAL_STEPS = 5

litellm.set_verbose = False


# =============================================================================
#  STEP METADATA  (for UI)
# =============================================================================
ALIGNMENT_STEP_INFO = [
    {"step": 1, "name": "Intent Extractor",        "desc": "Distil PRD into verifiable intent dimensions",                "phase": 1},
    {"step": 2, "name": "Implementation Analyzer",  "desc": "Parse dev output into implementation claims & decisions",     "phase": 1},
    {"step": 3, "name": "Deviation Detector",       "desc": "Cross-compare intent vs implementation, flag drift",          "phase": 2},
    {"step": 4, "name": "Risk Assessor",            "desc": "Classify deviations and assess risk severity",                "phase": 2},
    {"step": 5, "name": "Alignment Synthesiser",    "desc": "Produce final alignment report, score, and recommendation",   "phase": 3},
]


# =============================================================================
#  AGENT DEFINITIONS
# =============================================================================
ALIGNMENT_AGENTS = {
    "intent_extractor": {
        "role": "Product Intent Extractor",
        "system": (
            "You are a product intent analyst. Your SOLE job is to extract the TRUE product intent "
            "from a Product Requirements Document (PRD). Do NOT interpret, do NOT suggest solutions.\n\n"
            "Extract these EXACT dimensions:\n"
            "1. CORE VALUE PROPOSITION — What the product promises users\n"
            "2. BEHAVIORAL EXPECTATIONS — How the product should behave (real-time? offline? async?)\n"
            "3. PERFORMANCE REQUIREMENTS — Speed, latency, throughput, scale targets\n"
            "4. UX PROMISES — What user experience was pledged (simplicity? power? accessibility?)\n"
            "5. SECURITY & COMPLIANCE — Data protection, auth, regulatory requirements\n"
            "6. INTEGRATION EXPECTATIONS — External systems, APIs, data sources\n"
            "7. SCOPE BOUNDARIES — What is explicitly in-scope and out-of-scope\n"
            "8. PRIORITY HIERARCHY — P0/P1/P2 feature priorities\n"
            "9. SUCCESS METRICS — How success is measured (KPIs, acceptance criteria)\n"
            "10. IMPLICIT ASSUMPTIONS — Unstated but implied expectations\n\n"
            "For EACH dimension, output:\n"
            "- Dimension name\n"
            "- Explicit statements from the PRD (quote when possible)\n"
            "- Implied expectations\n"
            "- Verifiable criteria (how would you TEST this is met?)\n\n"
            "Output a structured Intent Specification."
        ),
    },
    "implementation_analyzer": {
        "role": "Implementation Claims Analyzer",
        "system": (
            "You are a technical implementation analyst. Your SOLE job is to extract what the "
            "developer architecture ACTUALLY delivers — not what it claims to deliver.\n\n"
            "From the architecture document, extract:\n"
            "1. ARCHITECTURE PATTERN — What pattern was chosen and WHY\n"
            "2. BEHAVIORAL IMPLEMENTATION — How does the system actually behave? (sync/async/event-driven/polling?)\n"
            "3. PERFORMANCE REALITY — What performance characteristics does this architecture actually provide?\n"
            "4. UX DELIVERY — What UX does this architecture enable or constrain?\n"
            "5. SECURITY IMPLEMENTATION — What security measures are actually in place?\n"
            "6. INTEGRATION APPROACH — How are external systems actually connected?\n"
            "7. SCOPE COVERAGE — What features are actually architected vs hand-waved?\n"
            "8. TECH STACK IMPLICATIONS — What does the chosen stack enable/limit?\n"
            "9. TRADE-OFF DECISIONS — What compromises were made and why?\n"
            "10. MISSING IMPLEMENTATIONS — What was in requirements but not in architecture?\n\n"
            "For EACH claim, note:\n"
            "- What is explicitly architected\n"
            "- What is only mentioned but not designed\n"
            "- What is completely absent\n\n"
            "Be brutally honest. If the architecture says 'real-time' but uses polling, say so."
        ),
    },
    "deviation_detector": {
        "role": "Alignment Deviation Detector",
        "system": (
            "You are a cross-layer alignment analyst. You compare PRODUCT INTENT against "
            "TECHNICAL IMPLEMENTATION to find mismatches.\n\n"
            "For EACH of the 10 intent dimensions, determine:\n"
            "- ALIGNED: Implementation faithfully reflects intent\n"
            "- DRIFTED: Implementation subtly differs from intent\n"
            "- COMPROMISED: Implementation makes trade-offs that weaken intent\n"
            "- VIOLATED: Implementation contradicts intent\n"
            "- MISSING: Intent dimension not addressed at all\n\n"
            "Look specifically for these alignment risks:\n"
            "- Real-time intent → batch/polling implementation\n"
            "- Simplicity intent → over-engineered architecture\n"
            "- Security expectations → weaker implementation guarantees\n"
            "- UX promises → backend-constraint compromises\n"
            "- Scale targets → architecture that won't scale\n"
            "- Offline intent → online-only architecture\n"
            "- Privacy requirements → insufficient data protection\n"
            "- Performance SLAs → architecture that can't meet them\n\n"
            "Output a DEVIATION TABLE with columns:\n"
            "| Dimension | Intent Summary | Implementation Reality | Status | Gap Description |\n\n"
            "Be forensic. Small misalignments compound into product failure."
        ),
    },
    "risk_assessor": {
        "role": "Deviation Risk Assessor",
        "system": (
            "You are a technical risk analyst specializing in product-engineering alignment.\n\n"
            "For EVERY detected deviation, you must:\n\n"
            "1. CLASSIFY the deviation as one of:\n"
            "   - NECESSARY_TRADEOFF: Technical constraints make this unavoidable; "
            "     the trade-off is reasonable and documented\n"
            "   - RISKY_COMPROMISE: The deviation weakens product intent; could cause "
            "     user dissatisfaction or technical debt\n"
            "   - INTENT_VIOLATION: The implementation directly contradicts a core product requirement; "
            "     this MUST be fixed before proceeding\n\n"
            "2. ASSESS risk level:\n"
            "   - LOW: Minor impact, acceptable with documentation\n"
            "   - MEDIUM: Noticeable impact, needs monitoring or future fix\n"
            "   - HIGH: Significant impact on product success, needs immediate attention\n"
            "   - CRITICAL: Blocks product viability, MUST be resolved\n\n"
            "3. JUSTIFY the classification — explain WHY this deviation matters or doesn't\n\n"
            "4. RECOMMEND action:\n"
            "   - ACCEPT: Proceed as-is\n"
            "   - REVISE_DEV: Dev team should redesign this component\n"
            "   - REVISIT_PRODUCT: Product scope needs clarification or adjustment\n"
            "   - ESCALATE: Needs stakeholder decision\n\n"
            "Output a RISK ASSESSMENT TABLE:\n"
            "| Deviation | Classification | Risk Level | Justification | Recommendation |\n\n"
            "Then provide:\n"
            "- Count of each classification type\n"
            "- Count of each risk level\n"
            "- Overall risk summary\n"
        ),
    },
    "alignment_synthesiser": {
        "role": "Alignment Bridge Synthesiser",
        "system": (
            "You are the Alignment Bridge — the final authority on product-dev coherence.\n\n"
            "You produce the DEFINITIVE Alignment Report. You MUST include ALL of these sections "
            "in EXACTLY this format:\n\n"
            "## ALIGNMENT SUMMARY\n"
            "A 3-5 sentence executive summary of alignment health.\n\n"
            "## ALIGNMENT SCORE: XX/100\n"
            "A numeric score from 0-100 based on:\n"
            "- Intent Preservation (0-25): Are core product intents preserved?\n"
            "- Behavioral Fidelity (0-20): Does the system behave as the product expects?\n"
            "- UX Consistency (0-20): Does the architecture enable the promised UX?\n"
            "- Performance Alignment (0-20): Can the architecture meet performance targets?\n"
            "- Risk Tolerance (0-15): Are the trade-offs acceptable?\n\n"
            "Show the breakdown:\n"
            "| Sub-Score | Points | Max | Rationale |\n\n"
            "## DEVIATION TABLE\n"
            "| # | Dimension | Intent | Implementation | Status | Classification | Risk |\n"
            "Include ALL deviations found.\n\n"
            "## RISK ASSESSMENT\n"
            "Summary of risk levels and classifications.\n"
            "- Critical risks: X\n"
            "- High risks: X\n"
            "- Medium risks: X\n"
            "- Low risks: X\n\n"
            "## ACTION RECOMMENDATION\n"
            "ONE of:\n"
            "- ACCEPT: Alignment is sufficient. Proceed to implementation.\n"
            "- REVISE_DEV: Dev architecture needs changes. List what to fix.\n"
            "- REVISIT_PRODUCT: Product scope needs clarification. List what's ambiguous.\n"
            "- ESCALATE: Fundamental misalignment needs stakeholder decision.\n\n"
            "Include specific action items for the recommended path.\n\n"
            "## ALIGNMENT HEALTH VERDICT\n"
            "ALIGNED / PARTIALLY_ALIGNED / MISALIGNED\n\n"
            "CRITICAL RULE: Your score MUST honestly reflect the alignment state. "
            "Do NOT inflate scores. A score of 70+ means the system is GENUINELY aligned. "
            "If there are INTENT_VIOLATIONS, the score CANNOT exceed 50."
        ),
    },
}


# =============================================================================
#  LLM CALL
# =============================================================================
def _call_llm(system_prompt, user_prompt, step_num=0, step_name="", on_event=None, max_tokens=None):
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    max_retries = 6
    for attempt in range(1, max_retries + 1):
        try:
            resp = litellm.completion(
                model=MODEL, messages=messages,
                max_tokens=max_tokens, temperature=TEMPERATURE, api_key=API_KEY,
            )
            content = resp.choices[0].message.content
            if content is None:
                content = getattr(resp.choices[0].message, 'reasoning_content', '') or ''
            return content.strip()
        except Exception as e:
            err = (type(e).__name__ + " " + str(e)).lower()
            transient = any(
                s in err
                for s in (
                    "rate_limit", "ratelimit", "429", "503",
                    "high traffic", "try again soon", "too many requests",
                    "over capacity", "temporarily unavailable",
                )
            )
            if transient:
                wait = min(180, 25 * (2 ** (attempt - 1)))
                msg = f"API busy / rate-limited, retrying in {wait}s (attempt {attempt}/{max_retries})"
                if on_event:
                    on_event("align_retry", step_num, step_name, msg)
                print(f"   [!] {msg}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {max_retries} retries — {step_name}")


def _trunc(text, n=2000):
    if not text:
        return ""
    return text[:n] + ("\n... [truncated]" if len(text) > n else "")


# =============================================================================
#  EMIT HELPER
# =============================================================================
def _emit(on_event, event, step_num, step_name, data=""):
    if on_event:
        on_event(event, step_num, step_name, data)


# =============================================================================
#  AGENT RUNNER
# =============================================================================
def _run_alignment_agent(agent_key, step_num, task_prompt, context="", on_event=None):
    agent = ALIGNMENT_AGENTS[agent_key]
    step_name = ALIGNMENT_STEP_INFO[step_num - 1]["name"]

    if context:
        user_msg = (
            "CONTEXT:\n" + ("-" * 40) + "\n" + context + "\n" + ("-" * 40)
            + "\n\nTASK:\n" + task_prompt
        )
    else:
        user_msg = task_prompt

    _emit(on_event, "align_start", step_num, step_name, "")
    _emit(on_event, "align_agent_info", step_num, step_name, json.dumps({
        "role": agent["role"],
        "system_prompt": agent["system"],
        "task": task_prompt,
        "context_preview": context[:800] if context else "(none)",
        "context_length": len(context) if context else 0,
    }))
    print(f"  ALIGN STEP {step_num}/{ALIGNMENT_TOTAL_STEPS} — {step_name} [{agent_key}]")

    tokens = MAX_TOKENS_LONG if agent_key == "alignment_synthesiser" else MAX_TOKENS
    result = _call_llm(
        system_prompt=agent["system"],
        user_prompt=user_msg,
        step_num=step_num,
        step_name=step_name,
        on_event=on_event,
        max_tokens=tokens,
    )

    _emit(on_event, "align_output_chunk", step_num, step_name, result)
    _emit(on_event, "align_done", step_num, step_name, result)
    print(f"   -> done ({len(result)} chars)")
    return result


# =============================================================================
#  SCORE PARSER
# =============================================================================
def _parse_alignment_score(report_text):
    """Extract alignment score (0-100) from synthesiser output."""
    score = None
    # Try: "ALIGNMENT SCORE: 75/100", "Score: 82/100", "XX/100"
    patterns = [
        r'alignment\s+score[:\s—\-]+(\d+)\s*/?\s*100',
        r'score[:\s—\-]+(\d+)\s*/\s*100',
        r'(\d+)\s*/\s*100',
        r'alignment\s+score[:\s—\-]+(\d+)',
    ]
    for pat in patterns:
        m = _re.search(pat, report_text, _re.IGNORECASE)
        if m:
            score = int(m.group(1))
            break
    if score is None:
        score = 50  # default if unparseable
    score = max(0, min(100, score))  # clamp
    return score


def _parse_alignment_verdict(report_text):
    """Extract verdict from the report."""
    upper = report_text.upper()
    if "MISALIGNED" in upper:
        return "MISALIGNED"
    if "PARTIALLY_ALIGNED" in upper or "PARTIALLY ALIGNED" in upper:
        return "PARTIALLY_ALIGNED"
    if "ALIGNED" in upper:
        return "ALIGNED"
    return "UNKNOWN"


def _parse_action_recommendation(report_text):
    """Extract the primary action recommendation."""
    upper = report_text.upper()
    # Look in the ACTION RECOMMENDATION section
    rec_section = ""
    lines = report_text.split('\n')
    in_rec = False
    for line in lines:
        if 'ACTION RECOMMENDATION' in line.upper():
            in_rec = True
            continue
        if in_rec and line.strip().startswith('##'):
            break
        if in_rec:
            rec_section += line + '\n'

    rec_upper = rec_section.upper() if rec_section else upper
    if "ESCALATE" in rec_upper:
        return "ESCALATE"
    if "REVISIT_PRODUCT" in rec_upper or "REVISIT PRODUCT" in rec_upper:
        return "REVISIT_PRODUCT"
    if "REVISE_DEV" in rec_upper or "REVISE DEV" in rec_upper:
        return "REVISE_DEV"
    return "ACCEPT"


def _parse_deviations(report_text):
    """Extract deviation entries from the report."""
    deviations = []
    lines = report_text.split('\n')
    in_table = False
    for line in lines:
        stripped = line.strip()
        if 'DEVIATION TABLE' in stripped.upper():
            in_table = True
            continue
        if in_table and stripped.startswith('|') and '---' not in stripped:
            cells = [c.strip() for c in stripped.split('|') if c.strip()]
            if len(cells) >= 4 and not cells[0].startswith('#'):
                deviations.append({
                    "dimension": cells[1] if len(cells) > 1 else "",
                    "intent": cells[2] if len(cells) > 2 else "",
                    "implementation": cells[3] if len(cells) > 3 else "",
                    "status": cells[4] if len(cells) > 4 else "",
                    "classification": cells[5] if len(cells) > 5 else "",
                    "risk": cells[6] if len(cells) > 6 else "",
                })
        if in_table and stripped.startswith('##') and 'DEVIATION' not in stripped.upper():
            in_table = False
    return deviations[:20]  # cap at 20


def _parse_sub_scores(report_text):
    """Extract sub-score breakdown from the report."""
    sub_scores = {}
    lines = report_text.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('|') and '---' not in stripped:
            cells = [c.strip() for c in stripped.split('|') if c.strip()]
            if len(cells) >= 3:
                name = cells[0].lower()
                try:
                    points = int(_re.search(r'(\d+)', cells[1]).group(1))
                    max_pts = int(_re.search(r'(\d+)', cells[2]).group(1))
                    if name and max_pts > 0 and max_pts <= 25 and 'sub' not in name and 'points' not in name:
                        sub_scores[cells[0]] = {"points": points, "max": max_pts, "rationale": cells[3] if len(cells) > 3 else ""}
                except (AttributeError, ValueError):
                    pass
    return sub_scores


def _parse_risk_counts(report_text):
    """Extract risk level counts."""
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for level in counts:
        m = _re.search(rf'{level}\s*(?:risks?)?[:\s]+(\d+)', report_text, _re.IGNORECASE)
        if m:
            counts[level] = int(m.group(1))
    return counts


# =============================================================================
#  THE ALIGNMENT BRIDGE PIPELINE
# =============================================================================
def run_alignment_bridge(prd_text, dev_output, dev_state=None, on_event=None):
    """
    Run the Product–Dev Alignment Bridge.

    Args:
        prd_text:   The approved PRD (Product MCP final output)
        dev_output: The dev architecture package (Dev MCP final output)
        dev_state:  Optional dict with individual dev agent outputs for deeper analysis
        on_event:   SSE event callback

    Returns:
        Dict with alignment report, score, verdict, deviations, recommendation, etc.
    """
    t0 = time.time()
    R = {}
    step_outputs = []

    def save(key, step_num, result):
        R[key] = result
        step_outputs.append({"step": step_num, "key": key, "chars": len(result)})

    _emit(on_event, "align_pipeline_start", 0, "Alignment Bridge", json.dumps({
        "total_steps": ALIGNMENT_TOTAL_STEPS,
        "threshold": ALIGNMENT_PASS_THRESHOLD,
    }))

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1 — Extraction (Steps 1 & 2 — can run in parallel)
    # ══════════════════════════════════════════════════════════════════

    # Extra dev context from pipeline state if available
    dev_context_extra = ""
    if dev_state:
        parts = []
        for key in ["architecture", "tech_stack", "api_contracts", "failure_modes", "eng_requirements"]:
            val = dev_state.get(key, "")
            if val:
                parts.append(f"{key.upper().replace('_',' ')}:\n{_trunc(val, 1500)}")
        dev_context_extra = "\n\n".join(parts)

    # Step 1 — Intent Extractor
    res = _run_alignment_agent("intent_extractor", 1,
        task_prompt=(
            "Extract ALL product intent dimensions from this PRD. "
            "Be exhaustive. Every requirement, every promise, every metric, every constraint. "
            "Include both explicit statements and implied expectations. "
            "This intent specification will be used to verify technical alignment."
        ),
        context=f"APPROVED PRD:\n{_trunc(prd_text, 6000)}",
        on_event=on_event)
    save("intent", 1, res)

    # Step 2 — Implementation Analyzer
    impl_ctx = f"DEV ARCHITECTURE PACKAGE:\n{_trunc(dev_output, 5000)}"
    if dev_context_extra:
        impl_ctx += f"\n\nADDITIONAL DEV AGENT OUTPUTS:\n{_trunc(dev_context_extra, 3000)}"

    res = _run_alignment_agent("implementation_analyzer", 2,
        task_prompt=(
            "Analyze what this architecture ACTUALLY delivers. "
            "Extract every implementation decision, every trade-off, every capability, "
            "and every gap. Distinguish between what is genuinely architected vs. "
            "merely mentioned. Be brutally honest about limitations."
        ),
        context=impl_ctx,
        on_event=on_event)
    save("implementation", 2, res)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2 — Analysis (Steps 3 & 4 — sequential for accuracy)
    # ══════════════════════════════════════════════════════════════════

    # Step 3 — Deviation Detector
    compare_ctx = (
        f"PRODUCT INTENT SPECIFICATION:\n{_trunc(R['intent'], 4000)}\n\n"
        f"IMPLEMENTATION ANALYSIS:\n{_trunc(R['implementation'], 4000)}"
    )

    res = _run_alignment_agent("deviation_detector", 3,
        task_prompt=(
            "Cross-compare the product intent against the implementation analysis. "
            "For EVERY intent dimension, determine alignment status. "
            "Look for silent scope shifts, behavioral deviations, performance gaps, "
            "UX compromises, and security shortfalls. "
            "Output a comprehensive deviation table."
        ),
        context=compare_ctx,
        on_event=on_event)
    save("deviations", 3, res)

    # Step 4 — Risk Assessor
    res = _run_alignment_agent("risk_assessor", 4,
        task_prompt=(
            "Classify and assess every deviation found. "
            "For EACH: classify as Necessary Trade-off, Risky Compromise, or Intent Violation. "
            "Assign risk levels (Low/Medium/High/Critical). "
            "Recommend action for each (Accept / Revise Dev / Revisit Product / Escalate). "
            "Provide an overall risk summary."
        ),
        context=(
            f"DEVIATION ANALYSIS:\n{_trunc(R['deviations'], 4000)}\n\n"
            f"PRODUCT INTENT:\n{_trunc(R['intent'], 2000)}\n\n"
            f"IMPLEMENTATION CLAIMS:\n{_trunc(R['implementation'], 2000)}"
        ),
        on_event=on_event)
    save("risk_assessment", 4, res)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3 — Synthesis (Step 5 — final alignment report)
    # ══════════════════════════════════════════════════════════════════

    synth_ctx = (
        f"PRODUCT INTENT:\n{_trunc(R['intent'], 2500)}\n\n"
        f"IMPLEMENTATION ANALYSIS:\n{_trunc(R['implementation'], 2500)}\n\n"
        f"DEVIATION ANALYSIS:\n{_trunc(R['deviations'], 3000)}\n\n"
        f"RISK ASSESSMENT:\n{_trunc(R['risk_assessment'], 3000)}\n\n"
        f"ORIGINAL PRD (for reference):\n{_trunc(prd_text, 1500)}"
    )

    report = _run_alignment_agent("alignment_synthesiser", 5,
        task_prompt=(
            "Produce the FINAL Alignment Report. Synthesize all analyses into a single, "
            "authoritative document. You MUST include: "
            "Alignment Summary, Alignment Score (0-100) with sub-score breakdown, "
            "Deviation Table, Risk Assessment summary, Action Recommendation, "
            "and Alignment Health Verdict. "
            "Be honest. If alignment is poor, say so. If it's good, explain why."
        ),
        context=synth_ctx,
        on_event=on_event)
    save("alignment_report", 5, report)

    # ── Parse results ──
    alignment_score = _parse_alignment_score(report)
    verdict = _parse_alignment_verdict(report)
    recommendation = _parse_action_recommendation(report)
    deviations = _parse_deviations(report)
    sub_scores = _parse_sub_scores(report)
    risk_counts = _parse_risk_counts(report)

    # ── Determine pass/fail ──
    passed = alignment_score >= ALIGNMENT_PASS_THRESHOLD

    # ── Iteration recommendation ──
    iteration_action = None
    if not passed:
        if recommendation in ("REVISE_DEV", "ESCALATE"):
            iteration_action = "REVISE_DEV"
        elif recommendation == "REVISIT_PRODUCT":
            iteration_action = "REVISIT_PRODUCT"
        else:
            iteration_action = "REVISE_DEV"  # default

    elapsed = round(time.time() - t0, 1)

    result = {
        "alignment_report": report,
        "alignment_score": alignment_score,
        "verdict": verdict,
        "recommendation": recommendation,
        "passed": passed,
        "threshold": ALIGNMENT_PASS_THRESHOLD,
        "deviations": deviations,
        "sub_scores": sub_scores,
        "risk_counts": risk_counts,
        "iteration_action": iteration_action,
        "step_outputs": step_outputs,
        "elapsed": elapsed,
        "pipeline_state": {
            "intent": R.get("intent", ""),
            "implementation": R.get("implementation", ""),
            "deviations": R.get("deviations", ""),
            "risk_assessment": R.get("risk_assessment", ""),
            "alignment_report": report,
        },
    }

    # ── Emit final event ──
    _emit(on_event, "align_complete", 0, "Alignment Bridge Complete", json.dumps({
        "alignment_report": report,
        "alignment_score": alignment_score,
        "verdict": verdict,
        "recommendation": recommendation,
        "passed": passed,
        "threshold": ALIGNMENT_PASS_THRESHOLD,
        "deviations": deviations,
        "sub_scores": sub_scores,
        "risk_counts": risk_counts,
        "iteration_action": iteration_action,
        "elapsed": elapsed,
        "step_outputs": step_outputs,
    }))

    print(f"\n  Alignment Bridge: score={alignment_score}/100, "
          f"verdict={verdict}, recommendation={recommendation}, "
          f"elapsed={elapsed}s")

    return result


# =============================================================================
#  CLI
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Product–Dev Alignment Bridge (CLI)")
    print("=" * 60)
    print("  This agent requires both a PRD and a Dev output.")
    print("  Use the web UI for the full workflow.")
    print("=" * 60 + "\n")
