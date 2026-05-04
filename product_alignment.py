"""
AI-Powered Product Alignment System — TRUE Multi-Agent Orchestration
=====================================================================
This is NOT a sequential pipeline with different prompts.
Agents genuinely communicate, debate, critique, and iterate:

  Phase 1  — Discovery          (steps 1-2: sequential)
  Phase 2  — Parallel Analysis  (steps 3/4/5: simultaneous)
  Phase 3  — Cross-Agent Debate (step 6: synthesiser critiques all three)
  Phase 4  — Quality-Gated PRD  (step 7 draft → step 8 QG → step 9 revision)
  Phase 5  — Final Compilation  (step 10)

~10-13 LLM calls.  Impossible to replicate with a single prompt.
"""

import os
import sys
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

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
MAX_TOKENS = 4096              # default for most agents
MAX_TOKENS_LONG = 8192         # for PRD-producing agents that need more room
TEMPERATURE = 0.7
API_KEY = CEREBRAS_API_KEY
PAUSE_BETWEEN_AGENTS = 0
MAX_QG_ITERATIONS = 3        # max revision cycles before forced approval
QG_PASS_THRESHOLD = 7        # quality gate score >= this = APPROVED

# Agents that produce long documents get higher token limits
LONG_TOKEN_AGENTS = {"prd_author", "prd_reviser", "mcp_orchestrator", "debate_synthesiser"}

litellm.set_verbose = False


# =============================================================================
#  STEP METADATA  (for UI — now 10 steps with phases)
# =============================================================================
STEP_INFO = [
    {"num": 1,  "name": "Product Input",       "icon": "inbox",          "desc": "Capturing & analysing raw idea",          "phase": "Discovery"},
    {"num": 2,  "name": "Interrogation",        "icon": "help-circle",    "desc": "Stress-testing with probing questions",   "phase": "Discovery"},
    {"num": 3,  "name": "Pattern Analysis",     "icon": "search",         "desc": "Matching against known product patterns", "phase": "Analysis"},
    {"num": 4,  "name": "Consistency Audit",    "icon": "check-circle",   "desc": "Finding contradictions & gaps",           "phase": "Analysis"},
    {"num": 5,  "name": "Scope & Priority",     "icon": "layers",         "desc": "Categorising P0 / P1 / P2",              "phase": "Analysis"},
    {"num": 6,  "name": "Cross-Agent Debate",   "icon": "message-circle", "desc": "Agents critique each other's findings",   "phase": "Debate"},
    {"num": 7,  "name": "PRD Draft",            "icon": "file-text",      "desc": "Writing initial structured PRD",          "phase": "PRD"},
    {"num": 8,  "name": "Quality Gate",         "icon": "shield",         "desc": "Meta-orchestrator evaluates PRD quality", "phase": "PRD"},
    {"num": 9,  "name": "PRD Revision",         "icon": "edit",           "desc": "Revising PRD based on quality feedback",  "phase": "PRD"},
    {"num": 10, "name": "Final Compilation",    "icon": "package",        "desc": "Assembling investor-grade deliverable",   "phase": "Final"},
]

TOTAL_STEPS = len(STEP_INFO)


# =============================================================================
#  AGENT DEFINITIONS  (10 distinct agents)
# =============================================================================
AGENTS = {
    "input_analyst": {
        "role": "Product Input Analyst",
        "system": (
            "You are a meticulous product analyst. Deeply understand the user's raw idea. "
            "Restate it faithfully, then identify: (a) the core value proposition, "
            "(b) the implied user, (c) hidden assumptions baked into the phrasing. "
            "Think step by step."
        ),
    },
    "interrogator": {
        "role": "Product Interrogator",
        "system": (
            "You are a senior PM with 15 years shipping products at scale. "
            "Stress-test this idea with 10-12 probing questions covering: "
            "persona definition, problem validation, scope boundaries, success/failure metrics, "
            "technical feasibility, edge cases, competitive landscape, go-to-market risks. "
            "For EACH question provide a well-reasoned Default Answer."
        ),
    },
    "pattern_analyst": {
        "role": "Product Pattern Analyst",
        "system": (
            "You are a product strategist who has studied hundreds of SaaS launches. "
            "Compare this feature against established patterns and best practices.\n"
            "Cover: 1. Matched Patterns (e.g. RBAC, file management, audit trails)\n"
            "2. Missing Typical Requirements  3. Red Flags  4. Comparable Products (2-3) with lessons.\n"
            "Use concrete examples. Think step by step."
        ),
    },
    "consistency_auditor": {
        "role": "Requirement Consistency Auditor",
        "system": (
            "You think like a lawyer reading a contract — every ambiguity is a future bug. "
            "Deep audit covering:\n"
            "1. Contradictions  2. Vague language  3. Unclear ownership\n"
            "4. Missing edge cases (empty states, errors, permissions, bulk ops, concurrency)\n"
            "5. Security & compliance gaps.\n"
            "For each: issue, real-world impact, concrete resolution."
        ),
    },
    "scope_manager": {
        "role": "Scope & Priority Manager",
        "system": (
            "You ship lean V0s on time. Ruthlessly prioritise:\n"
            "P0 (launch blocker) — P1 (should have) — P2 (future).\n"
            "For each item explain WHY. Flag scope creep. "
            "End with V0 scope boundary + complexity estimate (S/M/L) per P0."
        ),
    },
    "debate_synthesiser": {
        "role": "Cross-Agent Debate Synthesiser",
        "system": (
            "You are the moderator of a product review meeting. Three analysts just presented "
            "their independent findings. Your job is to:\n"
            "1. Identify where they AGREE (consensus points)\n"
            "2. Identify where they DISAGREE or have gaps between them\n"
            "3. For each disagreement, argue which side is more valid and WHY\n"
            "4. Surface NEW insights that emerge only when combining all three perspectives\n"
            "5. Produce a unified 'Debate Summary' that feeds into the PRD.\n"
            "Be specific — cite each analyst by name (Pattern Analyst, Auditor, Scope Manager)."
        ),
    },
    "prd_author": {
        "role": "PRD Author",
        "system": (
            "You are a staff-level PM writing a PRD for engineering kick-off.\n"
            "Sections: 1. Problem Statement  2. User Personas (2-3)\n"
            "3. User Flows (happy + error paths)  4. Functional Reqs (P0/P1/P2 with acceptance criteria)\n"
            "5. Non-Functional Reqs  6. Assumptions & Dependencies\n"
            "7. Out-of-Scope  8. Success Metrics (KPIs with targets).\n"
            "Use markdown. Be thorough and actionable."
        ),
    },
    "quality_gate": {
        "role": "Quality Gate — Meta-Orchestrator",
        "system": (
            "You are the chief product officer reviewing a PRD before it goes to engineering. "
            "Evaluate ruthlessly:\n"
            "1. **Completeness** — Any sections thin, missing, or hand-wavy?\n"
            "2. **Clarity** — Could an engineer build from this without asking questions?\n"
            "3. **Consistency** — Do priorities match the problem? Do flows match requirements?\n"
            "4. **Feasibility** — Unrealistic assumptions? Missing technical constraints?\n"
            "5. **Quality Score** — Rate 1-10 with justification.\n\n"
            "VERDICT: Output either:\n"
            "  APPROVED — if score >= 7 (with minor suggestions)\n"
            "  REVISION_NEEDED — if score < 7 (with specific numbered feedback)\n\n"
            "Be honest. Do NOT rubber-stamp. A bad PRD wastes engineering weeks."
        ),
    },
    "prd_reviser": {
        "role": "PRD Revision Author",
        "system": (
            "You are the same PRD author. The Quality Gate sent back feedback. You must:\n"
            "1. Address EVERY piece of feedback point by point\n"
            "2. Rewrite the FULL PRD (not patches) incorporating improvements\n"
            "3. Add 'Revision Notes' at top listing what changed and why.\n"
            "Produce the complete improved PRD in markdown."
        ),
    },
    "mcp_orchestrator": {
        "role": "MCP Final Orchestrator",
        "system": (
            "Compile everything into an investor-grade final deliverable.\n"
            "### 1. Executive Summary (3-4 sentences)\n"
            "### 2. Consolidated PRD (FULL — do NOT summarise)\n"
            "### 3. Surfaced Assumptions (with risk: Low/Med/High)\n"
            "### 4. Cross-Agent Debate Highlights (disagreements + resolutions)\n"
            "### 5. Quality Gate Results (score, verdict, what was revised)\n"
            "### 6. Future Misalignment Areas\n"
            "### 7. Clarity Confidence Score (Low/Med/High + justification)\n"
            "This should impress a VC reviewing the team's process."
        ),
    },
}


# =============================================================================
#  LLM CALL  (with retry)
# =============================================================================
def _call_llm(system_prompt, user_prompt, step_num=0, step_name="", on_event=None, max_tokens=None):
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    max_retries = 3
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
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = 12 * attempt
                msg = f"Rate-limited, retrying in {wait}s (attempt {attempt}/{max_retries})"
                if on_event:
                    on_event("retry", step_num, step_name, msg)
                print(f"   [!] {msg}")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Step {step_num} ({step_name}) failed after {max_retries} retries")


def _trunc(text, n=2000):
    if not text:
        return ""
    return text[:n] + ("\n... [truncated]" if len(text) > n else "")


import re as _re

def _parse_qg_score(qg_text):
    """Extract numeric score and verdict from Quality Gate output."""
    score = None
    verdict = "REVISION_NEEDED"

    # Try to find score patterns like "Score: 8/10", "Quality Score: 7", "8/10", "Score — 6"
    patterns = [
        r'(?:quality\s*)?score[:\s—\-]+(\d+)\s*/?\s*10',
        r'(\d+)\s*/\s*10',
        r'(?:quality\s*)?score[:\s—\-]+(\d+)',
        r'\brating[:\s]+(\d+)',
    ]
    for pat in patterns:
        m = _re.search(pat, qg_text, _re.IGNORECASE)
        if m:
            score = int(m.group(1))
            break

    if score is None:
        score = 5  # default if unparseable

    # Numeric score is the SOLE authority — LLM text cannot override
    if score >= QG_PASS_THRESHOLD:
        verdict = "APPROVED"
    else:
        verdict = "REVISION_NEEDED"

    return score, verdict


def _parse_qg_improvements(qg_text):
    """Extract structured improvements, remaining issues, and score justification from QG output."""
    improvements = []
    remaining_issues = []
    score_justification = []
    
    lines = qg_text.split('\n')
    section = None
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        # Detect section headers
        if any(k in upper for k in ['IMPROVEMENT', 'WHAT IMPROVED', 'BETTER THAN', 'PROGRESS']):
            section = 'improvements'
            continue
        if any(k in upper for k in ['REMAINING', 'STILL NEED', 'ISSUES', 'GAPS', 'WEAKNESSES', 'AREAS FOR']):
            section = 'issues'
            continue
        if any(k in upper for k in ['SCORE JUSTIFICATION', 'SCORE EXPLANATION', 'WHY THIS SCORE', 'SCORING RATIONALE']):
            section = 'justification'
            continue
        if any(k in upper for k in ['SCORE:', 'VERDICT:', 'OVERALL:', 'SUMMARY:', 'APPROVED', 'REVISION_NEEDED']):
            if section != 'justification':  # Don't break out of justification section on SCORE mentions
                section = None
            continue
        # Collect bullet points or paragraph text
        if stripped and section:
            if stripped.startswith('-') or stripped.startswith('•') or stripped.startswith('*') or _re.match(r'^\d+[.)\s]', stripped):
                text = _re.sub(r'^[-•*\d.)+\s]+', '', stripped).strip()
                if text and len(text) > 5:
                    if section == 'improvements':
                        improvements.append(text)
                    elif section == 'issues':
                        remaining_issues.append(text)
                    elif section == 'justification':
                        score_justification.append(text)
            elif section == 'justification' and len(stripped) > 10:
                # Justification can be paragraph text, not just bullets
                score_justification.append(stripped)
    
    # Fallback: if no structured sections found, try to extract from full text
    if not improvements and not remaining_issues:
        for line in lines:
            s = line.strip()
            if s.startswith('+') or s.lower().startswith('improved:'):
                improvements.append(_re.sub(r'^[+\s]+|^improved:\s*', '', s, flags=_re.I).strip())
            elif s.startswith('-') and ('lack' in s.lower() or 'miss' in s.lower() or 'need' in s.lower() or 'weak' in s.lower()):
                remaining_issues.append(_re.sub(r'^[-\s]+', '', s).strip())
    
    return improvements[:6], remaining_issues[:6], score_justification[:4]


# =============================================================================
#  AGENT RUNNER  (emits rich events for live UI)
# =============================================================================
def _emit(on_event, event, step_num, step_name, data=""):
    if on_event:
        on_event(event, step_num, step_name, data)


def _run_agent(agent_key, step_num, task_prompt, context="", on_event=None):
    """Run a named agent and return result text. Emits UI events."""
    agent = AGENTS[agent_key]
    step_name = STEP_INFO[step_num - 1]["name"]

    if context:
        user_msg = (
            "CONTEXT:\n" + ("-" * 40) + "\n" + context + "\n" + ("-" * 40)
            + "\n\nTASK:\n" + task_prompt
        )
    else:
        user_msg = task_prompt

    _emit(on_event, "start", step_num, step_name, "")
    _emit(on_event, "agent_info", step_num, step_name, json.dumps({
        "role": agent["role"],
        "system_prompt": agent["system"],
        "task": task_prompt,
        "context_preview": context[:800] if context else "(none — first step)",
        "context_length": len(context) if context else 0,
    }))
    print(f"  STEP {step_num}/{TOTAL_STEPS} — {step_name} [{agent_key}]")

    # Use higher token limit for document-producing agents
    tokens = MAX_TOKENS_LONG if agent_key in LONG_TOKEN_AGENTS else MAX_TOKENS

    result = _call_llm(
        system_prompt=agent["system"],
        user_prompt=user_msg,
        step_num=step_num,
        step_name=step_name,
        on_event=on_event,
        max_tokens=tokens,
    )

    _emit(on_event, "output_chunk", step_num, step_name, result)
    _emit(on_event, "done", step_num, step_name, result)
    print(f"   -> done ({len(result)} chars)")
    return result


# =============================================================================
#  THE ORCHESTRATED PIPELINE
# =============================================================================
def run_pipeline(raw_idea, on_event=None):
    """
    True multi-agent orchestration:
      Phase 1 — Discovery (sequential)
      Phase 2 — Parallel Analysis (3 agents)
      Phase 3 — Cross-Agent Debate (synthesiser)
      Phase 4 — Quality-Gated PRD (draft → evaluate → revise)
      Phase 5 — Final Compilation
    """
    t0 = time.time()
    R = {}             # results dict
    step_outputs = []

    def save(key, step_num, result):
        R[key] = result
        step_outputs.append({
            "step": step_num,
            "name": STEP_INFO[step_num - 1]["name"],
            "output": result,
        })

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1 — DISCOVERY
    # ══════════════════════════════════════════════════════════════════

    res = _run_agent("input_analyst", 1,
        task_prompt=(
            f'Analyze and restate this raw product idea. '
            f'Identify the core value proposition, implied user, and hidden assumptions:\n\n'
            f'"{raw_idea}"'
        ),
        on_event=on_event)
    save("input", 1, res)

    res = _run_agent("interrogator", 2,
        task_prompt=(
            "Generate 10-12 probing questions to stress-test this product idea. "
            "Cover personas, problem validation, scope, metrics, feasibility, edges, competition. "
            "For EACH give a well-reasoned Default Answer."
        ),
        context=_trunc(R["input"], 2000),
        on_event=on_event)
    save("interrogation", 2, res)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2 — PARALLEL ANALYSIS  (3 agents at once)
    # ══════════════════════════════════════════════════════════════════

    _emit(on_event, "phase", 0, "Parallel Analysis",
          "Launching 3 analysis agents simultaneously...")

    shared_ctx = (
        f"PRODUCT IDEA: {raw_idea}\n\n"
        f"INPUT ANALYSIS:\n{_trunc(R['input'], 1500)}\n\n"
        f"INTERROGATION:\n{_trunc(R['interrogation'], 3000)}"
    )

    def _par(agent_key, step_num, task):
        return step_num, _run_agent(agent_key, step_num,
            task_prompt=task, context=shared_ctx, on_event=on_event)

    tasks_3_4_5 = [
        ("pattern_analyst", 3,
         "Compare this feature against known product patterns. "
         "Cover: matched patterns, missing requirements, red flags, comparable products."),
        ("consistency_auditor", 4,
         "Deep-audit all requirements. Find contradictions, vague language, "
         "unclear ownership, missing edge cases, security gaps. "
         "For each: issue, impact, resolution."),
        ("scope_manager", 5,
         "Categorise every requirement into P0/P1/P2 with reasoning. "
         "Flag scope creep. End with V0 boundary + complexity estimates."),
    ]

    with ThreadPoolExecutor(max_workers=3) as pool:
        futs = {pool.submit(_par, a, s, t): s for a, s, t in tasks_3_4_5}
        for f in as_completed(futs):
            sn, res = f.result()
            key = {3: "patterns", 4: "consistency", 5: "scope"}[sn]
            save(key, sn, res)

    step_outputs.sort(key=lambda x: x["step"])

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3 — CROSS-AGENT DEBATE
    #
    #  The three analysts' outputs go to a debate synthesiser who finds
    #  agreements, disagreements, and emergent insights. This is what
    #  makes it truly multi-agent — agents interact with EACH OTHER's
    #  work, not just the original input.
    # ══════════════════════════════════════════════════════════════════

    debate_ctx = (
        f"=== PATTERN ANALYST OUTPUT ===\n{_trunc(R.get('patterns',''), 2500)}\n\n"
        f"=== CONSISTENCY AUDITOR OUTPUT ===\n{_trunc(R.get('consistency',''), 2500)}\n\n"
        f"=== SCOPE MANAGER OUTPUT ===\n{_trunc(R.get('scope',''), 2500)}"
    )

    res = _run_agent("debate_synthesiser", 6,
        task_prompt=(
            "Three analysts completed independent reviews. Read ALL three, then:\n"
            "1. Where do they AGREE? (consensus)\n"
            "2. Where do they DISAGREE or leave gaps?\n"
            "3. For each disagreement — which side is right and WHY?\n"
            "4. What NEW insights emerge from combining all three?\n"
            "5. Produce a unified Debate Summary for the PRD author."
        ),
        context=debate_ctx,
        on_event=on_event)
    save("debate", 6, res)

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 4 — QUALITY-GATED PRD  (with real iteration loop)
    #
    #  Step 7: PRD Draft
    #  Step 8: Quality Gate evaluates → score + verdict
    #  If REVISION_NEEDED and iterations < MAX_QG_ITERATIONS:
    #    Step 9: PRD Revision → loop back to Step 8
    #  Repeat until APPROVED or max iterations reached.
    # ══════════════════════════════════════════════════════════════════

    prd_ctx = (
        f"PRODUCT IDEA: {raw_idea}\n\n"
        f"INTERROGATION:\n{_trunc(R.get('interrogation',''), 2500)}\n\n"
        f"PATTERNS:\n{_trunc(R.get('patterns',''), 2000)}\n\n"
        f"ISSUES:\n{_trunc(R.get('consistency',''), 2000)}\n\n"
        f"PRIORITIES:\n{_trunc(R.get('scope',''), 2000)}\n\n"
        f"DEBATE SUMMARY:\n{_trunc(R.get('debate',''), 2500)}"
    )

    # Step 7 — Initial PRD Draft
    res = _run_agent("prd_author", 7,
        task_prompt=(
            "Write the comprehensive PRD incorporating ALL analysis and debate findings. "
            "Address every issue raised. Include acceptance criteria. "
            "Must be engineering-kickoff ready."
        ),
        context=prd_ctx,
        on_event=on_event)
    save("prd_draft", 7, res)

    current_prd = res
    iteration = 0
    qg_history = []     # list of {iteration, score, verdict, feedback}
    best_prd = res       # track best-scoring version
    best_score = 0       # monotonic: never go below this

    while True:
        iteration += 1

        # Emit iteration_start event so UI can show which cycle we are on
        _emit(on_event, "iteration_start", 8, "Quality Gate", json.dumps({
            "iteration": iteration,
            "max_iterations": MAX_QG_ITERATIONS,
        }))

        # Step 8 — Quality Gate evaluates current PRD
        qg_ctx = (
            f"ORIGINAL IDEA: {raw_idea}\n\n"
            f"PRD TO REVIEW (iteration {iteration}):\n{_trunc(current_prd, 5000)}\n\n"
            f"DEBATE FINDINGS TO CHECK AGAINST:\n{_trunc(R.get('debate',''), 2000)}"
        )
        if iteration > 1:
            qg_ctx += f"\n\nPREVIOUS QG FEEDBACK (iteration {iteration-1}):\n{_trunc(qg_history[-1]['feedback'], 1500)}"
            qg_ctx += f"\nPrevious score: {qg_history[-1]['score']}/10 — the author was asked to revise. Check if the revision addressed the feedback."

        qg_task = (
            f"Review this PRD ruthlessly (iteration {iteration}/{MAX_QG_ITERATIONS}). "
            "Evaluate completeness, clarity, consistency, feasibility. "
            "Score 1-10.\n\n"
            "YOU MUST structure your response with these EXACT sections:\n"
            "1. SCORE: X/10\n"
            "2. VERDICT: APPROVED or REVISION_NEEDED\n"
        )
        if iteration > 1:
            qg_task += (
                "3. IMPROVEMENTS SINCE LAST ITERATION:\n"
                "   - List each specific improvement as a bullet point\n"
                "   - Be concrete: 'Added error handling section' not 'Better'\n"
                "4. REMAINING ISSUES:\n"
                "   - List each remaining gap or weakness\n"
                "   - Explain WHY it's still a problem\n"
                "5. SCORE JUSTIFICATION:\n"
                "   - Explain exactly WHY you chose this score\n"
                "   - If improvements were found but score didn't increase, explain what offset the gains\n"
                "   - Reference the previous score\n"
            )
        else:
            qg_task += (
                "3. STRENGTHS:\n   - List what's good about this PRD\n"
                "4. REMAINING ISSUES:\n   - List each gap or weakness with WHY\n"
            )
        if iteration > 1:
            qg_task += (
                f"\nIMPORTANT — COMPARATIVE SCORING:\n"
                f"Previous score: {qg_history[-1]['score']}/10.\n"
                f"If issues were FIXED, the score MUST increase.\n"
                f"Score can ONLY stay the same if improvements are offset by NEW problems.\n"
            )
        qg_task += "\nDo NOT rubber-stamp."

        qg_res = _run_agent("quality_gate", 8,
            task_prompt=qg_task,
            context=qg_ctx,
            on_event=on_event)
        save("qg", 8, qg_res)

        raw_score, verdict = _parse_qg_score(qg_res)
        improvements, remaining_issues, score_justification = _parse_qg_improvements(qg_res)
        score = raw_score
        score_note = ""

        # ── Monotonic best-version tracking ──
        if score >= best_score:
            best_score = score
            best_prd = current_prd
        else:
            # Score dropped — revert to best version
            score_note = f"Raw score was {raw_score}/10 (dropped from {best_score}/10). Kept best version to prevent regression."
            print(f"   ⚠ Score dropped {raw_score} < {best_score}. Keeping best version.")
            current_prd = best_prd
            score = best_score  # report the effective score
            verdict = "APPROVED" if score >= QG_PASS_THRESHOLD else "REVISION_NEEDED"

        qg_history.append({
            "iteration": iteration,
            "score": score,
            "verdict": verdict,
            "feedback": qg_res,
            "improvements": improvements,
            "remaining_issues": remaining_issues,
            "score_justification": score_justification,
            "score_note": score_note,
            "raw_score": raw_score,
        })

        # Emit iteration result so UI can update badges and history
        _emit(on_event, "iteration_result", 8, "Quality Gate", json.dumps({
            "iteration": iteration,
            "max_iterations": MAX_QG_ITERATIONS,
            "score": score,
            "verdict": verdict,
            "history": qg_history,
        }))

        print(f"   QG iteration {iteration}: score={score}/10, verdict={verdict}")

        # APPROVED — break out of loop
        if verdict == "APPROVED":
            if iteration == 1:
                # First-pass approval — still run revision step once as polish
                rev_ctx = (
                    f"YOUR ORIGINAL PRD:\n{_trunc(current_prd, 4000)}\n\n"
                    f"QUALITY GATE FEEDBACK (APPROVED with minor suggestions):\n{_trunc(qg_res, 3000)}\n\n"
                    f"DEBATE SUMMARY:\n{_trunc(R.get('debate',''), 2000)}"
                )
                res = _run_agent("prd_reviser", 9,
                    task_prompt=(
                        "The Quality Gate APPROVED your PRD but may have minor suggestions. "
                        "Polish the PRD incorporating any minor feedback. "
                        "Rewrite the FULL PRD with 'Revision Notes' at top."
                    ),
                    context=rev_ctx,
                    on_event=on_event)
                save("prd_final", 9, res)
                current_prd = res
            else:
                # Already revised — use the current PRD as final
                save("prd_final", 9, current_prd)
            break

        # REVISION_NEEDED — check if we can iterate more
        if iteration >= MAX_QG_ITERATIONS:
            # Max iterations reached — force accept with note
            _emit(on_event, "iteration_result", 8, "Quality Gate", json.dumps({
                "iteration": iteration,
                "max_iterations": MAX_QG_ITERATIONS,
                "score": score,
                "verdict": "FORCE_ACCEPTED",
                "history": qg_history,
                "note": f"Max iterations ({MAX_QG_ITERATIONS}) reached. Proceeding with best version.",
            }))
            print(f"   Max QG iterations ({MAX_QG_ITERATIONS}) reached — force accepting.")
            save("prd_final", 9, current_prd)
            break

        # ── Revision needed: Step 9 revises, then loop back to Step 8 ──

        _emit(on_event, "iteration_start", 9, "PRD Revision", json.dumps({
            "iteration": iteration,
            "max_iterations": MAX_QG_ITERATIONS,
        }))

        rev_ctx = (
            f"YOUR PRD (iteration {iteration}):\n{_trunc(best_prd, 4000)}\n\n"
            f"QUALITY GATE FEEDBACK (score {best_score}/10 — REVISION_NEEDED):\n{_trunc(qg_res, 3000)}\n\n"
            f"DEBATE SUMMARY:\n{_trunc(R.get('debate',''), 2000)}"
        )
        if len(qg_history) > 1:
            rev_ctx += f"\n\nPrevious iteration scores: {', '.join(str(h['score'])+'/10' for h in qg_history[:-1])}"

        res = _run_agent("prd_reviser", 9,
            task_prompt=(
                f"The Quality Gate REJECTED your PRD (score {best_score}/10, iteration {iteration}/{MAX_QG_ITERATIONS}). "
                "Address EVERY piece of feedback point by point. "
                "IMPORTANT: You MUST preserve all good aspects of the previous version. "
                "Only ADD or IMPROVE — never remove content that was already good. "
                "Rewrite the FULL PRD incorporating improvements. "
                "Add 'Revision Notes' at top listing what changed and why.\n"
                "This must score higher on the next review."
            ),
            context=rev_ctx,
            on_event=on_event)
        save("prd_revision_" + str(iteration), 9, res)

        current_prd = res
        # Loop back → Quality Gate will re-evaluate

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 5 — FINAL COMPILATION
    # ══════════════════════════════════════════════════════════════════

    final_ctx = (
        f"REVISED PRD:\n{_trunc(R['prd_final'], 5000)}\n\n"
        f"QUALITY GATE ITERATIONS: {len(qg_history)} cycle(s)\n"
        + "\n".join(f"  Iteration {h['iteration']}: score {h['score']}/10 — {h['verdict']}" for h in qg_history)
        + f"\n\nFINAL QG FEEDBACK:\n{_trunc(qg_history[-1]['feedback'], 2000)}\n\n"
        f"DEBATE HIGHLIGHTS:\n{_trunc(R.get('debate',''), 2000)}\n\n"
        f"PATTERNS:\n{_trunc(R.get('patterns',''), 1500)}\n\n"
        f"SCOPE:\n{_trunc(R.get('scope',''), 1500)}"
    )

    res = _run_agent("mcp_orchestrator", 10,
        task_prompt=(
            "Compile the FINAL investor-grade deliverable:\n"
            "1. Executive Summary\n"
            "2. Full Consolidated PRD (from revised version — do NOT summarise)\n"
            "3. Surfaced Assumptions with risk levels\n"
            "4. Cross-Agent Debate Highlights\n"
            "5. Quality Gate Results\n"
            "6. Future Misalignment Areas\n"
            "7. Clarity Confidence Score"
        ),
        context=final_ctx,
        on_event=on_event)
    save("final", 10, res)

    elapsed = time.time() - t0
    print(f"\n   Pipeline complete: {elapsed:.1f}s  |  LLM calls: {len(R)}")

    return {
        "final_output": res,
        "step_outputs": sorted(step_outputs, key=lambda x: x["step"]),
        "pipeline_state": {
            "raw_idea": raw_idea,
            "prd_final": R.get("prd_final", ""),
            "debate": R.get("debate", ""),
            "patterns": R.get("patterns", ""),
            "consistency": R.get("consistency", ""),
            "scope": R.get("scope", ""),
            "qg_history": qg_history,
        },
    }


# =============================================================================
#  USER FEEDBACK REVISION  (post-pipeline human-in-the-loop)
# =============================================================================
def run_feedback_revision(user_feedback, pipeline_state, on_event=None):
    """
    Run a revision cycle based on user feedback.
    Steps: PRD Reviser (9) → Quality Gate (8) → return.
    """
    current_prd = pipeline_state["prd_final"]
    qg_history = pipeline_state.get("qg_history", [])
    best_score = max((h["score"] for h in qg_history), default=0)
    iteration = len(qg_history) + 1

    # Emit that user-feedback revision is starting
    _emit(on_event, "user_feedback_start", 9, "PRD Revision", json.dumps({
        "iteration": iteration,
        "source": "user",
        "feedback_preview": user_feedback[:500],
    }))

    # Step 9 — Revise PRD based on user feedback
    rev_ctx = (
        f"YOUR CURRENT PRD:\n{_trunc(current_prd, 4000)}\n\n"
        f"USER FEEDBACK:\n{user_feedback}\n\n"
        f"DEBATE SUMMARY:\n{_trunc(pipeline_state.get('debate',''), 2000)}"
    )

    revised = _run_agent("prd_reviser", 9,
        task_prompt=(
            "A human reviewer provided feedback on your PRD. "
            "Address EVERY point they raised. "
            "IMPORTANT: You MUST preserve all good aspects of the previous version. "
            "Only ADD or IMPROVE — never remove content that was already good. "
            "Rewrite the FULL PRD incorporating their improvements. "
            "Add 'User Feedback Revision Notes' at top listing what changed and why."
        ),
        context=rev_ctx,
        on_event=on_event)

    # Step 8 — QG evaluates revised PRD
    _emit(on_event, "iteration_start", 8, "Quality Gate", json.dumps({
        "iteration": iteration,
        "max_iterations": iteration,
        "source": "user_feedback",
    }))

    # Build QG context with previous score & issues for proper comparative evaluation
    qg_ctx = (
        f"ORIGINAL IDEA: {pipeline_state['raw_idea']}\n\n"
        f"PRD TO REVIEW (after user feedback revision, iteration {iteration}):\n{_trunc(revised, 5000)}\n\n"
        f"USER FEEDBACK THAT WAS ADDRESSED:\n{_trunc(user_feedback, 1500)}\n\n"
        f"DEBATE FINDINGS:\n{_trunc(pipeline_state.get('debate',''), 2000)}"
    )

    # Include previous QG evaluation so the QG can compare & score incrementally
    prev_entry = qg_history[-1] if qg_history else None
    if prev_entry:
        prev_score = prev_entry.get("score", "?")
        prev_feedback = prev_entry.get("feedback", "")
        prev_issues = prev_entry.get("remaining_issues", [])
        issues_text = "\n".join(f"  - {iss}" for iss in prev_issues) if prev_issues else "(none listed)"
        qg_ctx += (
            f"\n\n── PREVIOUS QG EVALUATION (iteration {prev_entry.get('iteration','?')}) ──\n"
            f"Previous score: {prev_score}/10\n"
            f"Previous remaining issues:\n{issues_text}\n"
            f"Previous full feedback:\n{_trunc(prev_feedback, 1500)}\n"
            f"── END PREVIOUS QG ──"
        )

    qg_task = (
        f"Review this revised PRD (iteration {iteration}). A human reviewer provided feedback and the author revised it. "
        "Check if the user's feedback was properly addressed. "
    )
    if prev_entry:
        qg_task += (
            f"\n\nIMPORTANT — COMPARATIVE SCORING RULES:\n"
            f"The PREVIOUS iteration scored {prev_entry.get('score','?')}/10.\n"
            f"You MUST compare the new PRD against the previous version's issues listed above.\n"
            f"- If issues from the previous iteration were FIXED, the score MUST increase.\n"
            f"- If new content was added that improves the PRD, the score MUST increase.\n"
            f"- The score can ONLY stay the same if improvements are offset by NEW problems.\n"
            f"- The score should NEVER decrease if improvements were made and no new issues appeared.\n\n"
        )
    qg_task += (
        "Score 1-10.\n\n"
        "YOU MUST structure your response with these EXACT sections:\n"
        "1. SCORE: X/10\n"
        "2. VERDICT: APPROVED or REVISION_NEEDED\n"
        "3. IMPROVEMENTS SINCE LAST ITERATION:\n"
        "   - List each specific improvement as a bullet point\n"
        "   - Be concrete about what changed\n"
        "4. REMAINING ISSUES:\n"
        "   - List each remaining gap or weakness\n"
        "   - Explain WHY it's still a problem\n"
        "5. SCORE JUSTIFICATION:\n"
        "   - Explain exactly WHY you chose this score\n"
        "   - If the score did NOT increase despite improvements, explain specifically what offset the gains\n"
        "   - Reference the previous score and what changed\n"
    )

    qg_res = _run_agent("quality_gate", 8,
        task_prompt=qg_task,
        context=qg_ctx,
        on_event=on_event)

    raw_score, verdict = _parse_qg_score(qg_res)
    improvements, remaining_issues, score_justification = _parse_qg_improvements(qg_res)
    score = raw_score
    score_note = ""

    # ── Same-score explanation ──
    prev_score_val = prev_entry.get("score", 0) if prev_entry else 0
    if prev_entry and score == prev_score_val and improvements:
        if score_justification:
            score_note = "Score unchanged despite improvements. QG justification: " + " ".join(score_justification[:2])
        else:
            score_note = (f"Score stayed at {score}/10 despite {len(improvements)} improvement(s). "
                         f"New issues may have offset the gains, or improvements were minor.")

    # ── Monotonic best-version tracking ──
    if score >= best_score:
        best_score = score
        best_prd = revised
    else:
        score_note = f"Raw score was {raw_score}/10 (dropped from {best_score}/10). Kept best version to prevent regression."
        print(f"   ⚠ Score dropped {raw_score} < {best_score}. Keeping best version.")
        revised = current_prd
        score = best_score
        best_prd = current_prd
        verdict = "APPROVED" if score >= QG_PASS_THRESHOLD else "REVISION_NEEDED"

    qg_history.append({
        "iteration": iteration,
        "score": score,
        "verdict": verdict,
        "feedback": qg_res,
        "source": "user_feedback",
        "improvements": improvements,
        "remaining_issues": remaining_issues,
        "score_justification": score_justification,
        "score_note": score_note,
        "raw_score": raw_score,
    })

    _emit(on_event, "iteration_result", 8, "Quality Gate", json.dumps({
        "iteration": iteration,
        "max_iterations": iteration,
        "score": score,
        "verdict": verdict,
        "history": qg_history,
        "source": "user_feedback",
    }))

    print(f"   User-feedback revision: score={score}/10, verdict={verdict}")

    # Update the pipeline state in-place
    pipeline_state["prd_final"] = best_prd if score >= best_score else current_prd
    pipeline_state["qg_history"] = qg_history

    return {
        "revised_prd": best_prd if score >= best_score else current_prd,
        "qg_result": qg_res,
        "score": score,
        "verdict": verdict,
        "qg_history": qg_history,
    }


# =============================================================================
#  CLI
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        raw_idea = " ".join(sys.argv[1:])
    else:
        print("\n" + "=" * 60)
        print("  Product Alignment System — Multi-Agent Orchestration")
        print("=" * 60)
        raw_idea = input("\n>> Enter your raw product idea:\n> ").strip()
        if not raw_idea:
            print("No idea provided — exiting.")
            sys.exit(0)

    print("\n" + "=" * 60)
    print("  Product Alignment System — Multi-Agent Orchestration")
    print(f"\n  Idea   : {raw_idea}")
    print(f"  Model  : {MODEL}")
    print(f"  Engine : True multi-agent | Debate + Quality Gate + Revision")
    print(f"  Steps  : {TOTAL_STEPS}")
    print("=" * 60 + "\n")

    try:
        result = run_pipeline(raw_idea)
        print("\n" + "=" * 80)
        print("  FINAL OUTPUT")
        print("=" * 80)
        print(result["final_output"])
    except Exception as e:
        print(f"\n  Error: {type(e).__name__}: {e}")
        sys.exit(1)
