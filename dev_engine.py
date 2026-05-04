"""
AI-Powered Developer Pipeline — Multi-Agent Orchestration
===========================================================
Takes an approved PRD and produces architecture + code scaffolding
through 11 specialised agents working in coordinated phases:

  Phase 1  — Interpretation    (steps 1-2: PRD Interpreter → Functional Decomposer)
  Phase 2  — Parallel Analysis (steps 3/4/5: Feasibility + Dependencies + Complexity)
  Phase 3  — Design            (steps 6/7: Architecture Designer + Tech Stack Selector)
  Phase 4  — Contract & Safety (steps 8/9: API Contracts + Failure Mode Analysis)
  Phase 5  — DevOps            (step 10: CI/CD + Deployment)
  Phase 6  — Quality-Gated     (step 11: Orchestrator compiles + QG loop)

Produces: architecture scaffolding + file tree in a local sandbox folder
"""

import os
import sys
import json
import time
import re as _re
from pathlib import Path
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
MAX_TOKENS = 4096
MAX_TOKENS_LONG = 8192
TEMPERATURE = 0.7
API_KEY = CEREBRAS_API_KEY

DEV_TOTAL_STEPS = 11
DEV_MAX_QG_ITERATIONS = 3
DEV_QG_PASS_THRESHOLD = 7

LONG_TOKEN_AGENTS = {
    "architecture_designer", "api_contract_generator",
    "devops_agent", "dev_orchestrator",
}

SANDBOX_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sandbox_projects")

litellm.set_verbose = False


# =============================================================================
#  STEP METADATA  (for UI)
# =============================================================================
DEV_STEP_INFO = [
    {"step": 1,  "name": "PRD Interpreter",        "desc": "Parse PRD into actionable engineering requirements",               "phase": 1},
    {"step": 2,  "name": "Functional Decomposer",   "desc": "Break down into modules, features, and services",                 "phase": 1},
    {"step": 3,  "name": "Feasibility Analyzer",    "desc": "Validate technical feasibility of each component",                "phase": 2},
    {"step": 4,  "name": "Dependency Mapper",        "desc": "Map inter-module dependencies and external libraries",            "phase": 2},
    {"step": 5,  "name": "Complexity Estimator",     "desc": "Estimate effort and complexity per module",                       "phase": 2},
    {"step": 6,  "name": "Architecture Designer",    "desc": "Design system architecture — patterns, layers, data flow",        "phase": 3},
    {"step": 7,  "name": "Tech Stack Selector",      "desc": "Choose optimal tech stack based on requirements",                 "phase": 3},
    {"step": 8,  "name": "API Contract Generator",   "desc": "Define API specs, endpoints, request/response schemas",           "phase": 4},
    {"step": 9,  "name": "Failure Mode Analyzer",    "desc": "Identify failure points, edge cases, and fallback strategies",    "phase": 4},
    {"step": 10, "name": "DevOps / CI-CD Agent",     "desc": "Generate pipeline configs, Dockerfiles, deployment specs",        "phase": 5},
    {"step": 11, "name": "Dev MCP Orchestrator",     "desc": "Compile final architecture, scaffold code, quality gate",         "phase": 6},
]


# =============================================================================
#  AGENT DEFINITIONS
# =============================================================================
DEV_AGENTS = {
    "prd_interpreter": {
        "role": "PRD Interpreter",
        "system": (
            "You are a senior engineering lead who translates Product Requirement Documents "
            "into precise engineering requirements. Extract: functional requirements (FRs), "
            "non-functional requirements (NFRs — performance, scalability, security, etc.), "
            "user stories with acceptance criteria, data entities and relationships, "
            "integration points, and constraints/assumptions. "
            "Output a structured engineering requirements spec."
        ),
    },
    "functional_decomposer": {
        "role": "Functional Decomposer",
        "system": (
            "You are a system decomposition expert. Given engineering requirements, "
            "break the product into: discrete modules/services, feature boundaries, "
            "shared components, data stores, and external integrations. "
            "Output a module map with clear boundaries, responsibilities, and interfaces. "
            "Use a table format: Module | Responsibility | Inputs | Outputs | Dependencies."
        ),
    },
    "feasibility_analyzer": {
        "role": "Feasibility Analyzer",
        "system": (
            "You are a technical feasibility expert. For each module/feature, assess: "
            "Is it technically achievable with current tech? What are the risks? "
            "Any regulatory or compliance issues? Estimated difficulty (1-5). "
            "Flag any requirements that are unrealistic or need re-scoping. "
            "Output a feasibility matrix with RAG (Red/Amber/Green) ratings."
        ),
    },
    "dependency_mapper": {
        "role": "Dependency Mapper",
        "system": (
            "You are a dependency analysis expert. Map: inter-module dependencies, "
            "external library/SDK dependencies, third-party API dependencies, "
            "data flow dependencies, and build/deploy order dependencies. "
            "Identify circular dependencies and suggest resolution. "
            "Output a dependency graph description and a suggested build order."
        ),
    },
    "complexity_estimator": {
        "role": "Complexity Estimator",
        "system": (
            "You are a software estimation expert. For each module, estimate: "
            "lines of code (rough), story points, developer-days, "
            "technical complexity (low/medium/high), and integration complexity. "
            "Identify the critical path and bottleneck modules. "
            "Output a complexity matrix and suggest a phased delivery plan."
        ),
    },
    "architecture_designer": {
        "role": "Architecture Designer",
        "system": (
            "You are a principal software architect. Design the system architecture: "
            "architecture pattern (microservices/monolith/serverless/hybrid), "
            "layer structure, data flow diagrams, key design decisions with rationale, "
            "scalability strategy, state management, caching strategy, "
            "and database design (schema overview, indexing strategy). "
            "Output a comprehensive Architecture Design Document (ADD)."
        ),
    },
    "tech_stack_selector": {
        "role": "Tech Stack Selector",
        "system": (
            "You are a technology strategist. Based on the architecture and requirements, "
            "recommend: programming language(s), framework(s), database(s), "
            "message queue/event system, caching layer, hosting/cloud platform, "
            "monitoring stack, and development tools. "
            "For each choice explain WHY and list alternatives considered. "
            "Output a Tech Stack Decision Matrix."
        ),
    },
    "api_contract_generator": {
        "role": "API Contract Generator",
        "system": (
            "You are an API design expert. Generate: REST/GraphQL endpoint definitions, "
            "request/response schemas (JSON), authentication & authorization flow, "
            "error codes and handling, rate limiting strategy, versioning strategy, "
            "and WebSocket events if applicable. "
            "Output an API Contract Document with concrete endpoint specs "
            "(method, path, request body, response body, status codes)."
        ),
    },
    "failure_mode_analyzer": {
        "role": "Failure Mode Analyzer",
        "system": (
            "You are a reliability engineer and security analyst. For each component, identify: "
            "failure modes (what can go wrong), impact (severity 1-5), "
            "detection method, mitigation/fallback strategy, "
            "security vulnerabilities (OWASP top 10), and edge cases. "
            "Output a Failure Mode & Effects Analysis (FMEA) table "
            "and a Security Threat Model."
        ),
    },
    "devops_agent": {
        "role": "DevOps / CI-CD Engineer",
        "system": (
            "You are a senior DevOps engineer. Generate: "
            "CI/CD pipeline configuration (GitHub Actions YAML), "
            "Dockerfile(s), docker-compose.yml, "
            "environment configuration (.env template), "
            "deployment strategy (blue-green/canary/rolling), "
            "monitoring & alerting setup, "
            "infrastructure-as-code snippets, "
            "and a README with setup instructions. "
            "Output actual config file contents that can be saved directly."
        ),
    },
    "dev_orchestrator": {
        "role": "Dev MCP Orchestrator",
        "system": (
            "You are the master technical lead who compiles all agent outputs "
            "into a final, coherent Architecture & Scaffolding Package. "
            "Synthesize: project directory structure (file tree), "
            "key source code scaffolds (entry points, models, routes, configs), "
            "setup scripts, environment configs, CI/CD configs, "
            "and a comprehensive Developer Onboarding Guide. "
            "Mark each file with ```filename.ext markers so they can be extracted. "
            "IMPORTANT: Every code file must have proper scaffolding with "
            "imports, class/function stubs, comments explaining what goes where, "
            "and TODO markers for implementation. "
            "The output must be directly usable by a developer to start coding."
        ),
    },
}


# =============================================================================
#  LLM CALL (shared with product side pattern)
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
                    on_event("dev_retry", step_num, step_name, msg)
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
def _run_dev_agent(agent_key, step_num, task_prompt, context="", on_event=None):
    agent = DEV_AGENTS[agent_key]
    step_name = DEV_STEP_INFO[step_num - 1]["name"]

    if context:
        user_msg = (
            "CONTEXT:\n" + ("-" * 40) + "\n" + context + "\n" + ("-" * 40)
            + "\n\nTASK:\n" + task_prompt
        )
    else:
        user_msg = task_prompt

    _emit(on_event, "dev_start", step_num, step_name, "")
    _emit(on_event, "dev_agent_info", step_num, step_name, json.dumps({
        "role": agent["role"],
        "system_prompt": agent["system"],
        "task": task_prompt,
        "context_preview": context[:800] if context else "(none)",
        "context_length": len(context) if context else 0,
    }))
    print(f"  DEV STEP {step_num}/{DEV_TOTAL_STEPS} — {step_name} [{agent_key}]")

    tokens = MAX_TOKENS_LONG if agent_key in LONG_TOKEN_AGENTS else MAX_TOKENS
    result = _call_llm(
        system_prompt=agent["system"],
        user_prompt=user_msg,
        step_num=step_num,
        step_name=step_name,
        on_event=on_event,
        max_tokens=tokens,
    )

    _emit(on_event, "dev_output_chunk", step_num, step_name, result)
    _emit(on_event, "dev_done", step_num, step_name, result)
    print(f"   -> done ({len(result)} chars)")
    return result


# =============================================================================
#  QUALITY GATE — SCORE PARSER
# =============================================================================
def _parse_dev_qg_score(qg_text):
    score = None
    verdict = "REVISION_NEEDED"
    patterns = [
        r'(?:quality\s*)?score[:\s—\-]+(\d+)\s*/?\s*10',
        r'(\d+)\s*/\s*10',
        r'(?:quality\s*)?score[:\s—\-]+(\d+)',
    ]
    for pat in patterns:
        m = _re.search(pat, qg_text, _re.IGNORECASE)
        if m:
            score = int(m.group(1))
            break
    if score is None:
        score = 5
    if score >= DEV_QG_PASS_THRESHOLD:
        verdict = "APPROVED"
    return score, verdict


def _parse_dev_qg_improvements(qg_text):
    improvements = []
    remaining_issues = []
    score_justification = []
    lines = qg_text.split('\n')
    section = None
    for line in lines:
        stripped = line.strip()
        upper = stripped.upper()
        if any(k in upper for k in ['IMPROVEMENT', 'WHAT IMPROVED', 'PROGRESS']):
            section = 'improvements'; continue
        if any(k in upper for k in ['REMAINING', 'ISSUES', 'GAPS', 'WEAKNESSES']):
            section = 'issues'; continue
        if any(k in upper for k in ['SCORE JUSTIFICATION', 'SCORING RATIONALE', 'WHY THIS SCORE']):
            section = 'justification'; continue
        if any(k in upper for k in ['SCORE:', 'VERDICT:']) and section != 'justification':
            section = None; continue
        if stripped and section:
            if stripped.startswith('-') or stripped.startswith('•') or stripped.startswith('*') or _re.match(r'^\d+[.)\s]', stripped):
                text = _re.sub(r'^[-•*\d.)+\s]+', '', stripped).strip()
                if text and len(text) > 5:
                    if section == 'improvements': improvements.append(text)
                    elif section == 'issues': remaining_issues.append(text)
                    elif section == 'justification': score_justification.append(text)
            elif section == 'justification' and len(stripped) > 10:
                score_justification.append(stripped)
    return improvements[:6], remaining_issues[:6], score_justification[:4]


# =============================================================================
#  SANDBOX — Extract files from orchestrator output
# =============================================================================
def _extract_scaffold_files(orchestrator_output):
    """
    Parse orchestrator output for code blocks marked with filenames.
    Patterns recognized:
      ```filename.ext     or     ```path/to/file.ext
      <code>
      ```
    Returns list of {"path": "relative/path", "content": "..."} dicts.
    """
    files = []
    # Match ```filename or ``` filename
    pattern = _re.compile(
        r'```\s*([\w./_-]+(?:\.\w+))\s*\n(.*?)```',
        _re.DOTALL
    )
    for m in pattern.finditer(orchestrator_output):
        filepath = m.group(1).strip()
        content = m.group(2)
        # Skip if it looks like a language tag without a file extension
        if '.' not in filepath and '/' not in filepath:
            continue
        files.append({"path": filepath, "content": content})

    return files


def _write_sandbox(project_name, files):
    """Write extracted files to a sandbox project folder. Returns the project path."""
    safe_name = _re.sub(r'[^\w\-]', '_', project_name.lower())[:50]
    project_dir = os.path.join(SANDBOX_ROOT, safe_name)
    os.makedirs(project_dir, exist_ok=True)

    written = []
    for f in files:
        fpath = os.path.join(project_dir, f["path"])
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w", encoding="utf-8") as fp:
            fp.write(f["content"])
        written.append(f["path"])

    return project_dir, written


# =============================================================================
#  CI/CD ALERT HELPER
# =============================================================================
def _emit_cicd_alert(on_event, step_num, step_name, alert_type, message, details=""):
    """Emit a live CI/CD-style alert to the UI."""
    _emit(on_event, "dev_cicd_alert", step_num, step_name, json.dumps({
        "alert_type": alert_type,  # "success", "warning", "error", "info"
        "message": message,
        "details": details,
        "timestamp": time.time(),
    }))


# =============================================================================
#  THE DEV PIPELINE
# =============================================================================
def run_dev_pipeline(prd_text, project_name="project", on_event=None):
    """
    Multi-agent developer pipeline:
      Phase 1 — Interpretation    (steps 1-2: sequential)
      Phase 2 — Parallel Analysis (steps 3/4/5: simultaneous)
      Phase 3 — Design            (steps 6/7: sequential)
      Phase 4 — Contract & Safety (steps 8/9: parallel)
      Phase 5 — DevOps            (step 10)
      Phase 6 — Quality-Gated     (step 11: orchestrator + QG loop)
    """
    t0 = time.time()
    R = {}
    step_outputs = []

    def save(key, step_num, result):
        R[key] = result
        step_outputs.append({"step": step_num, "key": key, "chars": len(result)})

    _emit(on_event, "dev_pipeline_start", 0, "Dev Pipeline",
          json.dumps({"total_steps": DEV_TOTAL_STEPS, "project_name": project_name}))

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 1 — Interpretation (Sequential)
    # ══════════════════════════════════════════════════════════════════
    _emit_cicd_alert(on_event, 1, "PRD Interpreter", "info",
                     "Phase 1: Starting PRD interpretation...")

    # Step 1 — PRD Interpreter
    res = _run_dev_agent("prd_interpreter", 1,
        task_prompt=(
            "Parse this approved PRD into structured engineering requirements. "
            "Extract ALL functional requirements (FRs), non-functional requirements (NFRs), "
            "user stories with acceptance criteria, data entities, "
            "integration points, and constraints. "
            "Be exhaustive — nothing from the PRD should be lost."
        ),
        context=f"APPROVED PRD:\n{_trunc(prd_text, 6000)}",
        on_event=on_event)
    save("eng_requirements", 1, res)

    _emit_cicd_alert(on_event, 1, "PRD Interpreter", "success",
                     f"Engineering requirements extracted ({len(res)} chars)",
                     "FRs, NFRs, user stories, data entities identified")

    # Step 2 — Functional Decomposer
    res = _run_dev_agent("functional_decomposer", 2,
        task_prompt=(
            "Decompose these engineering requirements into discrete modules/services. "
            "Define: module boundaries, responsibilities, interfaces, "
            "shared components, and data stores. "
            "Output a clear module map table."
        ),
        context=(
            f"ENGINEERING REQUIREMENTS:\n{_trunc(R['eng_requirements'], 4000)}\n\n"
            f"ORIGINAL PRD (summary):\n{_trunc(prd_text, 2000)}"
        ),
        on_event=on_event)
    save("module_map", 2, res)

    _emit_cicd_alert(on_event, 2, "Functional Decomposer", "success",
                     "Module decomposition complete",
                     "System broken into discrete modules with clear boundaries")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 2 — Parallel Analysis (Steps 3/4/5 simultaneously)
    # ══════════════════════════════════════════════════════════════════
    _emit_cicd_alert(on_event, 3, "Parallel Analysis", "info",
                     "Phase 2: Running feasibility, dependency, and complexity analysis in parallel...")

    analysis_ctx = (
        f"ENGINEERING REQUIREMENTS:\n{_trunc(R['eng_requirements'], 3000)}\n\n"
        f"MODULE MAP:\n{_trunc(R['module_map'], 3000)}"
    )

    def run_feasibility():
        return _run_dev_agent("feasibility_analyzer", 3,
            task_prompt=(
                "Assess technical feasibility for EVERY module. "
                "Rate each: RAG (Red/Amber/Green), risk level, "
                "and flag any that need re-scoping."
            ),
            context=analysis_ctx, on_event=on_event)

    def run_dependencies():
        return _run_dev_agent("dependency_mapper", 4,
            task_prompt=(
                "Map ALL dependencies: inter-module, external libraries, "
                "third-party APIs, data flow order, and build order. "
                "Flag circular dependencies."
            ),
            context=analysis_ctx, on_event=on_event)

    def run_complexity():
        return _run_dev_agent("complexity_estimator", 5,
            task_prompt=(
                "Estimate complexity for each module: story points, "
                "developer-days, LOC estimate, technical difficulty. "
                "Identify the critical path and bottleneck modules. "
                "Suggest a phased delivery plan."
            ),
            context=analysis_ctx, on_event=on_event)

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_feasibility): "feasibility",
            executor.submit(run_dependencies): "dependencies",
            executor.submit(run_complexity): "complexity",
        }
        for future in as_completed(futures):
            key = futures[future]
            result = future.result()
            step_num = {"feasibility": 3, "dependencies": 4, "complexity": 5}[key]
            save(key, step_num, result)
            _emit_cicd_alert(on_event, step_num, DEV_STEP_INFO[step_num-1]["name"],
                             "success", f"{key.title()} analysis complete",
                             f"Output: {len(result)} chars")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 3 — Design (Sequential: Architecture then Tech Stack)
    # ══════════════════════════════════════════════════════════════════
    _emit_cicd_alert(on_event, 6, "Architecture Designer", "info",
                     "Phase 3: Designing system architecture...")

    design_ctx = (
        f"ENGINEERING REQUIREMENTS:\n{_trunc(R['eng_requirements'], 2500)}\n\n"
        f"MODULE MAP:\n{_trunc(R['module_map'], 2500)}\n\n"
        f"FEASIBILITY:\n{_trunc(R.get('feasibility',''), 2000)}\n\n"
        f"DEPENDENCIES:\n{_trunc(R.get('dependencies',''), 2000)}\n\n"
        f"COMPLEXITY:\n{_trunc(R.get('complexity',''), 2000)}"
    )

    # Step 6 — Architecture Designer
    res = _run_dev_agent("architecture_designer", 6,
        task_prompt=(
            "Design the complete system architecture. Include: "
            "architecture pattern choice (with rationale), layer structure, "
            "data flow, database design, caching strategy, "
            "scalability approach, and key design decisions."
        ),
        context=design_ctx, on_event=on_event)
    save("architecture", 6, res)

    _emit_cicd_alert(on_event, 6, "Architecture Designer", "success",
                     "Architecture design complete",
                     "Patterns, layers, data flow, and DB design defined")

    # Step 7 — Tech Stack Selector
    res = _run_dev_agent("tech_stack_selector", 7,
        task_prompt=(
            "Based on the architecture and requirements, recommend the optimal tech stack. "
            "For each choice: language, framework, database, hosting, monitoring — "
            "explain WHY and list alternatives considered."
        ),
        context=(
            f"ARCHITECTURE:\n{_trunc(R['architecture'], 3000)}\n\n"
            f"REQUIREMENTS:\n{_trunc(R['eng_requirements'], 2000)}\n\n"
            f"FEASIBILITY:\n{_trunc(R.get('feasibility',''), 1500)}"
        ),
        on_event=on_event)
    save("tech_stack", 7, res)

    _emit_cicd_alert(on_event, 7, "Tech Stack Selector", "success",
                     "Tech stack selected",
                     "Languages, frameworks, databases, and tools chosen")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 4 — Contract & Safety (Steps 8/9 parallel)
    # ══════════════════════════════════════════════════════════════════
    _emit_cicd_alert(on_event, 8, "API + Safety", "info",
                     "Phase 4: Generating API contracts and failure mode analysis in parallel...")

    contract_ctx = (
        f"ARCHITECTURE:\n{_trunc(R['architecture'], 3000)}\n\n"
        f"TECH STACK:\n{_trunc(R['tech_stack'], 2000)}\n\n"
        f"MODULE MAP:\n{_trunc(R['module_map'], 2000)}\n\n"
        f"REQUIREMENTS:\n{_trunc(R['eng_requirements'], 2000)}"
    )

    def run_api_contracts():
        return _run_dev_agent("api_contract_generator", 8,
            task_prompt=(
                "Generate complete API contracts for all endpoints. "
                "Include: HTTP method, path, request/response JSON schemas, "
                "auth requirements, error codes, rate limits, and versioning."
            ),
            context=contract_ctx, on_event=on_event)

    def run_failure_modes():
        return _run_dev_agent("failure_mode_analyzer", 9,
            task_prompt=(
                "Perform failure mode analysis and security threat modeling. "
                "For each component: failure modes, severity, detection, mitigation. "
                "Include OWASP top 10 relevance and edge cases."
            ),
            context=contract_ctx, on_event=on_event)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(run_api_contracts): ("api_contracts", 8),
            executor.submit(run_failure_modes): ("failure_modes", 9),
        }
        for future in as_completed(futures):
            key, snum = futures[future]
            result = future.result()
            save(key, snum, result)
            _emit_cicd_alert(on_event, snum, DEV_STEP_INFO[snum-1]["name"],
                             "success", f"{key.replace('_',' ').title()} complete",
                             f"Output: {len(result)} chars")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 5 — DevOps (Step 10)
    # ══════════════════════════════════════════════════════════════════
    _emit_cicd_alert(on_event, 10, "DevOps / CI-CD Agent", "info",
                     "Phase 5: Generating CI/CD pipeline and deployment configs...")

    res = _run_dev_agent("devops_agent", 10,
        task_prompt=(
            "Generate production-ready DevOps configs: "
            "1. GitHub Actions CI/CD pipeline (YAML) "
            "2. Dockerfile(s) "
            "3. docker-compose.yml "
            "4. .env.template "
            "5. Deployment strategy description "
            "6. Monitoring & alerting setup "
            "7. README setup instructions. "
            "Output the actual file contents with ```filename markers."
        ),
        context=(
            f"TECH STACK:\n{_trunc(R['tech_stack'], 2500)}\n\n"
            f"ARCHITECTURE:\n{_trunc(R['architecture'], 2500)}\n\n"
            f"MODULE MAP:\n{_trunc(R['module_map'], 2000)}\n\n"
            f"API CONTRACTS:\n{_trunc(R.get('api_contracts',''), 1500)}"
        ),
        on_event=on_event)
    save("devops", 10, res)

    _emit_cicd_alert(on_event, 10, "DevOps / CI-CD Agent", "success",
                     "CI/CD pipeline and deployment configs generated",
                     "GitHub Actions, Dockerfile, docker-compose, .env template")

    # ══════════════════════════════════════════════════════════════════
    #  PHASE 6 — Quality-Gated Orchestrator (Step 11 + QG loop)
    # ══════════════════════════════════════════════════════════════════
    _emit_cicd_alert(on_event, 11, "Dev MCP Orchestrator", "info",
                     "Phase 6: Compiling final architecture package with quality gate...")

    orch_ctx = (
        f"ORIGINAL PRD:\n{_trunc(prd_text, 2000)}\n\n"
        f"ENGINEERING REQUIREMENTS:\n{_trunc(R['eng_requirements'], 2000)}\n\n"
        f"MODULE MAP:\n{_trunc(R['module_map'], 2000)}\n\n"
        f"ARCHITECTURE:\n{_trunc(R['architecture'], 2500)}\n\n"
        f"TECH STACK:\n{_trunc(R['tech_stack'], 2000)}\n\n"
        f"API CONTRACTS:\n{_trunc(R.get('api_contracts',''), 2000)}\n\n"
        f"FAILURE MODES:\n{_trunc(R.get('failure_modes',''), 1500)}\n\n"
        f"DEVOPS CONFIGS:\n{_trunc(R.get('devops',''), 1500)}\n\n"
        f"FEASIBILITY:\n{_trunc(R.get('feasibility',''), 1000)}\n\n"
        f"DEPENDENCIES:\n{_trunc(R.get('dependencies',''), 1000)}\n\n"
        f"COMPLEXITY:\n{_trunc(R.get('complexity',''), 1000)}"
    )

    # Initial draft
    current_output = _run_dev_agent("dev_orchestrator", 11,
        task_prompt=(
            "Compile ALL agent outputs into a final Architecture & Scaffolding Package. "
            "You MUST include:\n"
            "1. Project file tree (directory structure)\n"
            "2. Key source code scaffolds with ```filename.ext markers\n"
            "3. Database schema/models\n"
            "4. API route stubs\n"
            "5. Config files (from DevOps agent)\n"
            "6. Developer Onboarding Guide\n\n"
            "Mark EVERY file with ```path/filename.ext so it can be extracted. "
            "Include proper imports, stubs, and TODO comments in each file."
        ),
        context=orch_ctx, on_event=on_event)
    save("orchestrator_draft", 11, current_output)

    # ── Quality Gate Loop ──
    iteration = 0
    qg_history = []
    best_output = current_output
    best_score = 0

    while True:
        iteration += 1

        _emit(on_event, "dev_iteration_start", 11, "Quality Gate", json.dumps({
            "iteration": iteration,
            "max_iterations": DEV_MAX_QG_ITERATIONS,
        }))

        # QG evaluates
        qg_ctx = (
            f"ORIGINAL PRD:\n{_trunc(prd_text, 2000)}\n\n"
            f"ARCHITECTURE PACKAGE TO REVIEW (iteration {iteration}):\n"
            f"{_trunc(current_output, 6000)}"
        )
        if iteration > 1:
            qg_ctx += (
                f"\n\nPREVIOUS QG FEEDBACK (iteration {iteration-1}):\n"
                f"{_trunc(qg_history[-1]['feedback'], 1500)}\n"
                f"Previous score: {qg_history[-1]['score']}/10"
            )

        qg_task = (
            f"Review this Architecture & Scaffolding Package (iteration {iteration}/{DEV_MAX_QG_ITERATIONS}). "
            "Evaluate: completeness, code quality, architecture coherence, "
            "security considerations, deployability, and developer usability. "
            "Score 1-10.\n\n"
            "YOU MUST structure your response:\n"
            "1. SCORE: X/10\n"
            "2. VERDICT: APPROVED or REVISION_NEEDED\n"
        )
        if iteration > 1:
            qg_task += (
                "3. IMPROVEMENTS SINCE LAST ITERATION:\n"
                "   - List specific improvements\n"
                "4. REMAINING ISSUES:\n"
                "   - List remaining gaps\n"
                "5. SCORE JUSTIFICATION:\n"
                "   - Explain why this score\n"
                f"\nIMPORTANT: Previous score was {qg_history[-1]['score']}/10. "
                "If issues were FIXED, score MUST increase.\n"
            )
        else:
            qg_task += (
                "3. STRENGTHS:\n   - What's good\n"
                "4. REMAINING ISSUES:\n   - What needs improvement\n"
            )
        qg_task += "\nDo NOT rubber-stamp."

        qg_res = _call_llm(
            system_prompt=(
                "You are a ruthless senior engineering reviewer. "
                "Score architecture packages for completeness, quality, and deployability. "
                "Only score >= 7 if the package is genuinely ready for a dev team."
            ),
            user_prompt=f"CONTEXT:\n{qg_ctx}\n\nTASK:\n{qg_task}",
            step_num=11, step_name="Quality Gate",
            on_event=on_event, max_tokens=MAX_TOKENS,
        )

        raw_score, verdict = _parse_dev_qg_score(qg_res)
        improvements, remaining_issues, score_justification = _parse_dev_qg_improvements(qg_res)
        score = raw_score
        score_note = ""

        # Monotonic tracking
        if score >= best_score:
            best_score = score
            best_output = current_output
        else:
            score_note = f"Raw score {raw_score}/10 dropped from {best_score}/10. Kept best version."
            current_output = best_output
            score = best_score
            verdict = "APPROVED" if score >= DEV_QG_PASS_THRESHOLD else "REVISION_NEEDED"

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

        _emit(on_event, "dev_iteration_result", 11, "Quality Gate", json.dumps({
            "iteration": iteration,
            "max_iterations": DEV_MAX_QG_ITERATIONS,
            "score": score,
            "verdict": verdict,
            "history": qg_history,
        }))

        _emit_cicd_alert(on_event, 11, "Quality Gate",
                         "success" if verdict == "APPROVED" else "warning",
                         f"QG Iteration {iteration}: {score}/10 — {verdict}",
                         f"Improvements: {len(improvements)}, Issues: {len(remaining_issues)}")

        print(f"   QG iteration {iteration}: score={score}/10, verdict={verdict}")

        if verdict == "APPROVED":
            break
        if iteration >= DEV_MAX_QG_ITERATIONS:
            qg_history[-1]["verdict"] = "FORCE_ACCEPTED"
            _emit_cicd_alert(on_event, 11, "Quality Gate", "warning",
                             f"Max iterations reached — force-accepting at {score}/10")
            break

        # Revision
        _emit(on_event, "dev_iteration_start", 11, "Revision", json.dumps({
            "iteration": iteration + 1,
            "max_iterations": DEV_MAX_QG_ITERATIONS,
        }))

        current_output = _run_dev_agent("dev_orchestrator", 11,
            task_prompt=(
                f"The quality gate scored your package {score}/10. "
                f"Address EVERY issue:\n{qg_res}\n\n"
                "Rewrite the FULL Architecture & Scaffolding Package with fixes. "
                "Preserve all good parts. Only ADD or IMPROVE."
            ),
            context=orch_ctx, on_event=on_event)

    # ══════════════════════════════════════════════════════════════════
    #  SANDBOX — Write files
    # ══════════════════════════════════════════════════════════════════
    scaffold_files = _extract_scaffold_files(best_output)
    project_dir = ""
    written_files = []
    if scaffold_files:
        project_dir, written_files = _write_sandbox(project_name, scaffold_files)
        _emit_cicd_alert(on_event, 11, "Sandbox", "success",
                         f"Scaffold written to {project_dir}",
                         f"{len(written_files)} files created: {', '.join(written_files[:10])}")
        print(f"   Sandbox: {len(written_files)} files written to {project_dir}")

    elapsed = round(time.time() - t0, 1)

    # Final complete event
    _emit(on_event, "dev_complete", 0, "Dev Pipeline Complete", json.dumps({
        "final_output": best_output,
        "step_outputs": step_outputs,
        "elapsed": elapsed,
        "score": best_score,
        "verdict": qg_history[-1]["verdict"] if qg_history else "N/A",
        "qg_history": qg_history,
        "sandbox_path": project_dir,
        "scaffold_files": written_files,
        "file_count": len(written_files),
    }))

    return {
        "final_output": best_output,
        "step_outputs": step_outputs,
        "elapsed": elapsed,
        "score": best_score,
        "verdict": qg_history[-1]["verdict"] if qg_history else "N/A",
        "qg_history": qg_history,
        "sandbox_path": project_dir,
        "scaffold_files": written_files,
        "pipeline_state": {
            "raw_prd": prd_text,
            "project_name": project_name,
            "eng_requirements": R.get("eng_requirements", ""),
            "module_map": R.get("module_map", ""),
            "architecture": R.get("architecture", ""),
            "tech_stack": R.get("tech_stack", ""),
            "api_contracts": R.get("api_contracts", ""),
            "failure_modes": R.get("failure_modes", ""),
            "devops": R.get("devops", ""),
            "final_output": best_output,
            "qg_history": qg_history,
        },
    }
