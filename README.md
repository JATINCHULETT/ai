# Autonomous AI Software Company

A **26-agent, 3-layer AI system** that turns a raw product idea into an investor-grade PRD, a full architecture & code scaffold, and a cross-layer alignment report — all through a browser UI with real-time animated flowcharts and live agent streaming.

---

## What Has Been Implemented

### Layer 1 — Product Pipeline (10 AI Agents)

Takes a raw product idea and produces a polished, investor-grade **Product Requirements Document (PRD)**.

| # | Agent | What it does |
|---|-------|-------------|
| 1 | **Product Input Analyst** | Parses the raw idea, extracts core value proposition, target audience, key features |
| 2 | **Product Interrogator** | Asks probing questions about gaps — edge cases, monetization, compliance, scalability |
| 3 | **Pattern Analyst** | Identifies industry patterns, competitor landscape, differentiation opportunities |
| 4 | **Consistency Auditor** | Checks requirements for contradictions, ambiguities, missing dependencies |
| 5 | **Scope & Priority Manager** | Ranks features by impact vs effort, defines MVP vs future phases |
| 6 | **Cross-Agent Debate Synthesiser** | Runs a structured debate across agents 3-5, resolves conflicts, produces consensus |
| 7 | **PRD Author** | Writes the full PRD document using all prior analysis |
| 8 | **Quality Gate** | Scores the PRD 1-10, identifies weaknesses, decides APPROVED or REVISION_NEEDED |
| 9 | **PRD Reviser** | Rewrites the PRD addressing Quality Gate feedback (loops up to 3 times) |
| 10 | **MCP Final Orchestrator** | Compiles the final deliverable package with executive summary |

**Key features:**
- Parallel execution (agents 3-5 run simultaneously via ThreadPoolExecutor)
- Cross-agent debate resolves conflicting recommendations
- Quality gate loop with monotonic best-score tracking (keeps the best version if scores drop)
- Human-in-the-loop: after completion, user can approve or provide feedback for another revision cycle
- "Next →" button hands the approved PRD to the Dev Pipeline

---

### Layer 2 — Developer Pipeline (11 AI Agents)

Takes the approved PRD and produces a **complete architecture blueprint + code scaffolding**.

| # | Agent | What it does |
|---|-------|-------------|
| 1 | **PRD Interpreter** | Translates product requirements into engineering requirements |
| 2 | **Functional Decomposer** | Breaks the system into modules, services, and components |
| 3 | **Feasibility Analyzer** | Evaluates technical feasibility, identifies risky areas |
| 4 | **Dependency Mapper** | Maps inter-module dependencies, identifies coupling risks |
| 5 | **Complexity Estimator** | Estimates effort per component (story points, LOC, team size) |
| 6 | **Architecture Designer** | Designs the full system architecture (patterns, layers, data flow) |
| 7 | **Tech Stack Selector** | Recommends specific technologies, frameworks, databases with justification |
| 8 | **API Contract Generator** | Produces OpenAPI/REST contracts for all service interfaces |
| 9 | **Failure Mode Analyzer** | Identifies failure modes, cascading risks, proposes circuit breakers & fallbacks |
| 10 | **DevOps / CI-CD Agent** | Generates Dockerfiles, docker-compose, CI/CD pipeline configs, .env templates |
| 11 | **Dev MCP Orchestrator** | Assembles the final architecture package, quality-gated (score ≥ 7 to pass) |

**Key features:**
- Parallel analysis (agents 3-5 and agents 8-9 run simultaneously)
- CI/CD live alert panel shows build-style notifications during execution
- Code scaffold extraction — parses code blocks from the final output and writes actual files to `sandbox_projects/`
- File tree viewer shows all generated project files
- Quality gate loop (up to 3 iterations) for the final package
- "Run Alignment Check →" button passes output to the Alignment Bridge

---

### Layer 3 — Alignment Bridge (5 AI Agents)

Compares the **product intent** (PRD) against the **dev implementation** (architecture) to detect misalignments before any code is written.

| # | Agent | What it does |
|---|-------|-------------|
| 1 | **Intent Extractor** | Extracts 10 intent dimensions from the PRD (core value, behavioral expectations, performance, UX, security, integration, scope, priority, metrics, assumptions) |
| 2 | **Implementation Analyzer** | Extracts what the architecture ACTUALLY delivers across the same 10 dimensions |
| 3 | **Deviation Detector** | Cross-compares intent vs implementation, classifies each dimension as ALIGNED / DRIFTED / COMPROMISED / VIOLATED / MISSING |
| 4 | **Risk Assessor** | Classifies deviations: NECESSARY_TRADEOFF / RISKY_COMPROMISE / INTENT_VIOLATION. Assigns risk levels (LOW / MEDIUM / HIGH / CRITICAL) |
| 5 | **Alignment Synthesiser** | Produces the final alignment report with 0-100 score, sub-scores, deviation table, verdict, and action recommendation |

**Key features:**
- Alignment score 0-100 with 5 sub-scores: Intent Preservation (25), Behavioral Fidelity (20), UX Consistency (20), Performance Alignment (20), Risk Tolerance (15)
- Threshold: score ≥ 70 = ALIGNED, below = action required
- Recommendation: ACCEPT / REVISE_DEV / REVISIT_PRODUCT / ESCALATE
- Deviation table with colored status pills and risk badges
- Lower temperature (0.4 vs 0.7) for analytical precision

---

### Web UI — Three Connected Pages

All three pages share the same dark theme with:
- **Animated SVG flowcharts** — nodes pulse gold when active, turn green when done
- **Real-time SSE streaming** — see agent output character-by-character
- **Slide-in modals** — click any agent node to see its role, system prompt, task, and live output
- **Timer + progress bar** tracking elapsed time and step completion

| Page | URL | Content |
|------|-----|---------|
| Product Pipeline | `/` | 10-node flowchart, feedback prompt, iteration timeline, PRD viewer |
| Dev Pipeline | `/dev` | 11-node flowchart, CI/CD alert panel, file tree, scaffold viewer |
| Alignment Bridge | `/alignment` | 5-node flowchart, alignment gauge, deviation table, risk summary, recommendation banner |

**Navigation flow:** Product (approve PRD) → `Next →` → Dev (generate architecture) → `Run Alignment Check →` → Alignment Bridge

---

### Technology Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python 3.13 + Flask |
| LLM Provider | Cerebras via LiteLLM (`cerebras/gpt-oss-120b`) |
| Streaming | Server-Sent Events (SSE) |
| Parallelism | `concurrent.futures.ThreadPoolExecutor` |
| Frontend | Vanilla HTML/CSS/JS (no frameworks) |
| State transfer | `localStorage` for cross-page data |
| Scaffold output | Files written to `sandbox_projects/` |

---

## Project Structure

```
Autonomous AI software company/
├── app.py                        # Flask server — routes, SSE, orchestration (243 lines)
├── product_alignment.py          # Product engine — 10 agents (932 lines)
├── dev_engine.py                 # Dev engine — 11 agents (871 lines)
├── alignment_bridge.py           # Alignment bridge — 5 agents (638 lines)
├── main.py                       # Legacy CLI pipeline (not used by web app)
├── README.md                     # This file
├── templates/
│   ├── index.html                # Product pipeline UI (1,704 lines)
│   ├── dev.html                  # Dev pipeline UI (981 lines)
│   └── alignment.html            # Alignment bridge UI (777 lines)
├── crewai-project/
│   └── venv/                     # Python virtual environment
│       └── Scripts/python.exe
│       └── Lib/site-packages/    # All dependencies installed here
└── sandbox_projects/             # Generated code scaffolds (created at runtime)
```

---

## How to Run Locally

### Prerequisites

- **Python 3.10+** (tested with Python 3.13)
- **pip** (Python package manager)
- Internet connection (for Cerebras API calls)

### Step 1 — Clone or extract the project

```bash
# If from zip:
unzip autonomous-ai-software-company.zip
cd "Autonomous AI software company"
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv
```

### Step 3 — Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### Step 4 — Install dependencies

```bash
pip install flask litellm
```

> **Note:** `litellm` handles the Cerebras API integration. No other dependencies are required.

### Step 5 — (Optional) Set your own API key

The project ships with a Cerebras API key. If you want to use your own, edit these files and replace the `CEREBRAS_API_KEY` value:
- `product_alignment.py` (line ~14)
- `dev_engine.py` (line ~14)
- `alignment_bridge.py` (line ~14)

```python
CEREBRAS_API_KEY = "your-cerebras-api-key-here"
```

### Step 6 — Fix the venv path (if needed)

If your virtual environment is NOT at `crewai-project/venv/`, edit the top of `app.py` and update the `VENV_SITE` path:

```python
VENV_SITE = os.path.join(os.path.dirname(__file__), "venv", "Lib", "site-packages")
```

Or simply remove the `sys.path.insert` line if your environment is already activated.

### Step 7 — Run the server

```bash
python app.py
```

You should see:
```
  ╔══════════════════════════════════════════════╗
  ║  Product Alignment System — Web UI          ║
  ║  Open: http://localhost:5000                 ║
  ╚══════════════════════════════════════════════╝

 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### Step 8 — Open the app

Open **http://localhost:5000** in your browser.

---

## Usage Workflow

1. **Product Pipeline** (`/`):
   - Type your product idea in the text box
   - Click "Run Pipeline" — watch 10 agents work in real-time
   - Review the PRD, provide feedback or approve
   - Click "Next →" to proceed

2. **Dev Pipeline** (`/dev`):
   - The approved PRD is loaded automatically
   - Click "Generate Architecture" — watch 11 agents build the blueprint
   - Review architecture, file tree, CI/CD alerts
   - Click "Run Alignment Check →" to proceed

3. **Alignment Bridge** (`/alignment`):
   - Both PRD and dev output are loaded automatically
   - Click "Run Alignment Check" — watch 5 agents analyze coherence
   - Review: alignment score, deviation table, risk assessment, recommendation

---

## API Endpoints

| Method | Endpoint | Body | Response |
|--------|---------|------|----------|
| `GET` | `/` | — | Product pipeline HTML |
| `GET` | `/dev` | — | Dev pipeline HTML |
| `GET` | `/alignment` | — | Alignment bridge HTML |
| `GET` | `/api/steps` | — | JSON array of product step metadata |
| `POST` | `/api/run` | `{"idea": "..."}` | SSE stream of pipeline events |
| `POST` | `/api/feedback` | `{"feedback": "...", "state": {...}}` | SSE stream of revision events |
| `POST` | `/api/dev/run` | `{"prd": "...", "project_name": "..."}` | SSE stream of dev pipeline events |
| `POST` | `/api/alignment/run` | `{"prd": "...", "dev_output": "..."}` | SSE stream of alignment events |

---

## Architecture Diagram

```
User (Browser)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    Flask Server (app.py)                  │
│                                                          │
│   GET /           → index.html (Product UI)              │
│   POST /api/run   → product_alignment.run_pipeline()     │
│                                                          │
│   GET /dev        → dev.html (Dev UI)                    │
│   POST /api/dev/run → dev_engine.run_dev_pipeline()      │
│                                                          │
│   GET /alignment  → alignment.html (Bridge UI)           │
│   POST /api/alignment/run → alignment_bridge.run_...()   │
│                                                          │
│   All streaming via Server-Sent Events (SSE)             │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │  Cerebras Cloud  │
                │  gpt-oss-120b    │
                │  via LiteLLM     │
                └──────────────────┘
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: flask` | Run `pip install flask litellm` in your activated venv |
| `Address already in use` | Kill other Python processes: `Get-Process python | Stop-Process -Force` |
| Port 5000 conflict | Edit `app.py` last line: `app.run(port=5001)` |
| `UnicodeDecodeError` | Ensure your terminal uses UTF-8: `$env:PYTHONIOENCODING="utf-8"` |
| Cerebras API errors | Check your API key or rate limits; the system auto-retries 3 times |
| `venv path not found` | Update `VENV_SITE` in `app.py` to match your actual venv location |

---

## License

This project is for educational and demonstration purposes.
