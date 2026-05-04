import os
import sys

# Activate the venv's packages so we can run from the project root
VENV_SITE = os.path.join(os.path.dirname(__file__), "crewai-project", "venv", "Lib", "site-packages")
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

from crewai import Agent, Task, Crew, Process, LLM

# ── Configuration ──────────────────────────────────────────────────────────────
os.environ.setdefault("CEREBRAS_API_KEY", os.environ.get("CEREBRAS_API_KEY", ""))
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY"] = "false"

# ── LLM (Llama on Cerebras via litellm) ───────────────────────────────────────
llm = LLM(
    model="cerebras/llama3.1-8b",
    api_key=os.environ.get("CEREBRAS_API_KEY", ""),
    temperature=0.7,
)

# ── Get the user prompt ───────────────────────────────────────────────────────
if len(sys.argv) > 1:
    user_prompt = " ".join(sys.argv[1:])
else:
    user_prompt = input("\n🚀 What do you want the AI crew to build?\n> ").strip()
    if not user_prompt:
        print("No prompt given – exiting.")
        sys.exit(0)

print(f"\n📋 Prompt: {user_prompt}\n")

# ── Agents ─────────────────────────────────────────────────────────────────────
product_manager = Agent(
    role="Senior Product Manager",
    goal="Translate the user's idea into a clear product requirements document (PRD).",
    backstory=(
        "You are a seasoned product manager with deep experience shipping "
        "software products. You turn vague ideas into structured, actionable PRDs."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

architect = Agent(
    role="Software Architect",
    goal="Design the technical architecture and choose the right tech stack.",
    backstory=(
        "You are an expert software architect who designs scalable, maintainable "
        "systems. You pick the best technologies for the job."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

developer = Agent(
    role="Senior Software Developer",
    goal="Write clean, production-ready code that implements the architecture.",
    backstory=(
        "You are a senior full-stack developer. You write well-structured, "
        "documented code and follow best practices."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ── Tasks (sequential pipeline) ───────────────────────────────────────────────
task_prd = Task(
    description=(
        f"The user wants to build: '{user_prompt}'\n\n"
        "Create a detailed Product Requirements Document that includes:\n"
        "1. Project overview & goals\n"
        "2. Target users\n"
        "3. Core features (prioritised)\n"
        "4. Success metrics"
    ),
    expected_output="A structured PRD in markdown format.",
    agent=product_manager,
)

task_architecture = Task(
    description=(
        "Based on the PRD, design the technical architecture:\n"
        "1. System components & how they interact\n"
        "2. Recommended tech stack with justifications\n"
        "3. Data models / database schema\n"
        "4. API endpoints or interfaces\n"
        "5. Deployment strategy"
    ),
    expected_output="A technical architecture document in markdown format.",
    agent=architect,
)

task_code = Task(
    description=(
        "Based on the architecture document, write the initial implementation code.\n"
        "Include:\n"
        "1. Project folder structure\n"
        "2. Key source files with full code\n"
        "3. Configuration files (package.json, requirements.txt, docker-compose, etc.)\n"
        "4. Brief setup instructions"
    ),
    expected_output="Complete starter codebase with setup instructions.",
    agent=developer,
)

# ── Crew ───────────────────────────────────────────────────────────────────────
crew = Crew(
    agents=[product_manager, architect, developer],
    tasks=[task_prd, task_architecture, task_code],
    process=Process.sequential,
    verbose=True,
)

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        result = crew.kickoff()
        print("\n" + "=" * 80)
        print("✅ FINAL OUTPUT")
        print("=" * 80)
        print(result)
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        sys.exit(1)
