"""
Product Alignment System — Web UI  (Flask + SSE)
==================================================
Run:  python app.py
Open:  http://localhost:5000
"""

import os
import sys
import json
import queue
import threading

# ── Load .env file if present ──────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Venv packages ──────────────────────────────────────────────────────────────
VENV_SITE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "crewai-project", "venv", "Lib", "site-packages",
)
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

from flask import Flask, render_template, request, Response, jsonify

# Import the engines
from product_alignment import run_pipeline, run_feedback_revision, STEP_INFO, TOTAL_STEPS
from dev_engine import run_dev_pipeline, DEV_STEP_INFO, DEV_TOTAL_STEPS
from alignment_bridge import (
    run_alignment_bridge, ALIGNMENT_STEP_INFO, ALIGNMENT_TOTAL_STEPS,
    ALIGNMENT_PASS_THRESHOLD,
)

app = Flask(__name__)

# Store pipeline state for feedback rounds (protected by lock)
_state_lock = threading.Lock()
_pipeline_state = None
_dev_pipeline_state = None
_alignment_state = None

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", steps=STEP_INFO, total=TOTAL_STEPS)


@app.route("/api/steps")
def api_steps():
    return jsonify(STEP_INFO)


@app.route("/api/run", methods=["POST"])
def api_run():
    """
    Start the pipeline and stream progress via Server-Sent Events.
    The client POSTs { "idea": "..." } and receives an SSE stream.
    """
    data = request.get_json(silent=True) or {}
    raw_idea = (data.get("idea") or "").strip()
    if not raw_idea:
        return jsonify({"error": "No idea provided"}), 400

    # Use a queue to bridge the pipeline thread → SSE response
    event_queue = queue.Queue()

    def on_event(event_type, step_num, step_name, data_str):
        event_queue.put({
            "event": event_type,
            "step": step_num,
            "name": step_name,
            "data": data_str,
        })

    def pipeline_worker():
        global _pipeline_state
        try:
            result = run_pipeline(raw_idea, on_event=on_event)
            with _state_lock:
                _pipeline_state = result.get("pipeline_state")
            event_queue.put({
                "event": "complete",
                "step": 0,
                "name": "Done",
                "data": json.dumps({
                    "final_output": result["final_output"],
                    "step_outputs": result["step_outputs"],
                }),
            })
        except Exception as e:
            event_queue.put({
                "event": "fatal",
                "step": 0,
                "name": "Error",
                "data": f"{type(e).__name__}: {e}",
            })

    # Start the pipeline in a background thread
    t = threading.Thread(target=pipeline_worker, daemon=True)
    t.start()

    def generate():
        while True:
            try:
                msg = event_queue.get(timeout=600)  # 10 min max
            except queue.Empty:
                yield "data: {\"event\":\"timeout\"}\n\n"
                break

            yield f"data: {json.dumps(msg)}\n\n"

            if msg["event"] in ("complete", "fatal", "timeout"):
                break

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    """
    Accept user feedback on the PRD and run a revision cycle.
    Requires a previous pipeline run (pipeline state must exist).
    """
    global _pipeline_state
    with _state_lock:
        if not _pipeline_state:
            return jsonify({"error": "No pipeline has been run yet. Run the pipeline first."}), 400
        current_state = _pipeline_state

    data = request.get_json(silent=True) or {}
    feedback = (data.get("feedback") or "").strip()
    if not feedback:
        return jsonify({"error": "No feedback provided"}), 400

    event_queue = queue.Queue()

    def on_event(event_type, step_num, step_name, data_str):
        event_queue.put({
            "event": event_type,
            "step": step_num,
            "name": step_name,
            "data": data_str,
        })

    def feedback_worker():
        try:
            result = run_feedback_revision(feedback, current_state, on_event=on_event)
            # Strip bulky raw feedback text from history to keep SSE payload small
            slim_history = []
            for h in (result.get("qg_history") or []):
                entry = dict(h)
                entry["feedback"] = entry.get("feedback", "")[:600]  # truncate raw QG text
                slim_history.append(entry)
            event_queue.put({
                "event": "feedback_complete",
                "step": 0,
                "name": "Feedback Revision Done",
                "data": json.dumps({
                    "revised_prd": result["revised_prd"],
                    "score": result["score"],
                    "verdict": result["verdict"],
                    "qg_history": slim_history,
                    "qg_feedback": (result.get("qg_result") or "")[:800],
                }),
            })
        except Exception as e:
            event_queue.put({
                "event": "fatal",
                "step": 0,
                "name": "Error",
                "data": f"{type(e).__name__}: {e}",
            })

    t = threading.Thread(target=feedback_worker, daemon=True)
    t.start()

    def generate():
        while True:
            try:
                msg = event_queue.get(timeout=300)
            except queue.Empty:
                yield "data: {\"event\":\"timeout\"}\n\n"
                break
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["event"] in ("feedback_complete", "fatal", "timeout"):
                break

    return Response(generate(), mimetype="text/event-stream")


# ── Dev Pipeline Routes ────────────────────────────────────────────────────────

@app.route("/dev")
def dev_page():
    return render_template("dev.html", steps=DEV_STEP_INFO, total=DEV_TOTAL_STEPS)


@app.route("/api/dev/run", methods=["POST"])
def api_dev_run():
    """Start the dev pipeline with an approved PRD. Streams SSE."""
    data = request.get_json(silent=True) or {}
    prd_text = (data.get("prd") or "").strip()
    project_name = (data.get("project_name") or "project").strip()
    if not prd_text:
        return jsonify({"error": "No PRD provided"}), 400

    event_queue = queue.Queue()

    def on_event(event_type, step_num, step_name, data_str):
        event_queue.put({
            "event": event_type,
            "step": step_num,
            "name": step_name,
            "data": data_str,
        })

    def dev_worker():
        global _dev_pipeline_state
        try:
            result = run_dev_pipeline(prd_text, project_name=project_name, on_event=on_event)
            with _state_lock:
                _dev_pipeline_state = result.get("pipeline_state")
        except Exception as e:
            event_queue.put({
                "event": "dev_fatal",
                "step": 0,
                "name": "Error",
                "data": f"{type(e).__name__}: {e}",
            })

    t = threading.Thread(target=dev_worker, daemon=True)
    t.start()

    def generate():
        while True:
            try:
                msg = event_queue.get(timeout=600)
            except queue.Empty:
                yield 'data: {"event":"timeout"}\n\n'
                break
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["event"] in ("dev_complete", "dev_fatal", "timeout"):
                break

    return Response(generate(), mimetype="text/event-stream")


# ── Alignment Bridge Routes ─────────────────────────────────────────────────────

@app.route("/alignment")
def alignment_page():
    return render_template("alignment.html",
                           steps=ALIGNMENT_STEP_INFO,
                           total=ALIGNMENT_TOTAL_STEPS,
                           threshold=ALIGNMENT_PASS_THRESHOLD)


@app.route("/api/alignment/run", methods=["POST"])
def api_alignment_run():
    """Run the alignment bridge with PRD + Dev output. Streams SSE."""
    global _alignment_state
    data = request.get_json(silent=True) or {}
    prd_text = (data.get("prd") or "").strip()
    dev_output = (data.get("dev_output") or "").strip()
    if not prd_text or not dev_output:
        return jsonify({"error": "Both PRD and dev output are required"}), 400

    # Use dev pipeline state for deeper analysis if available
    with _state_lock:
        dev_state = _dev_pipeline_state

    event_queue = queue.Queue()

    def on_event(event_type, step_num, step_name, data_str):
        event_queue.put({
            "event": event_type,
            "step": step_num,
            "name": step_name,
            "data": data_str,
        })

    def alignment_worker():
        global _alignment_state
        try:
            result = run_alignment_bridge(
                prd_text, dev_output,
                dev_state=dev_state,
                on_event=on_event,
            )
            _alignment_state = result
            # no lock needed: alignment_state only read after completion
        except Exception as e:
            event_queue.put({
                "event": "align_fatal",
                "step": 0,
                "name": "Error",
                "data": f"{type(e).__name__}: {e}",
            })

    t = threading.Thread(target=alignment_worker, daemon=True)
    t.start()

    def generate():
        while True:
            try:
                msg = event_queue.get(timeout=600)
            except queue.Empty:
                yield 'data: {"event":"timeout"}\n\n'
                break
            yield f"data: {json.dumps(msg)}\n\n"
            if msg["event"] in ("align_complete", "align_fatal", "timeout"):
                break

    return Response(generate(), mimetype="text/event-stream")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  ╔══════════════════════════════════════════════╗")
    print("  ║  Product Alignment System — Web UI          ║")
    print("  ║  Open: http://localhost:5000                 ║")
    print("  ╚══════════════════════════════════════════════╝\n")
    app.run(debug=False, port=5000, threaded=True)
