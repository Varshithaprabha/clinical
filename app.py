"""Clinician Workspace - Flask server.

Run:  python app.py   ->  http://127.0.0.1:5000
"""
import traceback

from flask import Flask, Response, jsonify, render_template, request

from src import llm
from src.config import LLM_MODEL
from src.inference import assess, models
from src.report_pdf import build_pdf

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({"models_loaded": bool(models()), "llm_available": llm.available(), "llm_model": LLM_MODEL})


@app.post("/api/assess")
def api_assess():
    body = request.get_json(force=True) or {}
    try:
        return jsonify(assess(body.get("form", {}), overrides=body.get("overrides"),
                              use_llm=body.get("use_llm", True)))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.post("/api/report.pdf")
def api_report_pdf():
    """Render the assessment (as currently shown in the GUI) to a downloadable PDF."""
    body = request.get_json(force=True) or {}
    try:
        pdf = build_pdf(body.get("patient", {}), body.get("result", {}), body.get("rx"), body.get("prescriber"))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    pid = "".join(c for c in str((body.get("patient") or {}).get("patient_id") or "assessment") if c.isalnum() or c in "-_")
    return Response(pdf, mimetype="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{pid or "assessment"}_report.pdf"'})


if __name__ == "__main__":
    models()  # load once at start-up
    app.run(host="127.0.0.1", port=5000, debug=False)
