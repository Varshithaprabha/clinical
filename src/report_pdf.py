"""PDF export of the clinical assessment report (reportlab).

build_pdf(patient, result, rx, prescriber) -> bytes
  patient    : dict from the screening form (patient_id, patient_name, age, sex, ...)
  result     : the dict returned by inference.assess()
  rx         : the prescription as the clinician left it in the GUI (items may have been removed)
  prescriber : {"name": str, "approved": bool}
"""
import io
import os
import re
from datetime import date
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# ---------- fonts: a Unicode TTF so ≥, ², ℞, μ, en-dashes render; Helvetica as last resort
_FONT, _BOLD = "Helvetica", "Helvetica-Bold"


def _register_fonts():
    global _FONT, _BOLD
    windir = os.environ.get("WINDIR", r"C:\Windows")
    candidates = [
        (os.path.join(windir, "Fonts", "segoeui.ttf"), os.path.join(windir, "Fonts", "segoeuib.ttf")),
        (os.path.join(windir, "Fonts", "arial.ttf"), os.path.join(windir, "Fonts", "arialbd.ttf")),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for reg, bold in candidates:
        if os.path.exists(reg) and os.path.exists(bold):
            try:
                pdfmetrics.registerFont(TTFont("Body", reg))
                pdfmetrics.registerFont(TTFont("Body-Bold", bold))
                pdfmetrics.registerFontFamily("Body", normal="Body", bold="Body-Bold",
                                              italic="Body", boldItalic="Body-Bold")
                _FONT, _BOLD = "Body", "Body-Bold"
                return
            except Exception:  # noqa: BLE001
                continue


_register_fonts()

# characters that are missing in common fonts -> safe equivalents
_REPL = {"\u2011": "-", "\u2010": "-", "\u202f": " ", "\u00a0": " ", "\u2009": " "}


def _clean(s) -> str:
    s = "" if s is None else str(s)
    for a, b in _REPL.items():
        s = s.replace(a, b)
    if _FONT == "Helvetica":  # Latin-1 only
        s = (s.replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("—", "-")
              .replace("℞", "Rx").replace("²", "2").replace("→", "->"))
        s = s.encode("latin-1", "replace").decode("latin-1")
    return s


def _n(x):
    """58.0 -> 58, 7.25 -> 7.25; non-numbers unchanged."""
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return "-" if x is None else str(x)


def _p(text, style):
    return Paragraph(escape(_clean(text)).replace("\n", "<br/>"), style)


# ---------- styles
INK, MUTED, LINE, ACCENT = colors.HexColor("#111827"), colors.HexColor("#6b7280"), colors.HexColor("#e5e7eb"), colors.HexColor("#1f2937")
LEVEL_COLOR = {"high": "#b91c1c", "known": "#7c3aed", "moderate": "#b45309", "low": "#15803d", "insufficient": "#6b7280"}
LEVEL_LABEL = {"high": "High", "known": "Known condition", "moderate": "Moderate", "low": "Low", "insufficient": "Insufficient data"}

_ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=_ss["Title"], fontName=_BOLD, fontSize=18, leading=22, alignment=TA_LEFT,
                            textColor=INK, spaceAfter=2),
    "sub": ParagraphStyle("s", fontName=_FONT, fontSize=9, leading=12, textColor=MUTED),
    "h2": ParagraphStyle("h2", fontName=_BOLD, fontSize=12, leading=15, textColor=INK, spaceBefore=10, spaceAfter=4,
                         keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName=_BOLD, fontSize=10, leading=13, textColor=INK, spaceBefore=6, spaceAfter=1,
                         keepWithNext=1),
    "body": ParagraphStyle("b", fontName=_FONT, fontSize=9.5, leading=13.5, textColor=INK),
    "cell": ParagraphStyle("c", fontName=_FONT, fontSize=8.8, leading=11.5, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName=_BOLD, fontSize=8.8, leading=11.5, textColor=INK),
    "small": ParagraphStyle("sm", fontName=_FONT, fontSize=8, leading=10.5, textColor=MUTED),
    "warn": ParagraphStyle("w", fontName=_FONT, fontSize=8.8, leading=11.5, textColor=colors.HexColor("#991b1b")),
    "rx": ParagraphStyle("rx", fontName=_BOLD, fontSize=22, leading=26, textColor=INK),
}

GRID = TableStyle([
    ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
])


def _table(rows, widths, header=True):
    data = [[c if isinstance(c, Paragraph) else _p(c, S["cellb"] if (header and i == 0) else S["cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    t.setStyle(GRID if header else TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                                ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return t


def _level(level):
    c = LEVEL_COLOR.get(level, "#6b7280")
    return Paragraph(f'<font color="{c}"><b>{escape(LEVEL_LABEL.get(level, level or "-"))}</b></font>', S["cell"])


# ---------- report text: strip the appended Rx block (rendered as a table instead), bold section headings
_HEAD = re.compile(r"^\s*([A-Z][A-Z0-9 &/,()\-\u2010\u2011\u2013]{3,60}):\s*(.*)$")


_KEEP_UPPER = {"BMI", "ADA", "HTN", "BP", "LLM", "NLP", "SHAP", "LIME", "CKD", "ECG", "HBA1C"}


def _title(h):
    return " ".join(w if w.upper() in _KEEP_UPPER else w.capitalize() if i == 0 else w.lower()
                    for i, w in enumerate(h.split()))


def _report_flow(text):
    text = _clean(text or "")
    cut = text.find("PRESCRIPTION (DRAFT")
    if cut != -1:
        text = text[:cut]
    out = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        m = _HEAD.match(line)
        if m:
            out.append(Paragraph(escape(_title(m.group(1))), S["h3"]))
            if m.group(2).strip():
                out.append(_p(m.group(2).strip(), S["body"]))
        elif re.match(r"^\s*[-•*]\s+", line):
            out.append(Paragraph("• " + escape(re.sub(r"^\s*[-•*]\s+", "", line)), ParagraphStyle(
                "li", parent=S["body"], leftIndent=10, firstLineIndent=-7)))
        else:
            out.append(_p(line.strip(), S["body"]))
    return out


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(_FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, _clean("AI-assisted decision support – to be verified by the treating clinician."))
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(patient: dict, result: dict, rx: dict | None = None, prescriber: dict | None = None) -> bytes:
    patient, result = patient or {}, result or {}
    rx = rx or result.get("prescription") or {}
    prescriber = prescriber or {}
    val = result.get("validation", {}) or {}
    v = val.get("values", {}) or {}
    bmi, dm, htn = result.get("bmi", {}) or {}, result.get("diabetes", {}) or {}, result.get("hypertension", {}) or {}
    W = A4[0] - 36 * mm
    story = []

    # ---- header
    story += [Paragraph("Clinical Assessment Report", S["title"]),
              _p(f"Diabetes & hypertension screening · generated {date.today().isoformat()}", S["sub"]),
              Spacer(1, 6)]
    sex = val.get("sex") or patient.get("sex") or "Not recorded"
    pid = [["Patient name", patient.get("patient_name") or "-", "Patient ID", patient.get("patient_id") or "-"],
           ["Age", f"{_n(v.get('age') or patient.get('age'))} years", "Sex", str(sex).capitalize()],
           ["Screening date", patient.get("screening_date") or date.today().isoformat(), "Follow-up", patient.get("follow_up") or "-"]]
    t = _table([[_p(a, S["small"]), _p(b, S["cellb"]), _p(c, S["small"]), _p(d, S["cellb"])] for a, b, c, d in pid],
               [W * .17, W * .33, W * .17, W * .33], header=False)
    t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, LINE), ("LEFTPADDING", (0, 0), (-1, -1), 6),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    story += [t, Spacer(1, 4)]

    # ---- measurements
    story.append(Paragraph("Measurements", S["h2"]))
    bp = (f"{_n(v.get('systolic_bp'))}/{_n(v.get('diastolic_bp'))} mmHg" if v.get("systolic_bp") else "-")
    meas = [["Weight", f"{_n(v['weight'])} kg" if v.get("weight") else "-",
             "Height", f"{_n(v['height'])} cm" if v.get("height") else "-"],
            ["BMI", f"{bmi['value']} kg/m² ({bmi.get('who', '')}; Asian: {bmi.get('asian', '')})" if bmi.get("value") else "Not available",
             "Blood pressure", f"{bp}  {htn.get('bp_category') or ''}".strip()],
            ["GRBS", f"{_n(v['grbs'])} mg/dL" if v.get("grbs") else "-",
             "FBS / HbA1c / eGFR", " / ".join(_n(v.get(k)) for k in ("fbs", "hba1c", "egfr"))]]
    story.append(_table([[_p(a, S["small"]), _p(b, S["cell"]), _p(c, S["small"]), _p(d, S["cell"])] for a, b, c, d in meas],
                        [W * .14, W * .36, W * .17, W * .33], header=False))

    # ---- summary
    story.append(Paragraph("Assessment summary", S["h2"]))
    dm_detail = dm.get("status", "")
    if dm.get("level") != "insufficient" and dm.get("hybrid_score") is not None:
        dm_detail += f"\nHybrid XGBoost {dm['hybrid_score']} (threshold {dm.get('hybrid_threshold')}); {dm.get('models_flagging', '')} models flag elevated risk."
    if dm.get("criteria_met"):
        dm_detail += "\nCriteria: " + "; ".join(dm["criteria_met"])
    htn_detail = htn.get("status", "")
    if htn.get("model_score") is not None:
        htn_detail += f"\nHypertension XGBoost {htn['model_score']} (threshold {htn.get('threshold')})."
    story.append(_table([["Condition", "Level", "Finding"],
                         ["Diabetes", _level(dm.get("level")), dm_detail],
                         ["Hypertension", _level(htn.get("level")), htn_detail]],
                        [W * .16, W * .18, W * .66]))

    # ---- model scores
    if dm.get("structured_models"):
        rows = [["Model", "Probability", "At tuned threshold"]]
        for k, x in dm["structured_models"].items():
            rows.append([k, x.get("probability"), "Elevated" if x.get("flag") else "Not elevated"])
        for k, x in (dm.get("text_models") or {}).items():
            if isinstance(x, (int, float)):
                rows.append([f"{k} (text only)", x, "text signal"])
        rows.append([Paragraph("<b>Hybrid XGBoost (primary)</b>", S["cell"]), dm.get("hybrid_score"),
                     "Elevated" if (dm.get("hybrid_score") or 0) >= (dm.get("hybrid_threshold") or 1) else "Not elevated"])
        story += [Paragraph("Diabetes model scores", S["h2"]), _table(rows, [W * .5, W * .2, W * .3])]

    # ---- explainability
    shap = (result.get("explainability") or {}).get("shap")
    if shap:
        rows = [["Feature (value)", "SHAP contribution", "Effect"]]
        for s in shap[:8]:
            c = s.get("contribution", 0)
            rows.append([f"{s.get('feature')} ({_n(s.get('value'))})", f"{c:+.3f}", "raises risk" if c > 0 else "lowers risk"])
        story += [Paragraph("Key contributing factors (SHAP)", S["h2"]), _table(rows, [W * .56, W * .2, W * .24])]

    # ---- warnings / data-quality issues
    issues = [i for i in val.get("issues", []) if i.get("level") in ("error", "missing")]
    if issues:
        story.append(Paragraph("Data quality notes", S["h2"]))
        story += [_p(f"• {i.get('field')}: {i.get('message')}", S["body"]) for i in issues]

    # ---- clinician report (LLM / template)
    rep = result.get("report") or {}
    story.append(Paragraph("Clinician report", S["h2"]))
    story.append(_p(f"Source: {rep.get('source', '-')}", S["small"]))
    story += _report_flow(rep.get("text", ""))

    # ---- prescription
    approved = bool(prescriber.get("approved") and (prescriber.get("name") or "").strip())
    rx_block = [Paragraph("Prescription", S["h2"]),
                _p("Reviewed and approved by the prescriber." if approved else
                   "DRAFT – not valid until reviewed and signed by the treating clinician.",
                   S["small"] if approved else S["warn"]),
                Paragraph("R<font size=14>x</font>", S["rx"])]
    items = rx.get("items") or []
    rows = [["#", "Medicine", "Dose", "Frequency", "When", "Duration"]]
    for i, it in enumerate(items, 1):
        if it.get("action") == "continue":
            rows.append([str(i), f"{it.get('drug')} – continue as before", "", "", "", ""])
        else:
            rows.append([str(i), Paragraph(f"<b>Tab {escape(_clean(it.get('drug')))}</b><br/><font size=7.5 color='#6b7280'>"
                                           f"{escape(_clean(it.get('indication') or '').replace('_', ' '))}</font>", S["cell"]),
                         it.get("dose") or "", f"{it.get('pattern') or ''}\n{it.get('frequency_text') or ''}",
                         it.get("timing") or "", it.get("duration") or ""])
    if not items:
        rows.append(["", "No medication prescribed at this visit.", "", "", "", ""])
    rx_block.append(_table(rows, [W * .05, W * .33, W * .12, W * .2, W * .15, W * .15]))
    story.append(KeepTogether(rx_block))
    for w in rx.get("warnings") or []:
        story.append(_p(f"Warning: {w}", S["warn"]))
    for w in rx.get("removed_by_safety_check") or []:
        story.append(_p(f"Removed by safety check – {w}", S["small"]))
    for title, key in (("Advice", "non_drug_advice"), ("Investigations", "investigations")):
        if rx.get(key):
            story.append(Paragraph(title, S["h3"]))
            story += [_p(f"• {x}", S["body"]) for x in rx[key]]
    if rx.get("review_after"):
        story.append(_p(f"Review after: {rx['review_after']}", S["body"]))

    sig = [Spacer(1, 22), _p("Signature: ______________________________", S["body"]),
           _p(prescriber.get("name") or "Prescriber name and registration number: ____________________", S["body"])]
    story.append(KeepTogether(sig))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title=f"Clinical assessment {patient.get('patient_id') or ''}".strip(),
                            author="Clinician Workspace")
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()
