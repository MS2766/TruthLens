from flask import Flask, request, render_template_string, jsonify
from retrieval import multi_round_retrieve
from reasoning import reason_about_claim
from utils import clean_text
import os

app = Flask(__name__)

INDEX_HTML = """
<!doctype html>
<title>NarrativeGuard - Demo</title>
<h1>NarrativeGuard — Explainable Claim Verifier</h1>
<form action="/verify" method="post">
  <textarea name="claim" rows="6" cols="80" placeholder="Enter claim text..."></textarea><br>
  Retrieval rounds: <input name="rounds" type="number" value="2" min="1" max="4"><br>
  <input type="submit" value="Verify Claim">
</form>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/verify", methods=["POST"])
def verify():
    claim = request.form.get("claim", "").strip()
    rounds = int(request.form.get("rounds", 2))
    if not claim:
        return "Please provide claim text.", 400

    key = os.getenv("SERPAPI_API_KEY")
    snippets = multi_round_retrieve(query=claim, num_rounds=rounds, top_k=8, api_key=key)

    evidence_texts = [s.get("snippet") for s in snippets if s.get("snippet")]

    res = reason_about_claim(claim, evidence_texts)

    html = f"<h2>Verdict: {res['verdict']} (conf {res['confidence']:.2f})</h2>"
    html += f"<h3>Explanation</h3><p>{res['explanation']}</p>"
    html += "<h4>Top snippets</h4><ul>"
    for s in snippets[:6]:
        snippet_text = clean_text(s.get("snippet"))
        link = s.get("link", "#")
        html += f"<li>{snippet_text} — <a href='{link}' target='_blank'>{link}</a></li>"
    html += "</ul><a href='/'>Back</a>"

    return html

@app.route("/api/verify", methods=["POST"])
def api_verify():
    data = request.json or {}
    claim = data.get("claim", "").strip()
    rounds = int(data.get("rounds", 2))
    if not claim:
        return jsonify({"error": "claim required"}), 400

    key = os.getenv("SERPAPI_API_KEY")
    snippets = multi_round_retrieve(query=claim, num_rounds=rounds, top_k=8, api_key=key)
    evidence_texts = [s.get("snippet") for s in snippets if s.get("snippet")]

    res = reason_about_claim(claim, evidence_texts)

    return jsonify({
        "verdict": res["verdict"],
        "confidence": res["confidence"],
        "explanation": res["explanation"],
        "top_snippets": snippets[:6]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
