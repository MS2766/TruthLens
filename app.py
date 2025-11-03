from flask import Flask, request, render_template, jsonify
from retrieval import multi_round_retrieve
from reasoning import reason_about_claim
from utils import clean_text
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
        return render_template("index.html")

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
    top_snips = snippets[:6]
    return render_template("result.html", claim=claim, rounds=rounds, res=res, snippets=top_snips)

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
