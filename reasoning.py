# reasoning.py
import os
import json
import re
import torch
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai

# -----------------------------
# Environment / Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# -----------------------------
# Models (lazy-loaded)
# -----------------------------
# Local reasoning model name (kept here so it can be configured)
reasoner_model = "google/flan-t5-large"
# Handles will be created on demand by _load_local_models()
tokenizer = None
model = None
embedder = None

def _load_local_models(model_name: str = None):
    """Lazy-load tokenizer, seq2seq model and embedder when needed.

    Calling this during import is avoided so tests and light-weight runs
    that rely on Gemini don't trigger heavy downloads.
    """
    global tokenizer, model, embedder
    mname = model_name or reasoner_model
    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(mname)
        model = AutoModelForSeq2SeqLM.from_pretrained(mname).to(device)
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Utility: Compute similarity confidence
# -----------------------------
def compute_confidence(claim, evidence):
    # ensure embedder is available
    global embedder
    if embedder is None:
        _load_local_models()
    claim_emb = embedder.encode(claim, convert_to_tensor=True)
    evidence_emb = embedder.encode(evidence, convert_to_tensor=True)
    confidence = float(util.cos_sim(claim_emb, evidence_emb).item())
    return round(confidence, 2)


# -----------------------------
# Reasoning function
# -----------------------------
def reason_about_claim(claim, evidence_snippets):
    evidence = "\n".join(evidence_snippets[:3])

    if GEMINI_KEY:
        try:
            genai.configure(api_key=GEMINI_KEY)

            system_prompt = (
                "You are a fact-checking assistant. "
                "Determine if the claim is TRUE, FALSE, or UNVERIFIABLE. "
                "Respond only in JSON with keys: verdict, confidence (0-1), explanation. "
                "Example: {\"verdict\": \"Likely True\", \"confidence\": 0.87, \"explanation\": \"...\"}"
            )
            user_prompt = f"Claim: {claim}\nEvidence:\n{evidence}\nReturn only JSON."

            model_gen = genai.GenerativeModel("gemini-2.0-flash")
            prompt = f"{system_prompt}\n\n{user_prompt}"

            response = model_gen.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "candidate_count": 1,
                }
            )

            raw_output = getattr(response, "text", "").strip() or str(response).strip()

            try:
                parsed = json.loads(raw_output)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                    except Exception:
                        parsed = None
                else:
                    parsed = None

            if not parsed or not isinstance(parsed, dict):
                print(f"[reasoning.py] Non-JSON or empty response from Gemini:\n{raw_output}")
                parsed = {
                    "verdict": "Unverifiable",
                    "confidence": compute_confidence(claim, evidence),
                    "explanation": raw_output[:300] or "Gemini returned no valid output."
                }

            if "confidence" not in parsed or parsed["confidence"] == 0:
                parsed["confidence"] = compute_confidence(claim, evidence)

            return parsed

        except Exception as e:
            print(f"[reasoning.py] Gemini API failed: {e}")

    prompt = f"""
You are a scientific fact-checking assistant.
Determine if the claim is TRUE, FALSE, or UNVERIFIABLE given the evidence.

Claim: {claim}

Evidence:
{evidence}

Output JSON with fields "verdict", "confidence", "explanation".
"""
    # ensure local tokenizer/model are loaded before using them
    _load_local_models()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True, 
        num_beams=2       # improves quality
    )
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    try:
        parsed = json.loads(raw_output)
    except:
        match = re.search(r'\b(TRUE|FALSE|UNVERIFIABLE)\b', raw_output, re.IGNORECASE)
        verdict = match.group(1).title() if match else "Unverifiable"
        parsed = {
            "verdict": verdict,
            "confidence": compute_confidence(claim, evidence),
            "explanation": raw_output.strip()
        }

    if "confidence" not in parsed or parsed["confidence"] == 0:
        parsed["confidence"] = compute_confidence(claim, evidence)

    return parsed


if __name__ == "__main__":
    claim = "Electric cars reduce carbon emissions compared to petrol cars"
    evidence = [
        "Electric cars emit fewer greenhouse gases and air pollutants over their lifetime.",
        "Petrol vehicles produce CO2 during combustion, contributing to climate change.",
        "Battery production has environmental impacts, but still lower lifetime emissions for EVs."
    ]
    result = reason_about_claim(claim, evidence)
    print(json.dumps(result, indent=2))
