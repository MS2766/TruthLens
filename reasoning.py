# reasoning.py
import os
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import openai

# -----------------------------
# Environment / Device
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[reasoning.py] Device set to {device}")

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# Models
# -----------------------------
# Local reasoning model
reasoner_model = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(reasoner_model)
model = AutoModelForSeq2SeqLM.from_pretrained(reasoner_model).to(device)

# Embedding model for fallback confidence
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# -----------------------------
# Utility: Compute similarity confidence
# -----------------------------
def compute_confidence(claim, evidence):
    claim_emb = embedder.encode(claim, convert_to_tensor=True)
    evidence_emb = embedder.encode(evidence, convert_to_tensor=True)
    confidence = float(util.cos_sim(claim_emb, evidence_emb).item())
    return round(confidence, 2)


# -----------------------------
# Reasoning function
# -----------------------------
def reason_about_claim(claim, evidence_snippets):
    """
    Returns structured JSON:
    {
      "verdict": "Likely True / Likely False / Unverifiable",
      "confidence": 0.xx,
      "explanation": "..."
    }
    """
    evidence = "\n".join(evidence_snippets[:3])  # top 3 snippets only

    # Prefer OpenAI GPT if key exists
    if OPENAI_KEY:
        openai.api_key = OPENAI_KEY
        system_prompt = (
            "You are a fact-checking assistant. "
            "Determine if the claim is TRUE, FALSE, or UNVERIFIABLE. "
            "Provide JSON output with fields: verdict, confidence (0-1), explanation."
        )
        user_prompt = f"Claim: {claim}\nEvidence:\n{evidence}\nReturn only JSON."
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0
            )
            raw_output = resp["choices"][0]["message"]["content"]
            parsed = json.loads(raw_output)
            if "confidence" not in parsed:
                parsed["confidence"] = compute_confidence(claim, evidence)
            return parsed
        except Exception as e:
            print(f"[reasoning.py] OpenAI API failed: {e}")
            raw_output = ""

    # -----------------------------
    # Local Flan-T5 fallback
    # -----------------------------
    prompt = f"""
You are a scientific fact-checking assistant.
Determine if the claim is TRUE, FALSE, or UNVERIFIABLE given the evidence.

Claim: {claim}

Evidence:
{evidence}

Output JSON with fields "verdict", "confidence", "explanation".
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,   # enable some randomness
        num_beams=2       # improves quality
    )
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # -----------------------------
    # Parse output
    # -----------------------------
    try:
        parsed = json.loads(raw_output)
    except:
        # Regex fallback: extract TRUE / FALSE / UNVERIFIABLE
        match = re.search(r'\b(TRUE|FALSE|UNVERIFIABLE)\b', raw_output, re.IGNORECASE)
        verdict = match.group(1).title() if match else "Unverifiable"
        parsed = {
            "verdict": verdict,
            "confidence": compute_confidence(claim, evidence),
            "explanation": raw_output.strip()
        }

    # Ensure confidence exists
    if "confidence" not in parsed or parsed["confidence"] == 0:
        parsed["confidence"] = compute_confidence(claim, evidence)

    return parsed


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    claim = "Electric cars reduce carbon emissions compared to petrol cars"
    evidence = [
        "Electric cars emit fewer greenhouse gases and air pollutants over their lifetime.",
        "Petrol vehicles produce CO2 during combustion, contributing to climate change.",
        "Battery production has environmental impacts, but still lower lifetime emissions for EVs."
    ]
    result = reason_about_claim(claim, evidence)
    print(json.dumps(result, indent=2))
