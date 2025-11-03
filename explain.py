# explain.py
import os
from dotenv import load_dotenv
load_dotenv()

# Use Google's Generative AI (Gemini) when available, otherwise fall back to a local
# Flan-T5 generator via HuggingFace transformers.
import google.generativeai as genai
from transformers import pipeline

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
FALLBACK_MODEL = "google/flan-t5-large"

# helper removed: use response.text from current SDK

def explain_with_gemini(claim, supporting_snips, opposing_snips, max_tokens=200):
    if not GEMINI_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=GEMINI_KEY)
    system = "You are an assistant that writes clear concise explanations why a claim is true/false/unverifiable. Use the evidence provided."
    prompt = f"{system}\n\nClaim: {claim}\n\nSupporting snippets:\n"
    for s in supporting_snips[:3]:
        prompt += f"- {s.get('snippet')} (source: {s.get('link')})\n"
    prompt += "\nOpposing snippets:\n"
    for s in opposing_snips[:3]:
        prompt += f"- {s.get('snippet')} (source: {s.get('link')})\n"
    prompt += "\nWrite: 1-line verdict; then 2-3 sentence justification; then list top 3 provenance links."
    
    model_gen = genai.GenerativeModel("gemini-2.0-flash")
    response = model_gen.generate_content(
        prompt,
        generation_config={
            "temperature": 0.0,
            "candidate_count": 1,
            "max_output_tokens": max_tokens
        }
    )

    try:
        text = response.text
    except Exception as e:
        print(f"[explain.py] Failed to get response text: {e}")
        text = str(response)
    return text

# fallback local generator
_flan_pipe = None
def explain_with_flan(claim, supporting_snips, opposing_snips):
    global _flan_pipe
    if _flan_pipe is None:
        _flan_pipe = pipeline("text2text-generation", model=FALLBACK_MODEL, device=-1)
    prompt = "Claim: " + claim + "\nSupport:\n"
    for s in supporting_snips[:3]:
        prompt += "- " + (s.get("snippet") or s.get("title") or "") + "\n"
    prompt += "Oppose:\n"
    for s in opposing_snips[:3]:
        prompt += "- " + (s.get("snippet") or s.get("title") or "") + "\n"
    prompt += "\nWrite a 1-line verdict and a 2-sentence reasoning and list top 3 sources."
    out = _flan_pipe(prompt, max_length=256)[0]["generated_text"]
    return out

def generate_explanation(claim, support, oppose):
    sup_snips = [s for s,_ in support] if isinstance(support, list) and support and isinstance(support[0], tuple) else support
    opp_snips = [s for s,_ in oppose] if isinstance(oppose, list) and oppose and isinstance(oppose[0], tuple) else oppose
    if GEMINI_KEY:
        return explain_with_gemini(claim, sup_snips, opp_snips)
    else:
        return explain_with_flan(claim, sup_snips, opp_snips)
