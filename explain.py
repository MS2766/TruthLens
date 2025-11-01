# explain.py
import os
import openai
from transformers import pipeline

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
FALLBACK_MODEL = "google/flan-t5-large"

def explain_with_openai(claim, supporting_snips, opposing_snips, max_tokens=200):
    openai.api_key = OPENAI_KEY
    system = "You are an assistant that writes clear concise explanations why a claim is true/false/unverifiable. Use the evidence provided."
    prompt = f"Claim: {claim}\n\nSupporting snippets:\n"
    for s in supporting_snips[:3]:
        prompt += f"- {s.get('snippet')} (source: {s.get('link')})\n"
    prompt += "\nOpposing snippets:\n"
    for s in opposing_snips[:3]:
        prompt += f"- {s.get('snippet')} (source: {s.get('link')})\n"
    prompt += "\nWrite: 1-line verdict; then 2-3 sentence justification; then list top 3 provenance links."
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini" if True else "gpt-4",
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}],
        max_tokens=max_tokens,
        temperature=0.0
    )
    text = resp['choices'][0]['message']['content']
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
    if OPENAI_KEY:
        return explain_with_openai(claim, sup_snips, opp_snips)
    else:
        return explain_with_flan(claim, sup_snips, opp_snips)
