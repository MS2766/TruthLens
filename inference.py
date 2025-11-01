# inference.py
from transformers import pipeline
import numpy as np

# create NLI pipeline (uses bart-large-mnli)
nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli", device=-1)

def entailment_score(claim, evidence):
    # We feed premise=evidence, hypothesis=claim per NLI convention
    try:
        res = nli_pipeline(f"{evidence} </s> {claim}")
    except Exception:
        # fallback inference style
        res = nli_pipeline(f"{evidence} {claim}")
    # mapping: BART returns 'CONTRADICTION','NEUTRAL','ENTAILMENT' with scores
    mapping = {r['label'].upper(): r['score'] for r in res}
    return mapping

def score_cluster_against_claim(cluster_snippets, claim):
    entail_scores = []
    contra_scores = []
    supporting = []
    opposing = []
    for s in cluster_snippets:
        text = s.get("snippet") or s.get("title") or ""
        out = entailment_score(claim, text)
        entail = out.get("ENTAILMENT", 0.0)
        contra = out.get("CONTRADICTION", 0.0)
        entail_scores.append(entail)
        contra_scores.append(contra)
        if entail > contra:
            supporting.append((s, entail))
        elif contra > entail:
            opposing.append((s, contra))
    avg_entail = float(np.mean(entail_scores)) if entail_scores else 0.0
    avg_contra = float(np.mean(contra_scores)) if contra_scores else 0.0
    return {"avg_entail": avg_entail, "avg_contra": avg_contra,
            "supporting": sorted(supporting, key=lambda x: -x[1]),
            "opposing": sorted(opposing, key=lambda x: -x[1])}

def defense_inference(clusters, claim):
    narrative_results = []
    for cl in clusters:
        r = score_cluster_against_claim(cl, claim)
        score = r["avg_entail"] - r["avg_contra"]
        narrative_results.append({"cluster": cl, "score": score, "details": r})
    total_score = sum(nr["score"] for nr in narrative_results)
    if total_score > 0.05:
        verdict = "Likely True"
    elif total_score < -0.05:
        verdict = "Likely False"
    else:
        verdict = "Unverifiable"
    confidence = min(0.99, abs(total_score))
    return {"verdict": verdict, "confidence": float(confidence), "narratives": narrative_results}
