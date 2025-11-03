import os
from serpapi import GoogleSearch

def multi_round_retrieve(query, num_rounds=2, top_k=8, api_key=None):
    if api_key is None:
        api_key = os.getenv("SERPAPI_API_KEY")

    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": top_k,
    }

    snippets = []
    seen = set()
    for _ in range(num_rounds):
        search = GoogleSearch(params)
        results = search.get_dict()
        if "organic_results" in results:
            for r in results["organic_results"]:
                snippet = r.get("snippet")
                link = r.get("link")
                if snippet and snippet.lower() not in seen:
                    seen.add(snippet.lower())
                    snippets.append({"snippet": snippet.strip(), "link": link or "#"})

    return snippets[:top_k]

# retrieval module - no top-level execution code
