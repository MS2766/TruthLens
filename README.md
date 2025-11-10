# TruthLens

TruthLens is a Python-based tool for automated fake news detection and explanation. It combines content analysis, retrieval of supporting/contradicting evidence, reasoning via large language models (LLMs), and metadata/propagation features to flag news items for credibility, provide rationales, and help users or moderators evaluate content quickly.

## 🚀 Features  
- Classify news items as **credible** vs **non-credible (fake/misleading)**.  
- Leverage a retrieval pipeline to gather external evidence for each news item.  
- Generate human-readable **rationales/explanations** using LLM reasoning.  
- Incorporate metadata and propagation features (e.g., publisher credibility, social spread) alongside textual input.  
- Modular architecture: easily plug in new models, retrieval sources, or explanation components.

## 📁 Repository Structure  
/TruthLens
│ app.py # Main application entry point (e.g., web UI or CLI)
│ clustering.py # Module for clustering or section-segmentation of text (if applicable)
│ embeddings_store.py # Module to handle text embedding storage/retrieval
│ explain.py # Explanation-generation module using LLMs or other logic
│ inference.py # Inference module: takes input news, runs pipeline, returns classification + rationale
│ reasoning.py # Reasoning module: handles LLM or rule-based reasoning over retrieved evidence
│ retrieval.py # Retrieval pipeline: fetches relevant documents or data for evidence
│ utils.py # Utility functions and shared helpers
│ requirements.txt # Python dependencies
│ templates/ # HTML templates (if web UI)
│ static/ # Static assets (CSS, JS, images)
│ .gitignore

## 🛠️ Getting Started  

### Prerequisites  
- Python 3.8 or higher  
- Pip (or your preferred package manager)  
- Access to a retrieval/index system (e.g., news archive, fact-check data, search API)  
- (Optionally) Access to an LLM (API key) if using the explanation/reasoning module  

### Installation  
git clone https://github.com/MS2766/TruthLens.git

cd TruthLens

pip install -r requirements.txt

## Example usage

Input a news headline or full article text.

The retrieval module fetches supporting/contradicting evidence.

The inference module computes text embeddings + metadata features and classifies the item.

The explanation module (via LLM) generates a rationale: Why this item is flagged.

The reasoning module may highlight evidence links and produce a short summary.

The UI displays the result to the user with confidence score, rationale, and reference links.

## 🔍 Pipeline Overview

Preprocessing & embedding: Convert input text to embeddings (via transformer or word-embedding model).

Retrieval: Query an evidence source with key query terms extracted from the text.

Feature extraction: Gather metadata features (publisher credibility, propagation stats) and combine with content features.

Classification: Run model (e.g., fine-tuned transformer or hybrid classifier) to predict credibility.

Explanation & reasoning: Use an LLM prompt or reasoning module to generate human-readable rationale based on classification and retrieved evidence.

Output: Provide classification label, confidence, rationale text, and optionally links to evidence.

## 📊 Evaluation & Metrics
In internal experiments the pipeline was evaluated using standard classification metrics:

Accuracy

Precision

Recall

F1-score

AUC-ROC

For instance:

Model Type	Accuracy	Precision	Recall	F1-score

Content-only baseline	~92%	~90%	~88%	~89%

Retrieval-augmented LLM	~95%	~93%	~92%	~92.5%

Hybrid model (content + metadata + reasoning)	~97%	~95%	~94%	~94.5%

Note: these are illustrative numbers; actual results may vary based on dataset and domain.
