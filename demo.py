#!/usr/bin/env python3
"""
CAWC v5 - Unsupervised Hallucination Detection
Demo & Evaluation Script

Supports two modes:
1. CLI Evaluation Mode:
   python demo.py --query "..." --context "..." --response "..."

2. Web UI Presentation Mode:
   python demo.py
   (Starts a local server with a clean, light-theme academic frontend)
"""

import argparse
import json
import os
import sys
import time
import pickle
import numpy as np
from pathlib import Path
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# ------------------------------------------------------------------------
# CORE IMPORTS
# ------------------------------------------------------------------------
from demo_pipeline import (
    TRAIN_CACHE, METRIC_KEYS,
    compute_train_stats, compute_variance_weights, compute_composite_live
)
from model_utils import ModelConfig, load_generator_model, dual_forward_pass
from core_metrics import MetricEvaluator

# ------------------------------------------------------------------------
# FRONTEND HTML TEMPLATE
# ------------------------------------------------------------------------
HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CAWC v5 | Unsupervised Hallucination Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #f8f9fa;
            --surface: #ffffff;
            --border: #e2e8f0;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --danger: #ef4444;
            --success: #10b981;
            --warning: #f59e0b;
        }
        * { box-sizing: border-box; }
        body {
            font-family: 'Inter', system-ui, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            line-height: 1.5;
        }
        .header {
            background: var(--surface);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { margin: 0; font-size: 1.5rem; font-weight: 600; color: #0f172a; }
        .header p { margin: 0; color: var(--text-muted); font-size: 0.9rem; }
        
        .container {
            max-width: 1280px;
            margin: 2rem auto;
            padding: 0 1rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }
        
        .card {
            background: var(--surface);
            border-radius: 8px;
            padding: 1.5rem;
            border: 1px solid var(--border);
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .card h2 { margin-top: 0; font-size: 1.25rem; border-bottom: 1px solid var(--border); padding-bottom: 0.75rem; margin-bottom: 1rem; }
        
        .form-group { margin-bottom: 1rem; }
        label { display: block; font-weight: 500; margin-bottom: 0.5rem; color: #334155; }
        textarea {
            width: 100%;
            padding: 0.75rem;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-family: inherit;
            font-size: 0.95rem;
            resize: vertical;
            transition: border-color 0.2s;
        }
        textarea:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(37,99,235,0.1); }
        
        .btn {
            background: var(--primary);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 6px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            width: 100%;
            transition: background 0.2s;
        }
        .btn:hover { background: var(--primary-hover); }
        .btn:disabled { background: #94a3b8; cursor: not-allowed; }
        
        .loader {
            display: none;
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid var(--primary);
            border-radius: 50%;
            width: 24px;
            height: 24px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        
        .results-area { display: none; }
        
        .verdict {
            padding: 1rem;
            border-radius: 6px;
            text-align: center;
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        .verdict.high { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
        .verdict.moderate { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
        .verdict.low { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-box {
            background: #f8fafc;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var(--border);
            text-align: center;
        }
        .metric-box .label { font-size: 0.85rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }
        .metric-box .value { font-size: 1.5rem; font-weight: 700; color: var(--text-main); }
        
        .token-display {
            line-height: 2;
            padding: 1rem;
            background: #fff;
            border: 1px solid var(--border);
            border-radius: 6px;
            font-size: 1.05rem;
        }
        .token {
            display: inline-block;
            padding: 0 2px;
            border-radius: 3px;
            cursor: pointer;
            position: relative;
            transition: transform 0.1s;
        }
        .token:hover { transform: translateY(-1px); z-index: 10; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        
        .tooltip {
            visibility: hidden;
            background: #1e293b;
            color: #fff;
            text-align: left;
            padding: 0.75rem;
            border-radius: 6px;
            position: absolute;
            z-index: 100;
            bottom: 120%;
            left: 50%;
            transform: translateX(-50%);
            width: max-content;
            font-size: 0.85rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            opacity: 0;
            transition: opacity 0.2s;
            pointer-events: none;
            line-height: 1.4;
        }
        .token:hover .tooltip { visibility: visible; opacity: 1; }
        .tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #1e293b transparent transparent transparent;
        }
        .tt-row { display: flex; justify-content: space-between; gap: 1rem; }
        .tt-label { color: #94a3b8; }
        .tt-val { font-weight: 600; font-family: monospace; }
        
        .explanation {
            margin-top: 2rem;
            font-size: 0.9rem;
            color: #475569;
            background: #f8fafc;
            padding: 1rem;
            border-radius: 6px;
            border: 1px solid var(--border);
        }
        .explanation h3 { margin-top: 0; font-size: 1rem; color: #1e293b; }
        .explanation ul { padding-left: 1.5rem; margin-bottom: 0; }
        .explanation li { margin-bottom: 0.5rem; }
        
        /* Layout adjustments for small screens */
        @media (max-width: 900px) {
            .container { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

<div class="header">
    <div>
        <h1>CAWC v5 Pipeline</h1>
        <p>Unsupervised Token-Level Hallucination Detection (Track A)</p>
    </div>
    <div style="text-align: right;">
        <p><strong>Proxy Model:</strong> Qwen2.5-7B (4-bit)</p>
        <p><strong>Method:</strong> Dual Forward Pass (Context vs No-Context)</p>
    </div>
</div>

<div class="container">
    <!-- Input Section -->
    <div class="card">
        <h2>Input Data</h2>
        <div class="form-group">
            <label for="query">User Query / Question</label>
            <textarea id="query" rows="3" placeholder="Enter the user's question here...">What are the symptoms of COVID-19?</textarea>
        </div>
        <div class="form-group">
            <label for="context">Retrieved Context</label>
            <textarea id="context" rows="6" placeholder="Enter the retrieved background documents...">The most common symptoms of COVID-19 are fever, dry cough, and tiredness. Other symptoms that are less common include aches and pains, nasal congestion, headache, conjunctivitis, sore throat, diarrhea, loss of taste or smell.</textarea>
        </div>
        <div class="form-group">
            <label for="response">Generated Response to Evaluate</label>
            <textarea id="response" rows="5" placeholder="Enter the model's generated response...">Common symptoms of COVID-19 include fever, dry cough, tiredness, and occasionally internal bleeding and severe hair loss. Loss of taste and smell is also possible.</textarea>
        </div>
        <button id="analyzeBtn" class="btn" onclick="runAnalysis()">Analyze Hallucinations</button>
    </div>

    <!-- Output Section -->
    <div class="card">
        <h2>Detection Results</h2>
        
        <div id="loader" class="loader">
            <div class="spinner"></div>
            <div>Running dual forward pass and computing metrics...</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">(This may take 10-20 seconds depending on response length)</div>
        </div>
        
        <div id="initial-msg" style="text-align: center; padding: 3rem 1rem; color: #64748b;">
            Enter a query, context, and response on the left, then click <strong>Analyze</strong> to see token-level hallucination detection results.
        </div>
        
        <div id="results-area" class="results-area">
            <div id="verdict" class="verdict"></div>
            
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="label">Mean Composite</div>
                    <div id="mean-comp" class="value">-</div>
                </div>
                <div class="metric-box">
                    <div class="label">Max Composite</div>
                    <div id="max-comp" class="value">-</div>
                </div>
            </div>
            
            <label>Token-Level Map (Hover for details)</label>
            <div id="token-display" class="token-display"></div>
            
            <div class="explanation">
                <h3>How to Interpret Metrics</h3>
                <ul>
                    <li><strong>Composite (>0.75 flag):</strong> Our unsupervised, variance-weighted CAWC v5 score. Higher means higher hallucination risk.</li>
                    <li><strong>Information Gain (IG):</strong> High positive IG means the context <i>failed</i> to reduce uncertainty. The model ignored the facts.</li>
                    <li><strong>KL Divergence:</strong> High KL indicates the context caused a large distributional shift, changing predictions wildly.</li>
                    <li><strong>Confidence Drop (CD):</strong> High CD means introducing the context <i>reduced</i> the model's confidence in this specific token, indicating conflict.</li>
                </ul>
            </div>
        </div>
    </div>
</div>

<script>
    async function runAnalysis() {
        const query = document.getElementById('query').value.trim();
        const context = document.getElementById('context').value.trim();
        const response = document.getElementById('response').value.trim();
        
        if (!query || !context || !response) {
            alert("Please fill in all fields: Query, Context, and Response.");
            return;
        }
        
        document.getElementById('analyzeBtn').disabled = true;
        document.getElementById('analyzeBtn').innerText = "Analyzing...";
        document.getElementById('initial-msg').style.display = 'none';
        document.getElementById('results-area').style.display = 'none';
        document.getElementById('loader').style.display = 'block';
        
        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, context, response })
            });
            
            const data = await res.json();
            
            if (data.error) {
                alert("Error: " + data.error);
                resetUI();
                return;
            }
            
            renderResults(data);
            
        } catch (err) {
            alert("Failed to connect to the server.");
            console.error(err);
            resetUI();
        }
    }
    
    function resetUI() {
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').innerText = "Analyze Hallucinations";
        document.getElementById('loader').style.display = 'none';
        document.getElementById('initial-msg').style.display = 'block';
    }
    
    function renderResults(data) {
        document.getElementById('loader').style.display = 'none';
        document.getElementById('results-area').style.display = 'block';
        document.getElementById('analyzeBtn').disabled = false;
        document.getElementById('analyzeBtn').innerText = "Analyze Hallucinations";
        
        // Update Verdict
        const verdictEl = document.getElementById('verdict');
        if (data.mean_composite > 0.7) {
            verdictEl.className = 'verdict high';
            verdictEl.innerText = 'HIGH HALLUCINATION RISK';
        } else if (data.mean_composite > 0.5) {
            verdictEl.className = 'verdict moderate';
            verdictEl.innerText = 'MODERATE HALLUCINATION RISK';
        } else {
            verdictEl.className = 'verdict low';
            verdictEl.innerText = 'LOW HALLUCINATION RISK (Faithful)';
        }
        
        // Update Aggregates
        document.getElementById('mean-comp').innerText = data.mean_composite.toFixed(4);
        document.getElementById('max-comp').innerText = data.max_composite.toFixed(4);
        
        // Render Tokens
        const tokenContainer = document.getElementById('token-display');
        tokenContainer.innerHTML = '';
        
        for (let i = 0; i < data.tokens.length; i++) {
            const tokenText = data.tokens[i];
            const comp = data.composite[i];
            const ig = data.ig[i];
            const kl = data.kl[i];
            const cd = data.cd[i];
            
            // Map composite score (0 to 1) to a background color
            // Normal = white, High = Red
            let bg = 'transparent';
            if (comp > 0.5) {
                // Scale from 0.5 to 1.0 to opacity 0.1 to 0.8
                const intensity = Math.min(0.8, (comp - 0.5) * 2);
                bg = `rgba(239, 68, 68, ${intensity})`; // red
            }
            
            const span = document.createElement('span');
            span.className = 'token';
            span.style.backgroundColor = bg;
            if (comp > 0.8) span.style.color = '#7f1d1d';
            if (comp > 0.9) span.style.fontWeight = '700';
            
            // Ensure spaces are visible
            span.innerText = tokenText.replace(/ /g, '\u00A0');
            
            // Tooltip
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.innerHTML = `
                <div style="border-bottom: 1px solid #475569; padding-bottom: 4px; margin-bottom: 4px; text-align: center;">
                    <strong>Token: '${tokenText.replace(/ /g, ' ')}'</strong>
                </div>
                <div class="tt-row"><span class="tt-label">Composite:</span> <span class="tt-val" style="color: ${comp > 0.75 ? '#fca5a5' : '#fff'}">${comp.toFixed(4)}</span></div>
                <div class="tt-row"><span class="tt-label">IG:</span> <span class="tt-val">${ig.toFixed(4)}</span></div>
                <div class="tt-row"><span class="tt-label">KL:</span> <span class="tt-val">${kl.toFixed(4)}</span></div>
                <div class="tt-row"><span class="tt-label">CD:</span> <span class="tt-val">${cd.toFixed(4)}</span></div>
            `;
            
            span.appendChild(tooltip);
            tokenContainer.appendChild(span);
        }
    }
</script>

</body>
</html>
"""

# ------------------------------------------------------------------------
# PIPELINE BACKEND LOGIC
# ------------------------------------------------------------------------
class PipelineDetector:
    def __init__(self):
        self.train_stats = None
        self.weights = None
        self.model = None
        self.tokenizer = None
        self.evaluator = None
        self.cfg = None
        self._loaded = False

    def load_if_needed(self):
        if self._loaded:
            return
        
        print("\nLoading train statistics for normalization...")
        if not TRAIN_CACHE.exists():
            print(f"ERROR: Train cache not found at {TRAIN_CACHE}")
            print("Please ensure the project is run correctly from the root.")
            sys.exit(1)

        with open(TRAIN_CACHE, "rb") as f:
            train_data = pickle.load(f)
        train_results = train_data if isinstance(train_data, list) else train_data.get("results", [])
        
        self.train_stats = compute_train_stats(train_results)
        self.weights = compute_variance_weights(train_results)
        
        print("\nLoading Qwen2.5-7B Proxy Model...")
        self.cfg = ModelConfig(model_name="Qwen/Qwen2.5-7B", max_gpu_memory_gib=6)
        self.model, self.tokenizer = load_generator_model(self.cfg)
        self.evaluator = MetricEvaluator(self.tokenizer, generator_model=self.model)
        self._loaded = True
        print("Model loaded successfully.\n")

    def analyze(self, query: str, context: str, response: str) -> dict:
        self.load_if_needed()
        
        print("Running dual forward pass...")
        t0 = time.time()
        p_with, p_without, resp_ids = dual_forward_pass(
            self.model, self.tokenizer, query, context, response, self.cfg
        )
        elapsed = time.time() - t0
        n = len(p_with)
        print(f"Processed {n} tokens in {elapsed:.1f}s.")

        if n < 2:
            return {"error": "Response too short for meaningful analysis."}

        raw_scores = {mk: np.zeros(n, dtype=np.float64) for mk in METRIC_KEYS}
        for t in range(n):
            metrics = self.evaluator.token_metrics(p_with[t], p_without[t], int(resp_ids[t]))
            raw_scores["entropy"][t] = metrics["entropy"]
            raw_scores["ig"][t] = metrics["ig"]
            raw_scores["kl"][t] = metrics["kl"]
            raw_scores["cd"][t] = metrics["cd"]
            raw_scores["selfcheck"][t] = metrics["selfcheck"]
            raw_scores["se"][t] = np.nan # Skips heavy NLI clustering for demo speed

        composite = compute_composite_live(raw_scores, self.train_stats, n, self.weights)
        tokens = [self.tokenizer.decode([int(resp_ids[t])]) for t in range(n)]

        return {
            "tokens": tokens,
            "composite": composite.tolist(),
            "entropy": raw_scores["entropy"].tolist(),
            "ig": raw_scores["ig"].tolist(),
            "kl": raw_scores["kl"].tolist(),
            "cd": raw_scores["cd"].tolist(),
            "selfcheck": raw_scores["selfcheck"].tolist(),
            "se": raw_scores["se"].tolist(),
            "mean_composite": float(np.mean(composite)),
            "max_composite": float(np.max(composite)),
        }

# Global detector instance
detector = PipelineDetector()

# ------------------------------------------------------------------------
# CLI MODE
# ------------------------------------------------------------------------
def run_cli_mode(query: str, context: str, response: str):
    print("=" * 70)
    print("  CLI EVALUATION MODE -- CAWC v5 Hallucination Detection")
    print("=" * 70)
    print(f"Query:    {query}")
    print(f"Context:  {context[:100]}...")
    print(f"Response: {response}")
    print("-" * 70)
    
    res = detector.analyze(query, context, response)
    if "error" in res:
        print(f"ERROR: {res['error']}")
        sys.exit(1)
        
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    
    mean_comp = res["mean_composite"]
    max_comp = res["max_composite"]
    print(f"  Mean composite score: {mean_comp:.4f}")
    print(f"  Max  composite score: {max_comp:.4f}")
    
    if mean_comp > 0.7: verdict = "HIGH HALLUCINATION RISK"
    elif mean_comp > 0.5: verdict = "MODERATE HALLUCINATION RISK"
    else: verdict = "LOW HALLUCINATION RISK (likely faithful)"
    print(f"  Verdict: {verdict}\n")
    
    tokens = res["tokens"]
    comp = res["composite"]
    ig = res["ig"]
    kl = res["kl"]
    cd = res["cd"]
    n = len(tokens)
    
    print(f"  Token-level scores ({n} tokens):")
    print(f"  {'Pos':>4} {'Token':<20} {'Composite':>10} {'IG':>8} {'KL':>8} {'CD':>8} {'Flag':>6}")
    print("  " + "-" * 76)

    flagged = 0
    for i in range(n):
        flag = " !!!" if comp[i] > 0.75 else ""
        if flag: flagged += 1
        tok_display = repr(tokens[i])[:18]
        print(f"  {i:>4} {tok_display:<20} {comp[i]:>10.4f} {ig[i]:>8.4f} {kl[i]:>8.4f} {cd[i]:>8.4f}{flag}")

    print(f"\n  Flagged tokens (composite > 0.75): {flagged}/{n}")
    
    print("\n" + "=" * 70)
    print("  INTERPRETATION")
    print("=" * 70)
    top_idx = np.argsort(comp)[-3:][::-1]
    print("  Top-3 most suspicious tokens:")
    for rank, idx in enumerate(top_idx, 1):
        print(f"    #{rank}: pos={idx} token={repr(tokens[idx])} composite={comp[idx]:.4f}")
        if ig[idx] > 0: print(f"         IG={ig[idx]:.4f} -> context FAILED to reduce uncertainty")
        if kl[idx] > 0.5: print(f"         KL={kl[idx]:.4f} -> large distributional shift from context")
        if cd[idx] > 0: print(f"         CD={cd[idx]:.4f} -> context REDUCED confidence in this token")
    print()

# ------------------------------------------------------------------------
# WEB UI MODE
# ------------------------------------------------------------------------
class DemoRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            
    def do_POST(self):
        if self.path == '/api/analyze':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(body)
            
            q = data.get('query', '')
            c = data.get('context', '')
            r = data.get('response', '')
            
            try:
                result = detector.analyze(q, c, r)
            except Exception as e:
                import traceback
                traceback.print_exc()
                result = {"error": str(e)}
                
            self.send_response(200)
            self.send_header('Content-type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Mute standard HTTP logging for cleaner terminal output
        pass

def run_server(port=8080):
    # Preload the model so UI is snappy right away
    detector.load_if_needed()
    
    server = HTTPServer(('0.0.0.0', port), DemoRequestHandler)
    print("=" * 70)
    print(f"  WEB UI MODE STARTED")
    print(f"  Open your browser to: http://localhost:{port}")
    print("=" * 70)
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

# ------------------------------------------------------------------------
# MAIN ENTRY
# ------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAWC v5 Demo Script")
    parser.add_argument("--query", "-q", type=str, default=None, help="User query")
    parser.add_argument("--context", "-c", type=str, default=None, help="Retrieved context")
    parser.add_argument("--response", "-r", type=str, default=None, help="Model response")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port for Web UI mode")
    args = parser.parse_args()

    # Mode 1: CLI Evaluation
    if args.query and args.context and args.response:
        run_cli_mode(args.query, args.context, args.response)
    
    # Mode 2: Web UI
    elif not args.query and not args.context and not args.response:
        run_server(port=args.port)
        
    else:
        print("ERROR: For CLI mode, you must provide --query, --context, AND --response.")
        print("Or run without arguments to start the Web UI.")
        sys.exit(1)
