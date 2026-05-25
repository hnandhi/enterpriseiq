import json, datetime, os
from claude_client import chat, get_text
from config import MODEL_FAST

LOG_FILE = "eval_log.jsonl"

EVAL_SYSTEM = """You are an AI output quality evaluator.
Score the answer and return ONLY a valid JSON object — no markdown, no backticks, no extra text.
{
  "relevance":          <1-5>,
  "groundedness":       <1-5>,
  "completeness":       <1-5>,
  "clarity":            <1-5>,
  "hallucination_risk": "low" or "medium" or "high",
  "reasoning":          "<one sentence>",
  "flagged_claims":     ["<any claim not supported by context>"]
}
Scoring:
- relevance:          does it answer the question asked?
- groundedness:       is every claim supported by the provided context?
- completeness:       does it fully address all parts of the question?
- clarity:            is it clear and well-structured?
- hallucination_risk: did it state facts not present in the context?"""

def evaluate(question: str, context: str, answer: str) -> dict:
    user = f"""Question: {question}

Context (what Claude was given):
{context[:1500]}

Answer to evaluate:
{answer}

Return the JSON evaluation:"""

    msg = chat(MODEL_FAST, EVAL_SYSTEM, user)
    raw = get_text(msg).strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        scores = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        try:
            scores = json.loads(raw[start:end])
        except:
            scores = {"parse_error": True, "raw": raw[:200]}

    entry = {
        "ts":             datetime.datetime.now().isoformat(),
        "question":       question,
        "answer_preview": answer[:300],
        "scores":         scores
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return scores

def load_history() -> list[dict]:
    if not os.path.exists(LOG_FILE):
        return []
    results = []
    with open(LOG_FILE) as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except:
                pass
    return results

def run_golden_dataset(rag_answer_fn, namespace: str) -> dict:
    golden_path = "tests/golden_dataset.json"
    if not os.path.exists(golden_path):
        return {"error": "No golden dataset found at tests/golden_dataset.json"}
    with open(golden_path) as f:
        pairs = json.load(f)
    results = []
    for p in pairs:
        r      = rag_answer_fn(p["question"], namespace)
        scores = evaluate(p["question"], r.get("context",""), r["answer"])
        results.append({
            "question":    p["question"],
            "expected":    p["expected_keywords"],
            "scores":      scores,
            "keyword_hit": any(k.lower() in r["answer"].lower()
                               for k in p["expected_keywords"])
        })
    hits = sum(1 for r in results if r["keyword_hit"])
    return {
        "total":            len(results),
        "keyword_accuracy": round(hits / len(results) * 100, 1),
        "results":          results
    }
