# EnterpriseIQ — Enterprise AI Platform built with Claude

A production-ready Claude-powered platform demonstrating the 5 core
patterns enterprise customers need when adopting LLMs.

## Live demo
http://165.232.181.184:8502

## What it does

| Module | Capability | JD requirement covered |
|--------|-----------|----------------------|
| Document Q&A | RAG pipeline over PDFs with citations | LLM frameworks, cloud architecture |
| Data Analyst Agent | Multi-tool agentic loop over CSV/Excel | Tool use, Python, data analysis |
| Executive Summary | Streaming summaries for 3 audience types | C-suite to engineering translation |
| Eval Dashboard | LLM-as-judge scoring + golden dataset batch eval | Evaluation framework design |
| Safety & Config | Role-based system prompts + safety classifier | Safe and beneficial AI deployment |

## Architecture
User query
│
▼
Safety classifier (Claude Haiku — fast, cheap)
│
▼
RAG retrieval (ChromaDB + sentence-transformers)
│
▼
Claude Sonnet — reasoning + answer generation
│
▼
LLM-as-judge evaluator (Claude Haiku)
│
▼
Logged to JSONL eval store
## Key design decisions

- **Native Anthropic SDK over LangChain** — fewer abstractions, easier
  debugging, full control over message structure. Critical for production
  enterprise deployments where failure tracing matters.

- **Model routing** — Haiku for safety checks and evaluation (fast, cheap),
  Sonnet for reasoning and Q&A. Reduces API cost by ~70% vs using
  Sonnet everywhere.

- **ChromaDB for PoC, swappable for production** — local vector store
  with zero config. In a real enterprise deployment I would swap to
  Pinecone (AWS) or pgvector (Postgres-native) depending on the
  customer's cloud footprint.

- **LLM-as-judge evaluation** — scales to any query volume without
  human reviewers. Every response scored on relevance, groundedness,
  completeness, and hallucination risk. Logged for trend analysis.

- **Role-based system prompts** — extends Anthropic's Constitutional AI
  into customer-specific governance. Viewer, Analyst, Manager, Admin
  roles each get different data access rules baked into the system prompt.

## What I would add for a real enterprise deployment

- OAuth2 / SAML authentication
- PostgreSQL for eval log storage and querying
- Pinecone vector store for scale
- Prompt caching for frequently-used system prompts (Anthropic feature)
- CI/CD pipeline with automated golden dataset eval on every deploy
- Rate limiting per user
- Audit log for all queries and responses
- Grafana monitoring dashboard

## Running locally

```bash
git clone https://github.com/hnandhi/enterpriseiq.git
cd enterpriseiq
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=your-key-here" > .env
streamlit run app.py
```

## Tech stack

- Claude Sonnet + Haiku via Anthropic Python SDK
- ChromaDB — vector store
- sentence-transformers — embeddings (all-MiniLM-L6-v2)
- Streamlit — UI
- pandas — data handling
- pdfplumber — PDF extraction
- Deployed on DigitalOcean Ubuntu 24 VPS

## Author

Hari Kumar — Senior Data & AI Architect
15+ years enterprise architecture across Harman, KPMG, PayPal, NTT Data
github.com/hnandhi
