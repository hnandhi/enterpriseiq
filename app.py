import streamlit as st
import pandas as pd
import tempfile
import os
from rag_engine   import ingest, ingest_text, answer as rag_answer
from data_agent   import analyse
from evaluator    import evaluate, load_history, run_golden_dataset
from safety       import system_prompt, check, ROLE_CONFIG
from summariser   import summarise

st.set_page_config(
    page_title="EnterpriseIQ",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("EnterpriseIQ")
    st.caption("Enterprise AI Platform — Built with Claude")
    st.divider()
    role   = st.selectbox("User role", list(ROLE_CONFIG.keys()))
    domain = st.text_input("Domain", "enterprise data & analytics")
    org    = st.text_input("Organisation", "Acme Corp")
    st.divider()
    if "total_cost" not in st.session_state:
        st.session_state.total_cost = 0.0
    st.metric("Session cost (USD)", f"${st.session_state.total_cost:.4f}")
    if st.button("Reset session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── Tabs ─────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "Document Q&A",
    "Data Analyst",
    "Executive Summary",
    "Eval Dashboard",
    "Safety & Config"
])

# ── Tab 1: Document Q&A ──────────────────────────────────
with t1:
    st.header("Document Intelligence")
    col1, col2 = st.columns([1, 2])

    with col1:
        ns = st.text_input("Collection name", "default")
        uploaded = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        if uploaded and st.button("Ingest documents"):
            for f in uploaded:
                suffix = ".pdf" if f.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                if suffix == ".pdf":
                    n = ingest(tmp_path, ns)
                else:
                    n = ingest_text(tmp_path, ns)
                os.unlink(tmp_path)
                st.success(f"{f.name}: {n} chunks ingested")

    with col2:
        q = st.text_input("Ask a question about your documents")
        if q and st.button("Ask Claude", key="rag_ask"):
            safety = check(q)
            if not safety["safe"] and safety["confidence"] == "high":
                st.error(f"Safety check [{safety['risk']}]: {safety['reason']}")
            else:
                with st.spinner("Retrieving and reasoning..."):
                    result = rag_answer(q, ns)
                st.write(result["answer"])

                with st.expander("Source chunks used"):
                    for i, c in enumerate(result["chunks"]):
                        st.caption(f"[Source {i+1}] relevance: {c['score']}")
                        st.text(c["text"][:300])

                if result["answer"] and "cannot find" not in result["answer"]:
                    with st.spinner("Evaluating answer quality..."):
                        scores = evaluate(q, result.get("context",""),
                                          result["answer"])
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Relevance",     f"{scores.get('relevance','-')}/5")
                    c2.metric("Groundedness",  f"{scores.get('groundedness','-')}/5")
                    c3.metric("Completeness",  f"{scores.get('completeness','-')}/5")
                    c4.metric("Hallucination", scores.get("hallucination_risk","-"))

# ── Tab 2: Data Analyst ──────────────────────────────────
with t2:
    st.header("Data Analyst Agent")
    csv_file = st.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx"],
        key="data_upload"
    )
    if csv_file:
        df = (pd.read_csv(csv_file)
              if csv_file.name.endswith(".csv")
              else pd.read_excel(csv_file))
        st.session_state.loaded_df = df
        st.dataframe(df.head(10), use_container_width=True)

    if "loaded_df" in st.session_state:
        q2 = st.text_input("What do you want to know about this data?",
                            key="agent_q")
        if q2 and st.button("Analyse", key="agent_run"):
            with st.spinner("Agent working..."):
                result = analyse(st.session_state.loaded_df, q2)
            st.write(result["answer"])
            ca, cb = st.columns(2)
            ca.metric("Tool calls made", result["steps"])
            cb.metric("Cost (USD)",      f"${result['cost_usd']}")
            with st.expander("Tool call trace"):
                for t in result["tool_log"]:
                    st.json(t)
            st.session_state.total_cost += result["cost_usd"]

# ── Tab 3: Executive Summary ─────────────────────────────
with t3:
    st.header("Executive Summary Generator")
    content  = st.text_area("Paste content to summarise", height=200)
    audience = st.radio(
        "Audience",
        ["executive", "technical", "board"],
        horizontal=True
    )
    if content and st.button("Generate Summary"):
        safety2 = check(content[:300])
        if not safety2["safe"] and safety2["confidence"] == "high":
            st.error("Content flagged by safety check.")
        else:
            placeholder = st.empty()
            full = ""
            for chunk in summarise(content, audience):
                full += chunk
                placeholder.write(full)

# ── Tab 4: Eval Dashboard ────────────────────────────────
with t4:
    st.header("Evaluation Dashboard")
    history = load_history()

    if history:
        rows = []
        for h in history:
            s = h.get("scores", {})
            rows.append({
                "timestamp":     h.get("ts", "")[:16],
                "question":      h.get("question", "")[:50],
                "relevance":     s.get("relevance"),
                "groundedness":  s.get("groundedness"),
                "completeness":  s.get("completeness"),
                "hallucination": s.get("hallucination_risk"),
            })
        df_e = pd.DataFrame(rows)
        st.dataframe(df_e, use_container_width=True)

        numeric = df_e[["relevance","groundedness","completeness"]].dropna()
        if not numeric.empty:
            st.subheader("Score trends")
            st.line_chart(numeric)

        good = sum(1 for h in history
                   if h.get("scores",{}).get("hallucination_risk") == "low")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total evaluations", len(history))
        c2.metric("Low hallucination %",
                  f"{round(good/len(history)*100)}%")
        avg_g = df_e["groundedness"].mean()
        c3.metric("Avg groundedness",
                  f"{avg_g:.1f}/5" if not df_e["groundedness"].isna().all()
                  else "N/A")

        st.divider()
        st.subheader("Golden dataset batch eval")
        ns_eval = st.text_input("Collection to evaluate", "procurement")
        if st.button("Run batch eval"):
            with st.spinner("Running golden dataset..."):
                batch = run_golden_dataset(rag_answer, ns_eval)
            if "error" not in batch:
                st.metric("Keyword accuracy",
                          f"{batch['keyword_accuracy']}%")
                st.dataframe(
                    pd.DataFrame(batch["results"])[
                        ["question","keyword_hit"]
                    ],
                    use_container_width=True
                )
            else:
                st.warning(batch["error"])
    else:
        st.info("Ask questions in Document Q&A to populate this dashboard.")

# ── Tab 5: Safety & Config ───────────────────────────────
with t5:
    st.header("Safety & Governance Configuration")
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("System prompt preview")
        sp = system_prompt(role, domain, org)
        st.text_area("Generated system prompt", sp, height=300)

    with col_r:
        st.subheader("Safety classifier tester")
        test_in = st.text_input("Test any user message")
        if test_in and st.button("Check safety"):
            result = check(test_in)
            if result["safe"]:
                st.success(
                    f"Safe [{result['confidence']}]: {result['reason']}")
            else:
                st.error(
                    f"Flagged [{result['risk']}] "
                    f"[{result['confidence']}]: {result['reason']}")
