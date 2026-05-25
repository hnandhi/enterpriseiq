import pdfplumber, chromadb, re
from sentence_transformers import SentenceTransformer
from claude_client import chat, get_text
from config import MODEL_MAIN, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma   = chromadb.PersistentClient(path="./chroma_db")

def _chunk(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) > CHUNK_SIZE and cur:
            chunks.append(cur.strip())
            cur = cur[-CHUNK_OVERLAP:] + " " + s
        else:
            cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks

def ingest(file_path: str, namespace: str) -> int:
    with pdfplumber.open(file_path) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    text   = "\n".join(pages)
    chunks = _chunk(text)
    embeds = embedder.encode(chunks, show_progress_bar=False).tolist()
    col = chroma.get_or_create_collection(namespace)
    existing = col.count()
    col.add(
        documents  = chunks,
        embeddings = embeds,
        ids        = [f"{namespace}_{existing+i}" for i in range(len(chunks))],
        metadatas  = [{"source": file_path, "chunk_idx": existing+i}
                      for i in range(len(chunks))]
    )
    return len(chunks)

def retrieve(query: str, namespace: str) -> list[dict]:
    try:
        col    = chroma.get_collection(namespace)
    except Exception:
        return []
    q_emb  = embedder.encode([query]).tolist()
    result = col.query(query_embeddings=q_emb, n_results=TOP_K,
                       include=["documents","metadatas","distances"])
    return [{"text": d, "meta": m, "score": round(1 - s, 3)}
            for d, m, s in zip(result["documents"][0],
                                result["metadatas"][0],
                                result["distances"][0])]

def answer(query: str, namespace: str) -> dict:
    chunks = retrieve(query, namespace)
    if not chunks:
        return {"answer": "No documents ingested yet. Please upload a PDF first.",
                "chunks": [], "context": "", "usage": {}}

    context = "\n\n".join(
        [f"[Source {i+1}] (relevance {c['score']}):\n{c['text']}"
         for i, c in enumerate(chunks)]
    )
    system = """You are an enterprise knowledge assistant.
Rules:
1. Answer ONLY from the provided context — never from prior knowledge.
2. Cite sources inline using [Source N] whenever you use information.
3. If the answer is not in context, say exactly:
   "I cannot find this in the provided documents."
4. Be concise and precise. Executives read this output."""

    msg = chat(MODEL_MAIN, system,
               f"Context:\n{context}\n\nQuestion: {query}")
    return {
        "answer":  get_text(msg),
        "chunks":  chunks,
        "context": context,
        "usage":   {"input": msg.usage.input_tokens,
                    "output": msg.usage.output_tokens}
    }

def ingest_text(file_path: str, namespace: str) -> int:
    """Ingest a plain .txt file — useful for testing without PDF."""
    with open(file_path, "r") as f:
        text = f.read()
    chunks = _chunk(text)
    embeds = embedder.encode(chunks, show_progress_bar=False).tolist()
    col = chroma.get_or_create_collection(namespace)
    existing = col.count()
    col.add(
        documents  = chunks,
        embeddings = embeds,
        ids        = [f"{namespace}_{existing+i}" for i in range(len(chunks))],
        metadatas  = [{"source": file_path, "chunk_idx": existing+i}
                      for i in range(len(chunks))]
    )
    return len(chunks)
