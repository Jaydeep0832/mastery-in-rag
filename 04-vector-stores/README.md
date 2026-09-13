# 💾 Vector Stores: FAISS vs ChromaDB

[← Previous: Embeddings](../03-embeddings/README.md) | [Back to Master README](../README.md) | [Next: Retrieval & Generation →](../05-retrieval-and-generation/README.md)

---

## Table of Contents

- [Topic Overview](#topic-overview)
- [Why It Matters in RAG](#why-it-matters-in-rag)
- [Mental Model](#mental-model)
- [FAISS: The Engine](#faiss-the-engine)
- [ChromaDB: The Complete Car](#chromadb-the-complete-car)
- [Feature Comparison](#feature-comparison)
- [Metadata Filtering](#metadata-filtering)
- [Code Examples](#code-examples)
- [When to Use Which](#when-to-use-which)
- [Common Mistakes](#common-mistakes)
- [Key Takeaways](#key-takeaways)
- [Checkpoint Questions](#checkpoint-questions)

---

## Topic Overview

After embedding your chunks into dense vectors, you need somewhere to **store** them and **search** them efficiently. Vector stores are specialized databases optimized for nearest-neighbor search over high-dimensional vectors.

This section compares two widely-used options: **FAISS** (a low-level library) and **ChromaDB** (a full-featured vector database).

---

## Why It Matters in RAG

The vector store is your RAG pipeline's **memory**. It determines:

- **How fast** retrieval is (milliseconds vs. seconds)
- **What metadata** you can filter on (source, author, date)
- **How persistent** your index is (in-memory vs. on-disk)
- **How scalable** your system is (thousands vs. billions of vectors)

```mermaid
flowchart LR
    EMB["🔢 Embeddings"] --> VS["💾 Vector Store"]
    VS --> RET["🔍 Retrieval"]
    RET --> GEN["🤖 Generation"]

    style VS fill:#9b59b6,stroke:#fff,color:#fff
```

---

## Mental Model

Think of the difference between FAISS and ChromaDB as the difference between an **engine component** and a **complete car**:

- **FAISS** = The **engine**. Incredibly powerful and fast, but you need to build the rest of the car yourself (storage, metadata, CRUD operations)
- **ChromaDB** = The **complete car**. Ready to drive — it handles vectors, text, metadata, persistence, and filtering all in one package

---

## FAISS: The Engine

**FAISS (Facebook AI Similarity Search)** is a low-level, high-performance C++ library (with Python bindings) created by Meta.

### How It Works

- FAISS **does not know** what a "PDF", "Document", or "Text chunk" is
- It only understands **pure arrays of floating-point numbers** (`float32`)
- It loads vectors into RAM (or GPU memory) and applies specialized algorithms (Flat Inner Product, IVFFlat clustering, HNSW graphs) for lightning-fast nearest-neighbor search
- **You must manually maintain a separate file** (like `chunks.pkl`) to remember what text belongs to each vector

### Strengths

- ⚡ Extreme speed — scans millions of vectors in milliseconds
- 🖥️ GPU acceleration available
- 🔧 Fine-grained control over index types and algorithms

### Limitations

- No built-in text/metadata storage
- No native CRUD operations (create/update/delete individual vectors)
- No metadata filtering without custom code

---

## ChromaDB: The Complete Car

**ChromaDB** is a full-fledged, developer-friendly vector database built specifically for AI and RAG applications.

### How It Works

- Chroma combines **vector indexing** (using HNSW algorithms) with a **lightweight relational database** (SQLite) under the hood
- When you give Chroma a document, it stores **everything together**:
  - The vector embedding
  - The raw text content
  - The metadata dictionary (author, page number, date, etc.)
  - Unique document IDs
- Supports **filtered queries** out of the box

### Strengths

- 📦 Stores vectors + text + metadata in one place
- 🔍 Native metadata filtering
- 💾 Automatic persistence to disk (`persist_directory`)
- ✏️ Full CRUD operations (add, update, delete individual documents)

### Limitations

- Slower than raw FAISS for very large-scale (billions) vector searches
- Additional storage overhead from SQLite

---

## Feature Comparison

| Feature | FAISS | ChromaDB |
|---------|-------|----------|
| **Type** | Vector Indexing Library | Complete Vector Database |
| **Data Stored** | Only raw vector arrays | Vectors + Raw Text + Metadata + IDs |
| **Storage Engine** | In-Memory (saved as raw `.bin`/`.faiss`) | Embedded SQLite + Parquet files on disk |
| **Metadata Filtering** | ❌ Difficult (requires custom logic) | ✅ Native (`where={"author": "Alice"}`) |
| **CRUD Operations** | ❌ Must rebuild entire index | ✅ Add, update, delete individual docs |
| **Persistence** | Manual (`faiss.write_index()`) | Automatic (`persist_directory`) |
| **GPU Support** | ✅ Yes (faiss-gpu) | ❌ No |
| **Best For** | Extreme speed, millions/billions of vectors | Rapid RAG development, structured metadata |
| **Learning Curve** | Higher (manual plumbing) | Lower (developer-friendly API) |

---

## Metadata Filtering

One of ChromaDB's most powerful features for production RAG:

```python
# Store documents with metadata
Document(
    page_content="Settlement details for Client X...",
    metadata={"lawyer_id": "lawyer_101", "case_year": 2024}
)

# Filter at query time — only search within a specific lawyer's files
retriever = chroma_db.as_retriever(
    search_kwargs={"filter": {"lawyer_id": "lawyer_101"}}
)
```

Chroma's internal SQLite database **instantly excludes** all other lawyers' files before calculating vector similarities.

**In pure FAISS**, you would either have to:
- Create **50 separate FAISS index files** (one per lawyer), or
- Pull 100+ results into Python and filter them manually — slow and messy

---

## Code Examples

### ChromaDB: Persistence and Retrieval

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create and persist
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"  # Saved to disk automatically!
)

# Later: Load from disk (no re-embedding needed)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# Search with scores
results = vectorstore.similarity_search_with_score("What is data protection?", k=3)
```

### FAISS with LangChain Wrapper

```python
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create
vectorstore = FAISS.from_documents(chunks, embedding_model)

# Save and load
vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("faiss_index", embedding_model)
```

> **Note:** LangChain's FAISS wrapper handles the metadata pairing automatically — you don't need to write manual `pickle.dump()` code like with raw FAISS.

### End-to-End RAG Chain with Chroma

```python
from langchain_core.output_parsers import StrOutputParser

# Connect retriever → prompt → LLM
retriever = vectorstore.as_retriever()
chain = retriever | prompt | llm | StrOutputParser()
answer = chain.invoke("What are the penalties for data breach?")
```

---

## When to Use Which

| Scenario | Recommended | Why |
|----------|------------|-----|
| Quick RAG prototype | **ChromaDB** | Automatic persistence, metadata, simple API |
| Multi-tenant application (users can only see their own data) | **ChromaDB** | Native metadata filtering |
| Millions/billions of vectors with GPU | **FAISS** | Unmatched speed and GPU acceleration |
| Already using PostgreSQL/MongoDB for documents | **FAISS** | Use FAISS as a pure search index alongside your existing DB |
| Need to update/delete individual documents | **ChromaDB** | Full CRUD support |
| Embedding-only search, no metadata needed | **FAISS** | Minimal overhead, maximum speed |

---

## Common Mistakes

1. **Using raw FAISS for a prototype** — adds unnecessary complexity; use ChromaDB or LangChain's FAISS wrapper
2. **Forgetting `persist_directory` in ChromaDB** — without it, your index is lost when the process ends
3. **Using different embedding models for indexing vs. querying** — vectors won't be in the same space
4. **Not considering metadata filtering needs upfront** — retrofitting FAISS for metadata is painful

---

## Key Takeaways

1. **FAISS is a low-level vector indexing library** — ultra-fast but only stores raw vectors
2. **ChromaDB is a complete vector database** — stores vectors + text + metadata + IDs
3. **Metadata filtering is ChromaDB's killer feature** for multi-tenant and filtered search applications
4. **LangChain's FAISS wrapper** adds automatic metadata handling, bridging the gap
5. **Choose based on your needs**: prototype → ChromaDB; extreme scale → FAISS; metadata filtering → ChromaDB
6. **Always persist your index** — recomputing embeddings is expensive and wasteful

---

## Checkpoint Questions

### Checkpoint 1: Multi-Tenant RAG

**Question:** You are building a RAG application for a law firm with 50 different lawyers. Each lawyer should only be able to retrieve search results from their own client files. Between FAISS and ChromaDB, which one makes implementing this feature simpler, and what capability would you use?

**Answer:** **ChromaDB**, using **metadata filtering**. You attach a `lawyer_id` field to each document's metadata during ingestion, then filter at query time with `search_kwargs={"filter": {"lawyer_id": "lawyer_101"}}`. Chroma's internal database instantly excludes all other lawyers' files before running the vector search.

**Explanation:** In pure FAISS, there is no native metadata filtering engine. You would either need to maintain 50 separate FAISS index files (one per lawyer) or retrieve a large number of results and filter manually in Python — both approaches are slow, error-prone, and difficult to maintain. ChromaDB handles this natively because it stores metadata alongside vectors in a relational database (SQLite).

---

[← Previous: Embeddings](../03-embeddings/README.md) | [Back to Master README](../README.md) | [Next: Retrieval & Generation →](../05-retrieval-and-generation/README.md)
