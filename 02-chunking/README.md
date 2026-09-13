# ✂️ Chunking Strategies & Trade-offs

[← Previous: Data Ingestion](../01-data-ingestion/README.md) | [Back to Master README](../README.md) | [Next: Embeddings →](../03-embeddings/README.md)

---

## Table of Contents

- [Topic Overview](#topic-overview)
- [Why It Matters in RAG](#why-it-matters-in-rag)
- [Mental Model](#mental-model)
- [Core Theory](#core-theory)
- [Chunking Paradigms](#chunking-paradigms)
- [The Sliding Window Formula](#the-sliding-window-formula)
- [Language-Aware Splitting](#language-aware-splitting)
- [Comparison Table](#comparison-table)
- [Code Examples](#code-examples)
- [Common Mistakes](#common-mistakes)
- [Key Takeaways](#key-takeaways)
- [Checkpoint Questions](#checkpoint-questions)

---

## Topic Overview

**Chunking** is the process of breaking long, continuous text into smaller passages before embedding them as vectors. It directly determines retrieval quality and is one of the most impactful design decisions in a RAG pipeline.

---

## Why It Matters in RAG

You cannot throw a 100-page document at an embedding model or LLM. Chunking sits between ingestion and embedding — it determines what "units of knowledge" your vector store contains.

```mermaid
flowchart LR
    A["📄 Full Documents"] --> B["✂️ Chunking"]
    B --> C["📦 Chunks"]
    C --> D["🔢 Embedding"]
    D --> E["💾 Vector Store"]

    style B fill:#e94560,stroke:#fff,color:#fff
```

**The chunk is the atomic unit of retrieval.** When a user asks a question, your system retrieves *chunks*, not full documents. The quality of those chunks directly determines answer quality.

---

## Mental Model

Think of chunking as **cutting a book into index cards**:

- **Too small** (20 tokens): Each card has a sentence fragment — "He did it" without knowing who "He" is
- **Too large** (2000 tokens): Each card covers 3 different topics — the embedding averages out all topics, diluting the semantic signal
- **Just right** (200–500 tokens): Each card captures one complete idea with enough context

---

## Core Theory

### The Chunk Size Problem

| Chunk Size | Effect on Embeddings | Effect on Retrieval |
|-----------|---------------------|-------------------|
| **Too small** (< 50 tokens) | Captures fragments, not concepts | Returns incomplete context; pronouns without referents |
| **Too large** (> 1000 tokens) | Averages multiple topics into one vector | Poor similarity scores; diluted semantic signal |
| **Optimal** (200–500 tokens) | Captures a single coherent idea | Retrieves focused, relevant passages |

### Chunk Overlap

**Overlap** (`chunk_overlap`) ensures that sentences sitting on chunk boundaries aren't sliced in half, preserving semantic continuity across consecutive chunks.

- Without overlap: Information at chunk boundaries is lost
- With overlap: The end of chunk N overlaps with the start of chunk N+1

---

## The Sliding Window Formula

Chunking with overlap works as a **sliding window**:

$$\text{Step Size} = \text{chunk\_size} - \text{chunk\_overlap}$$

### Example: `chunk_size=200, chunk_overlap=0` (Step size = 200)

```
Chunk 0: characters 0 → 200
Chunk 1: characters 200 → 400    (no shared text)
Chunk 2: characters 400 → 600
```

### Example: `chunk_size=200, chunk_overlap=150` (Step size = 50)

```
Chunk 0: characters 0 → 200      (unchanged — nothing before it!)
Chunk 1: characters 50 → 250     (150 chars overlap with Chunk 0)
Chunk 2: characters 100 → 300    (150 chars overlap with Chunk 1)
```

> **Key insight:** `result[0]` (the first chunk) is always the same regardless of overlap — there's nothing *before* the start of the document to overlap with. Overlap affects chunk 1 onwards.

> **Warning:** Large overlap values cause `len(result)` to explode because the window advances by only a small step each time.

---

## Chunking Paradigms

### 1. Fixed-Size / Character Chunking

Blindly splits every N characters. Crude — frequently cuts words and sentences mid-thought.

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''  # Brute force — slices words in half!
)
```

### 2. Recursive Character Chunking

**The industry standard baseline.** Attempts to split by natural text hierarchy:

1. First tries paragraphs (`\n\n`)
2. Then sentences (`\n`)
3. Then words (` `)
4. Finally raw characters as a last resort

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)
```

### 3. Semantic Chunking

Calculates embedding distances between adjacent sentences and creates a boundary whenever the **topic changes significantly**.

```python
from langchain_experimental.text_splitter import SemanticChunker

chunker = SemanticChunker(embeddings_model)
```

### 4. Language-Aware Splitting (AST-based)

Uses a language-aware parser that understands code syntax (functions, classes, if/else blocks). Only creates chunk boundaries at **safe syntactic boundaries**.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50,
)
```

---

## Language-Aware Splitting

### Why Standard Splitters Break Code

When you feed source code (which has no natural sentences) to `RecursiveCharacterTextSplitter`:

```
❌ Generic splitter output:
def calculate_gpa(stu
dent):
    total = sum(stu.
```

The function is cut in half — it can't be parsed, executed, or understood by an LLM.

### How Language.PYTHON Fixes It

`RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)` uses **AST-based tokenization** (via tree-sitter) that respects function boundaries, class definitions, if/elif/else blocks, and docstrings:

```
✅ Language-aware splitter output:
def calculate_gpa(student):
    total = sum(student.grades)
    return total / len(student.grades)
```

Each chunk is a **complete, syntactically valid** unit of code.

### Why This Matters for RAG

When you embed code chunks, you want each vector to represent a **complete logical unit** (a function, a class). Otherwise, retrieval returns half-cut fragments that cannot be executed or understood by the LLM, causing hallucinations or syntax errors.

---

## Comparison Table

| Splitter | How It Works | Strengths | Weaknesses | Best For |
|----------|-------------|-----------|------------|----------|
| `CharacterTextSplitter` | Fixed-size windows on raw characters | Simple, predictable | Breaks mid-word, loses structure | Quick prototyping only |
| `RecursiveCharacterTextSplitter` | Tries paragraphs → sentences → words → chars | Preserves natural text hierarchy | Unaware of code/language tokens | **General text (industry standard)** |
| `SemanticChunker` | Embedding distance between adjacent sentences | Respects topic boundaries | Requires embedding model at chunk time | Topic-diverse documents |
| `from_language(Language.PYTHON)` | AST-based tokenization respecting syntax | Keeps functions/classes intact | Requires tree-sitter; language-specific | **Source code indexing** |

---

## Code Examples

### Comparing Splitters Side-by-Side

```python
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, grades):
        self.name = name
        self.grades = grades

    def calculate_gpa(self):
        total = sum(self.grades)
        return total / len(self.grades)

if __name__ == "__main__":
    s = Student("Alice", [90, 85, 92])
    print(s.calculate_gpa())
"""

# ❌ Naive character splitter
naive = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator='')
naive_chunks = naive.split_text(text)
print("--- Naive splitter chunk #2 ---")
print(naive_chunks[1])

# ✅ Language-aware splitter
py_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=100, chunk_overlap=0
)
py_chunks = py_splitter.split_text(text)
print("\n--- Language-aware splitter chunk #2 ---")
print(py_chunks[1])
```

### Observing Overlap Behavior

```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=150, separator='')
result = splitter.split_documents(docs)

print("Total chunks created:", len(result))
print("\n--- CHUNK 0 ---")
print(result[0].page_content)
print("\n--- CHUNK 1 ---")
print(result[1].page_content)  # Notice: start repeats middle/end of chunk 0
```

---

## Common Mistakes

1. **Using CharacterTextSplitter for production** — it breaks words and sentences; always prefer `RecursiveCharacterTextSplitter`
2. **Setting overlap too high** — causes an explosion in chunk count with mostly duplicate content
3. **Using default text splitters for code** — standard splitters destroy code syntax; use `from_language()`
4. **Ignoring chunk size tuning** — the optimal chunk size depends on your embedding model's training data and your document type
5. **Not inspecting chunks** — always print sample chunks to verify quality before embedding

---

## Key Takeaways

1. **Chunking determines retrieval quality** — it defines the "atoms" your system retrieves
2. **Too small = lost context; too large = diluted signal** — find the sweet spot (200–500 tokens for text)
3. **RecursiveCharacterTextSplitter is the industry baseline** for natural language text
4. **Language-aware splitting is essential for code** — prevents syntax breakage
5. **Overlap preserves boundary context** — but use it judiciously to avoid chunk explosion
6. **The sliding window formula**: `Step Size = chunk_size - chunk_overlap`
7. **Always inspect your chunks** before embedding — verify they contain coherent, complete units

---

## Checkpoint Questions

### Checkpoint 1: Code Chunking Problem

**Question:** If you are indexing a technical coding manual containing large Python functions and use `RecursiveCharacterTextSplitter` with a small `chunk_size`, what problem will occur when an engineer asks "How do I implement function X?"

**Answer:** The generic splitter will split in the middle of statements, breaking syntax. The retrieved chunk will be a syntactically incomplete fragment like `def calculate_gpa(stu\ndent):` — which the LLM cannot parse, execute, or reason about correctly.

**Explanation:** `RecursiveCharacterTextSplitter` splits by paragraphs → sentences → words → characters. It has no understanding of Python syntax (function boundaries, indentation, class definitions). The fix is to use `RecursiveCharacterTextSplitter.from_language(Language.PYTHON, ...)` which uses an AST-based parser that only creates boundaries at safe syntactic points.

---

### Checkpoint 2: Overlap Mechanics

**Question:** Why does `result[0]` (the first chunk) remain identical regardless of the `chunk_overlap` value?

**Answer:** Because there is no text *before* the start of the document for the first chunk to overlap with. `result[0]` always starts at character 0 and extends to `chunk_size`. Overlap only affects `result[1]` onwards, where each new chunk starts `step_size` characters after the previous one instead of `chunk_size` characters.

**Explanation:** The sliding window formula `Step Size = chunk_size - chunk_overlap` controls how far forward the window moves. For chunk 0, the window always starts at position 0. For chunk 1 with `chunk_size=200, chunk_overlap=150`, the window starts at position 50 (step size = 50), meaning it shares 150 characters with chunk 0.

---

### Checkpoint 3: Language-Aware Necessity

**Question:** When indexing Python files for a code-assistant RAG system, what goes wrong with the generic `RecursiveCharacterTextSplitter` (without `Language.PYTHON`), and which approach solves it?

**Answer:** The generic splitter can split in the middle of a statement, breaking syntax. The solution is `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` which uses AST-based tokenization that respects function boundaries, class definitions, and other language structures.

**Explanation:** Each embedded chunk should represent a **complete logical unit**. If a function is split across two chunks, neither chunk contains the full implementation. When retrieved, the LLM receives an incomplete fragment and either hallucinates the rest or produces syntax errors.

---

[← Previous: Data Ingestion](../01-data-ingestion/README.md) | [Back to Master README](../README.md) | [Next: Embeddings →](../03-embeddings/README.md)
