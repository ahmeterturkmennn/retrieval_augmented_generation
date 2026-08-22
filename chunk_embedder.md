Great question — this confusion is actually really common and pinpoints something subtle: **the query and the chunk are different *types* of text (a question vs. an explanation), so why would their vectors be close together?**

## The key idea: embeddings aren't matching words, they're matching "meaning space"

When you embed "what is RANSAC algorithm?" and embed a chunk like "RANSAC (Random Sample Consensus) is an iterative method to estimate parameters of a model from data containing outliers...", these two texts:

- Share almost no overlapping phrasing style (one's a question, one's a definition)
- But both are **"about" the same concept** — RANSAC

The embedding model doesn't store literal words. It's a neural network trained so that its output vector represents *what the text is about* in a high-dimensional space (usually 384–1024 dimensions). Two pieces of text that are conceptually related get pushed close together in that space **during training**, even if their surface wording is totally different.

## How the model learned to do this: training on question-answer-like pairs

This is the actual mechanism — it's not magic, it's **learned behavior from training data**:

Embedding models (MiniLM, BGE, etc.) are trained using **contrastive learning**: they're shown millions of pairs like:
- `("what is RANSAC?", "RANSAC is an algorithm for robust model fitting...")` → labeled as "these go together" (positive pair)
- `("what is RANSAC?", "The recipe calls for two cups of flour...")` → labeled as "these do NOT go together" (negative pair)

During training, the model adjusts its weights so that positive pairs end up with **high cosine similarity** and negative pairs end up with **low cosine similarity**. It's trained on huge datasets that include exactly this pattern: search queries paired with the passages that answer them (things like MS MARCO, Natural Questions, StackExchange Q&A pairs, Quora duplicate questions, etc.)

So by the time you use the model, it has already "seen" the general pattern of *question phrasing ↔ answer phrasing* millions of times and has baked that association into how it maps text to vectors. It generalizes this to new questions/topics it's never literally seen (like your RANSAC example) because it learned the *general shape* of "question → definition/explanation" mapping, not this one specific fact.

## A simpler intuition: think of it as a "meaning fingerprint," not a "word fingerprint"

- Keyword search (BM25) asks: *do these two texts share the same words?*
- Embedding search asks: *do these two texts occupy the same region of meaning-space?*

"What is RANSAC algorithm" and "RANSAC is an iterative method for robust estimation" occupy the same region because they're both fundamentally about the concept RANSAC, at a similar level of abstraction (definitional). The model was trained to recognize that relationship pattern.

## Quick way to verify this yourself

```python
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

query = "what is RANSAC algorithm"
chunk_relevant = "RANSAC is an iterative method to estimate model parameters from data with outliers."
chunk_irrelevant = "The stock market closed higher today amid inflation concerns."

emb_q = model.encode(query)
emb_rel = model.encode(chunk_relevant)
emb_irr = model.encode(chunk_irrelevant)

print("similarity to relevant chunk:", util.cos_sim(emb_q, emb_rel).item())
print("similarity to irrelevant chunk:", util.cos_sim(emb_q, emb_irr).item())
```

Run this and you'll see the relevant chunk scores much higher (often 0.6–0.8+) than the irrelevant one (often near 0 or negative). That gap is the whole reason vector search works — the model was explicitly optimized to produce that gap for question/answer-shaped pairs.

## Why this matters for your RAG design (and BM25's role)

This is actually a good moment to connect back to your hybrid retrieval question: BGE models are specifically fine-tuned with **asymmetric search** in mind (short query → longer passage), which is exactly your RANSAC case. That's part of why BGE tends to outperform generic sentence-similarity models like base MiniLM on retrieval tasks specifically — MiniLM was originally trained more for *symmetric* similarity (sentence vs. sentence of similar length/style), while BGE was trained with retrieval-style query→passage pairs as a first-class objective. Worth mentioning in your README as a design justification for picking BGE.