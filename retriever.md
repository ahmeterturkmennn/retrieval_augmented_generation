The problem RRF solves

You have two search methods giving you ranked lists of documents:

dense_search: embedding/vector similarity search — good at semantic meaning
bm25_search: keyword search — good at exact term matches

Each gives you a list of (doc_id, text, score). The problem: the scores aren't comparable. BM25 scores might range from 0 to 40. Dense search returns distances (lower = better, often between 0 and 2). You can't just average them — a BM25 score of 12 and a dense distance of 0.3 mean nothing next to each other.

RRF's trick: throw away the scores, use rank position instead

Instead of asking "how good is this score," RRF asks "what position did this document rank in each list?" Position is comparable across any two systems, regardless of their internal scoring scale.

The formula for a single list:

1 / (k + rank + 1)

Look at the loop:

python
for rank, (doc_id, text, _) in enumerate(dense_results):
    scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
enumerate gives rank = 0 for the top result, 1 for second, etc. Note the raw score (_) is discarded — only rank matters.
1 / (k + rank + 1): as rank grows, this shrinks. Rank 0 → 1/61. Rank 1 → 1/62. Rank 9 → 1/70. So top-ranked docs get bigger contributions, but the falloff is gentle.
k=60 is a smoothing constant. It dampens the difference between rank 0 and rank 1 (61 vs 62 barely differ) so that a document doesn't get wildly rewarded just for being #1 vs #2. It also prevents low ranks from contributing ~0.
Fusing the two lists
python
scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

This same accumulation happens for both dense_results and bm25_results, into the same scores dict. So:

A document that ranks well in both lists gets two additive contributions → high fused score.
A document that only appears in one list gets just one contribution.
A document ranking #1 in dense search but absent from BM25 results only gets credit from the one list.

texts[doc_id] = text just keeps track of the actual text alongside the id, so you can return it later — it has nothing to do with the ranking math itself.

Finally:

python
fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)

Sort all documents by their combined score, descending. That's your fused ranking — a consensus between the two retrieval methods, agnostic to their raw score scales.

Where reranking fits in

RRF gives you a cheap, scale-free way to merge two rankings. It's not very precise — it doesn't actually understand the query or the document, it just counts positions. That's what rerank() fixes afterward:

python
ce_scores = reranker.predict(pairs)

This runs a cross-encoder — a model that jointly looks at (query, document_text) together and outputs a relevance score. This is much more accurate than RRF but also much more expensive (it can't be precomputed like embeddings), so you only run it on the top candidates that survived the cheap fusion step.

The pipeline as a whole
dense_search → ranked list by meaning
bm25_search → ranked list by keyword overlap
reciprocal_rank_fusion → merge both rankings using position-based scores (cheap, scale-free)
rerank → take that fused shortlist and score it precisely with a cross-encoder (expensive, accurate)
