"""
Why not just use the vector store alone?
Dense (embedding) search is great at semantic matches but sometimes weak on:
Exact keywords, codes, names, acronyms, IDs (e.g., "error code E1042", "Article 12.3") —
embeddings can blur these together with similar-but-wrong text.
Rare/technical terms that didn't appear much in the embedding model's training data.
BM25 (classic keyword search, like what search engines used before embeddings) is the 
opposite: great at exact term matches, bad at synonyms/paraphrases.
Hybrid search runs both and merges the results, so you get semantic understanding and 
keyword precision. This is why it's a strong portfolio point — it shows you understand 
the failure modes of each method individually.
user query
   ├─→ dense search (embed query → vector store) → ranked list A
   └─→ BM25 search (keyword score over raw text)   → ranked list B
                    ↓
        combine A + B via Reciprocal Rank Fusion (RRF)
                    ↓
             final ranked chunk list
                    ↓
        (optional: rerank with cross-encoder)
                    ↓
             top-k chunks → passed to LLM
"""

"""
Why you need reranking at all, if you already have hybrid search

Your dense + BM25 + RRF pipeline is a bi-encoder approach: the query gets embedded
separately from each chunk, and you compare vectors afterward. This is fast (you can 
precompute all chunk embeddings once), but it's also a bit "lossy" — the query and c
hunk never actually get looked at together by the model.

A cross-encoder reranker does the opposite: it takes the query and one chunk together,
concatenated, and passes both through the model at once:

[CLS] how does ransac work [SEP] RANSAC is used to fit lines to noisy data... [SEP]

The model then directly outputs a single relevance score for that pair. Because the model can
"see" the query and chunk simultaneously (with full cross-attention between every word of both), 
it catches subtler relevance signals that separate embeddings miss — e.g., it notices that a chunk 
merely mentions RANSAC in passing is less relevant than a chunk that explains RANSAC.

The catch: this is much slower, because you can't precompute anything — you must run the model fresh
for every (query, chunk) pair, at query time. That's exactly why you don't use it for the first-pass
search over your whole corpus (imagine scoring a cross-encoder pair for 10,000 chunks — too slow), 
but only on the small shortlist (your top 10–20 fused results) to re-sort them more accurately before picking the final top-k.

Where it fits in your pipeline
user query
   ├─→ dense search (fast, whole corpus)   ─┐
   └─→ BM25 search (fast, whole corpus)    ─┼→ RRF fusion → top ~20 candidates
                                             ↓
                              cross-encoder reranker (slow but accurate, only on these 20)
                                             ↓
                                    re-sorted top 5 → passed to LLM

Reranking doesn't replace your hybrid retrieval — it sits after it, refining a small candidate set.
"""

from rank_bm25 import BM25Okapi
def rerank(query, fused_results,reranker, top_k):
    pairs = [(query, text) for doc_id, text, _ in fused_results]

    ce_scores = reranker.predict(pairs)
    reranked = sorted(
        zip(fused_results, ce_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [(doc_id, text, float(ce_score)) for (doc_id, text, _), ce_score in reranked[:top_k]]

def bm25_search(query, bm25, all_chunks, k):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(f"chunk_{i}", all_chunks[i], scores[i]) for i in ranked_idx]
def dense_search(query, collection, k):
    results = collection.query(query_texts=[query], n_results=k)
    # returns list of (id, document, score) tuples
    return list(zip(results["ids"][0], results["documents"][0], results["distances"][0]))

def reciprocal_rank_fusion(dense_results, bm25_results, k=60):
    scores = {}
    texts = {}

    for rank, (doc_id, text, _) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        texts[doc_id] = text

    for rank, (doc_id, text, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
        texts[doc_id] = text

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_id, texts[doc_id], score) for doc_id, score in fused]
"""
def hybrid_retrieve(query, collection, bm25, all_chunks, top_k=5):
    dense = dense_search(query, collection, k=10)
    sparse = bm25_search(query, bm25, all_chunks, k=10)
    fused = reciprocal_rank_fusion(dense, sparse)
    return fused[:top_k]
"""
def hybrid_retrieve(query, collection, bm25, all_chunks,reranker, candidate_k, top_k):
    dense = dense_search(query, collection, k=candidate_k//2)
    sparse = bm25_search(query, bm25, all_chunks, k=candidate_k//2)
    fused = reciprocal_rank_fusion(dense, sparse,60)         
    final = rerank(query, fused[:candidate_k], reranker,top_k=top_k) 
    return final
def retrieve(reranker,chunks,query_text, collection,candidate_k,top_k):
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    results = hybrid_retrieve(query_text, collection, bm25, chunks,reranker, candidate_k, top_k)
    return results
