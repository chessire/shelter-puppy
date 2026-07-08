"""임베딩 — bge-m3(Ollama, 1024차원). 검색 쿼리와 청크가 같은 모델을 써야 한다."""

from __future__ import annotations

from . import EMBED_DIM, EMBED_MODEL


def embed_texts(texts: "list[str]") -> "list[list[float]]":
    import ollama
    if not texts:
        return []
    r = ollama.embed(model=EMBED_MODEL, input=list(texts))
    vecs = list(r.embeddings)
    if vecs and len(vecs[0]) != EMBED_DIM:
        raise RuntimeError(
            f"임베딩 차원 불일치: {len(vecs[0])} != {EMBED_DIM} (스키마 vector({EMBED_DIM}))")
    return vecs
