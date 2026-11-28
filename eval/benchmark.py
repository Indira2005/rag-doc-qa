"""
Step 1 (free, no LLM): compare chunk size, top-k and dense / hybrid / rerank.
    python -m eval.benchmark retrieval paper.pdf
Step 2 (uses Groq): RAGAS on your best setting.
    python -m eval.benchmark ragas paper.pdf --size 1000 --k 5 --mode rerank
"""
import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from rag import config
from rag.chunking import chunk_pdf
from rag.index import Index

HERE = Path(__file__).parent


def load_golden():
    return [json.loads(l) for l in (HERE / "golden.jsonl").read_text().splitlines() if l.strip()]


def build(pdf, size):
    index = Index(name=f"bench-{size}", path=None)
    index.add(chunk_pdf(pdf, os.path.basename(pdf), size, int(size * 0.15)))
    return index


def retrieval(pdf):
    golden = load_golden()
    answerable = [g for g in golden if g["pages"]]
    unanswerable = [g for g in golden if not g["pages"]]
    rows = []
    for size in (500, 1000, 1500):
        index = build(pdf, size)
        for mode in ("dense", "hybrid", "rerank"):
            for k in (3, 5, 8):
                hit, rr = 0, 0.0
                for g in answerable:
                    pages = [h["page"] for h in index.search(g["question"], k, mode)]
                    rank = next((i for i, p in enumerate(pages, 1) if p in g["pages"]), None)
                    if rank:
                        hit += 1
                        rr += 1 / rank
                rows.append({"chunk_size": size, "mode": mode, "top_k": k,
                             "hit_rate": round(hit / len(answerable), 3),
                             "mrr": round(rr / len(answerable), 3)})
                print(rows[-1])
    with open(HERE / "retrieval_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Tune the refusal cutoff: compare best scores for answerable vs unanswerable questions.
    index = build(pdf, config.CHUNK_SIZE)
    print("\nMIN_SCORE | answerable wrongly refused | unanswerable correctly refused")
    best = {g["question"]: index.search(g["question"], 1)[0]["score"] for g in golden}
    for t in (0.01, 0.05, 0.1, 0.3, 0.5):
        wrong = sum(best[g["question"]] < t for g in answerable)
        right = sum(best[g["question"]] < t for g in unanswerable)
        print(f"{t:9} | {wrong}/{len(answerable)} | {right}/{len(unanswerable)}")


def judge(provider):
    if provider == "gemini":
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=config.GEMINI_MODEL, temperature=0, base_url=config.GEMINI_BASE_URL,
            api_key=os.environ["GEMINI_API_KEY"], max_retries=2,
            # one request every 6 seconds, to stay under the per-minute limit
            rate_limiter=InMemoryRateLimiter(requests_per_second=1 / 6, max_bucket_size=1))
    from langchain_groq import ChatGroq
    return ChatGroq(model=config.LLM_MODEL, temperature=0)


async def ragas_eval(pdf, size, k, mode, provider):
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, ResponseRelevancy
    from rag.generate import stream_answer

    index, samples = build(pdf, size), []
    for g in [g for g in load_golden() if g["pages"]]:
        hits = index.search(g["question"], k, mode)
        answer = "".join([t async for t in stream_answer(g["question"], hits, provider)])
        print(f"Answered {len(samples) + 1} question(s)")
        samples.append({"user_input": g["question"], "response": answer,
                        "retrieved_contexts": [h["text"] for h in hits],
                        "reference": g["reference"]})
        await asyncio.sleep(10)  # stay under free-tier rate limits
    return samples, lambda: evaluate(
        EvaluationDataset.from_list(samples),
        metrics=[Faithfulness(), ResponseRelevancy(strictness=1), LLMContextPrecisionWithReference()],
        llm=LangchainLLMWrapper(judge(provider)),
        embeddings=LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=config.EMBED_MODEL)),
        run_config=RunConfig(max_workers=1, max_retries=10, max_wait=90, timeout=300),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["retrieval", "ragas"])
    p.add_argument("pdf")
    p.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    p.add_argument("--k", type=int, default=config.TOP_K)
    p.add_argument("--mode", default="rerank", choices=["dense", "hybrid", "rerank"])
    p.add_argument("--provider", default="groq", choices=["groq", "gemini"])
    a = p.parse_args()
    if a.step == "retrieval":
        retrieval(a.pdf)
    else:
        samples, run = asyncio.run(ragas_eval(a.pdf, a.size, a.k, a.mode, a.provider))
        result = run()
        model = config.GEMINI_MODEL if a.provider == "gemini" else config.LLM_MODEL
        out = HERE / f"ragas_{a.mode}_{a.size}_k{a.k}_{a.provider}.csv"
        table = result.to_pandas()
        table.to_csv(out, index=False)
        print(f"\nModel: {model}\n{result}")
        inputs = {"user_input", "retrieved_contexts", "response", "reference"}
        for column in table.columns:
            if column not in inputs:
                print(f"{column}: scored {table[column].notna().sum()}/{len(samples)}")
        print(f"Saved {out.name}")
