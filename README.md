# Document Q&A with Source Citations

A prototype that answers questions about PDFs with selectable text, using
Retrieval-Augmented Generation (RAG). It can make mistakes: check the cited pages.

## How it works

    PDF -> text and table extraction -> chunks (file, page, section) -> ChromaDB
    Question -> dense search + BM25 -> Reciprocal Rank Fusion -> cross-encoder rerank
             -> top 5 chunks -> LLM -> answer with numbered citations

- **Chunking:** splits on recognised section headings, keeps detected tables whole, never crosses pages.
- **Retrieval:** `all-MiniLM-L6-v2` embeddings, BM25, `ms-marco-MiniLM-L-6-v2` reranker, optional filter by file.
- **Generation:** `openai/gpt-oss-120b` on Groq. The prompt tells it to refuse when the passages lack the answer.
- **Citations:** file and page come from stored metadata. The LLM chooses the citation numbers, so they can be wrong.
  The sources panel lists all retrieved passages.
- **Optional backup:** with `LLM_FALLBACK=gemini`, the app switches to Gemini if Groq hits a rate limit,
  connection error or server error before replying. It is off by default.
- **App:** async FastAPI with streamed responses, Redis answer cache, Streamlit UI, Docker Compose.

An optional reranker score cutoff (`MIN_SCORE`) is **off by default**. In testing on a short
document, valid and unrelated questions received very low, close scores, and the cutoff
rejected valid questions. Declining now relies on the LLM, which uses more tokens.

## Run
Create `.env` from `.env.example` and add your Groq API key, then:

    docker compose up --build

Open http://localhost:8501, choose a PDF and click **Index it**.

## Evaluation
Test set: "Attention Is All You Need", 9 answerable questions with labelled pages
(`eval/golden.jsonl`). The file also has 4 unanswerable questions, which these
retrieval metrics and the RAGAS runs do not use.

### Retrieval (no LLM)
MRR by chunk size (characters, 15% overlap). Hit rate was 1.0 in all 27 setups,
and top-k (3, 5, 8) did not change either metric.

| Method | 500 | 1000 | 1500 |
|---|---|---|---|
| Dense | 0.889 | 0.944 | 0.944 |
| Hybrid (dense + BM25, RRF) | 0.870 | 0.944 | 0.944 |
| Hybrid + reranking | 0.944 | 1.000 | 0.944 |

Hybrid search did not outperform dense search on this test set. A hit means a labelled
page was retrieved, not that the passage or the final answer was correct.

### RAGAS
In each run in this table, `gemini-3.5-flash-lite` generates the answers and judges them.
The same model is used for all four runs so they can be compared. The app's default model is different.

| Run | Mode | Chunk size | Top-k | Faithfulness | Answer relevancy | Context precision |
|---|---|---|---|---|---|---|
| A | Hybrid | 1000 | 5 | 1.000 | 0.841 | 0.806 |
| B | Dense | 1000 | 5 | 1.000 | 0.883 | 0.837 |
| C | Hybrid | 500 | 5 | 1.000 | 0.809 | 0.827 |
| D | Hybrid | 1000 | 3 | 1.000 | 0.881 | 0.944 |

Each run scored all 9 questions on all three metrics. Each setup has one reported run.

- **Hybrid vs. dense (A, B):** hybrid did not improve on dense here.
- **Chunk size (A, C):** mixed. 1000 had higher answer relevancy, 500 had higher context precision.
- **Top-k (A, D):** top-k 3 scored higher on answer relevancy and context precision. Context precision
  judges the ranking of the retrieved chunks; it does not show whether anything needed was left out.

With only 9 questions, small gaps are not meaningful. The app defaults to top-k 5 and uses reranking,
which these RAGAS runs do not measure.

### Reproduce
    docker compose exec api pip install -r requirements-eval.txt
    docker compose cp paper.pdf api:/app/paper.pdf
    docker compose exec api python -m eval.benchmark retrieval paper.pdf
    docker compose exec api python -m eval.benchmark ragas paper.pdf --mode hybrid --size 1000 --k 5 --provider gemini

The last command needs `GEMINI_API_KEY` in `.env`.

## Limitations
- Small test set on one paper. Settings were chosen using these same questions, and extraction
  and page labels were corrected after early runs. This does not show performance on other documents.
- No OCR for scanned pages, and images and charts are not interpreted.
- Complex layouts and unrecognised headings may be extracted or split badly.
- Only 5 chunks reach the LLM, so summaries and "list all" questions may be incomplete.
- Badly misspelled questions can fail. In one manual test, a heavily misspelled request got no answer, while
  the correctly spelled version did. Spelling is not corrected before search.
- Declining unanswerable questions has had limited testing.
- BM25 is rebuilt on every query, which suits small collections only.
- No login or user separation. Do not expose it publicly or upload private files to a shared instance.
- Questions and retrieved passages are sent to Groq, or to Gemini if the backup or a Gemini benchmark
  is used. Free-tier rate limits apply. Google may use content sent on Gemini's free tier to improve its
  products, and people may review it, so do not use it with private documents.


