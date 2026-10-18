# Semantic Search Engine

A generic, local semantic search pipeline over PDFs, meant as a starting point to build a
search engine for your own corpus or to experiment with one. Ask a question in plain language
and get back the most relevant documents and paragraphs, cited by section and PDF page.
Everything runs on your machine; only `make get` touches the network, to download the models.

```
PDFs ─▶ ingest ─▶ parse (docling → markdown) ─▶ index (chunk + embed) ─▶ search (recall + rerank)
```

## Quick start

Needs Python ≥3.12 and [uv](https://docs.astral.sh/uv/). No documents ship with the repo, so
the quickest test is one PDF you already have: a paper, a report, a manual. A born-digital PDF
(text you can select) works best; scanned ones need `OCR=1` on `make parse`.

```sh
git clone https://github.com/Manteplante/semantic_search_engine.git
cd semantic_search_engine
make sync                         # 1. install dependencies
make get                          # 2. download the models in model.yaml (once, a few GB)

mkdir -p data/example             # 3. a collection is a folder under data/
cp ~/Downloads/some-report.pdf data/example/

make ingest                       # 4. register data/<collection>/*.pdf in the database
make parse                        # 5. PDF -> markdown in parsing/output/example/
make index                        # 6. chunk and embed into search/output/
make search Q="a question the PDF answers"   # 7. best documents with their best passages
make search-paragraph Q="..."     #    whole paragraphs, with section and PDF page
```

Things that trip up a first run:

- **The PDF must sit in a collection folder**, `data/<collection>/<file>.pdf`. A PDF dropped
  straight into `data/` is silently ignored: `make ingest` doesn't count it.
- **The file name becomes the document id**: letters, digits, `.`, `_` and `-` only, so rename
  `My Report (final).pdf` to `my-report-final.pdf`. `make ingest` prints `[SKIP]` otherwise.
- **Ask something the PDF actually covers.** A question with no good match returns nothing
  rather than filler (see [the reranker](#models-why-an-embedder-and-a-reranker)).

To add more documents, copy them in and re-run steps 4–6: only new or changed PDFs are parsed
and embedded again. `make help` lists every target, `make hardware` shows whether the models
will run on the GPU.

## Customize ingestion first

Ingestion is the part to make your own. Out of the box, `make ingest` registers PDFs you copy
into `data/<collection>/` by hand, which is enough to test the pipeline. For a real corpus,
write a source that fetches documents (an API, a website, a bucket) and stores them the same
way: one row per document in the database, its PDF at `data/<collection>/<doc_id>.pdf`.
[ingestion/fetch.py](ingestion/fetch.py) has the HTTP plumbing (retries, atomic hashed
downloads), and [SKILLS.md](SKILLS.md) has the recipe. Parsing, indexing and search need no
changes. A source that also stores titles, authors and years gets better ranking and citations.

## Models: why an embedder and a reranker

Both are defined in [model.yaml](model.yaml) and can be swapped there:

- **Embedder:** [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
  turns every chunk and the question into vectors. Comparing vectors is fast enough to scan the
  whole corpus, but it only measures similarity: it always returns *something*, even when
  nothing answers the question.
- **Reranker:** [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) reads
  the question and each candidate passage *together* and scores relevance from 0 to 1. That is
  far more accurate but too slow to run on everything.

So search runs in two stages: the embedder recalls the best ~50 candidates, the reranker
re-orders them and drops anything under `min_relevance`, so a query with no good match says so
instead of listing filler.

## Configuration

- [search.yaml](search.yaml): chunk size, sections and noise lines to skip, results per query,
  relevance cutoff. `make inspect T=<name>` prints exactly what the embedder reads.
- [.env.example](.env.example): copy to `.env` for `HF_TOKEN` (gated models), `DATABASE_URL`
  (SQLite by default, Postgres works too) or `DEVICE`.

## Layout

| Path | What |
|---|---|
| `ingestion/` | database, local import, HTTP helpers, CSV export |
| `parsing/` | PDF → markdown with docling, one worker process per PDF |
| `search/` | config, models, chunking, embedding, ranking, CLI |
| `app/` | backend stub |
| `data/`, `parsing/output/`, `search/output/`, `models_local/` | generated, gitignored |

Development: `make ci` runs lint, format-check and tests, the same as GitHub Actions.
