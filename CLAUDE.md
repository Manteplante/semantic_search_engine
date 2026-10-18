# CLAUDE.md

Guidance for AI agents working in this repo. `README.md` is the human-facing source of truth;
`SKILLS.md` has step-by-step recipes for common tasks — read the relevant recipe before starting.

## Project

A local semantic search engine over a corpus of PDFs: register documents, convert them to
markdown with docling, chunk them section by section, embed the chunks with a Hugging Face
model, and answer questions with cosine recall + a cross-encoder reranker, citing document,
section and PDF page.

Pipeline, one package per stage:

| Stage | Path | Status |
|---|---|---|
| Ingestion | `ingestion/` — `db.py` (SQLAlchemy Core: `documents`, `chunks`, `paragraphs` tables), `local_import.py` (`make ingest`: registers `data/<collection>/*.pdf`, `safe_id()`), `fetch.py` (HTTP plumbing for networked sources: retrying session, `get_json`, atomic hashed `download_to`), `export.py` (`make export`) | working |
| Data | `data/documents.sqlite` (the database, or `DATABASE_URL`) + `data/<collection>/<doc_id>.pdf` | local, gitignored |
| Parsing | `parsing/pdf_parser.py` (`make parse`: work list from the database, newest `LIMIT` per collection, 0 = all; only PDFs changed since their markdown) → `parsing/output/<collection>/<doc_id>.md`, each PDF page led by `<!-- page N -->` (`PAGE_MARKER`); `docling_prototype.ipynb` for exploration | working |
| Search | `search/` — `config.py` (paths, `.env`, offline switch, device, `ConfigError`), `model.py` (reads `model.yaml`, download/load), `options.py` (reads `search.yaml`), `corpus.py` (sections → paragraphs with PDF pages → cleaned chunks + database metadata), `search.py` (embedded text, embeddings, per-document vector cache, cosine top-k, reranking + relevance cutoff, group by document, citation), `hardware.py`, `__main__.py` (CLI) → `search/output/` (`embeddings.npy`, `vectors/`) + the `chunks`/`paragraphs` tables | working, needs `make get` |
| App | `app/backend.py` | stub |

## Commands

Python ≥3.12, managed with `uv` (never `pip install`; add deps with `uv add <pkg>` or
`uv add --dev <pkg>`). `make help` lists all targets.

```sh
uv sync --all-groups   # install (same as make sync)
make ci                # lint + format-check + test — exactly what GitHub Actions runs
make lint / make format / make test
make ingest                     # register data/<collection>/*.pdf in the database
make export [FORMAT=parquet]    # data/documents.csv
make parse / hardware / get / index / search Q="..." [TOP_K=n]
make search-paragraph Q="..." [TOP_K=n]   # whole paragraphs with document, section, PDF page
make inspect T=<name>           # exactly what the embedder reads for matching documents
```

## Working style

- State assumptions before editing.
- If the task is ambiguous, ask one clarifying question.
- Touch only files needed for the requested change. Do not refactor unrelated code unless asked.
- Always use plan mode for complex issues.
- Match existing style: ruff, line length 100, rules `E,F,I,UP,B`. Plain functions and
  module-level constants; `[TAG]`-prefixed `print` for progress logging (no logging framework yet).

## Tests, CI/CD and docs — keep them in step with the code

Every change must leave `make ci` green. Tests and CI are part of the task, not a follow-up:

- **Tests** (`tests/test_smoke.py`, `tests/test_search.py`, `tests/test_ingestion.py`): new module → add its import + an `assert` in
  `test_layers_import`. New/changed behaviour (constants, env vars, folder mapping, parsing
  rules) → add or update a focused test. New generated file type → add it to the relevant
  `.gitignore` and extend the matching `*_ignores_generated_files` test.
- **Tests must not hit the network or write to `data/`.** Test pure functions and constants;
  monkeypatch `requests` or env vars if needed. Database tests use `db.engine("sqlite:///<tmp_path>/...")`;
  `tests/test_ingestion.py` also points `DATABASE_URL` at `tmp_path` for anything that falls back to it.
- **CI** (`.github/workflows/ci.yml`) mirrors `make ci`. If you add a check (new tool, new test
  path, notebook execution, new dependency group), add it to the `Makefile` `ci` target **and**
  the workflow so they stay identical. Also keep `.pre-commit-config.yaml` ruff `rev` in sync with
  the ruff version in `uv.lock`.
- **Docs**: if you change commands, layout or env vars, update `README.md` (and this file /
  `SKILLS.md` if agent guidance changes).
- Notebooks are committed **without outputs** (a test enforces this). Clear outputs before staging.
- Before finishing, run `make ci` and report the result. Running `make parse`, `make index` or
  the notebook end-to-end is the user's step unless they ask — it is slow and loads models.

## Gotchas

- Models are data: never hard-code a model id in Python. They live in `model.yaml`, one per
  role (`embedder`, `engine: embed`; `reranker`, `engine: rerank` — the engine is fixed per
  role); how they are applied (chunking, skipped sections, top_k, rerank cutoff) lives in
  `search.yaml`. New keys must be added to
  the `FIELDS` sets in `model.py` / `options.py` or they are rejected.
- `search.config` sets `HF_HUB_OFFLINE=1` on import; only `model.download()` (`make get`)
  may use the network. Tests must never download a model — fake the tokenizer/model.
- The package `search` contains a module `search.py`, so inside tests `search` is the module
  (`from search import search`); `search.search.top_k` from outside.
- **Database rules** (`ingestion/db.py`, the only module that talks to it): SQLAlchemy Core,
  every value a bound parameter, never SQL from f-strings/`%`/`.format` (a test greps for it);
  shape-changing input (stage, column) goes through a fixed dict or `checked()`. Keep types
  portable (SQLite now, Postgres via `DATABASE_URL` later); upserts go through `UPSERTS`. Print
  the URL only via `shown_url()`.
- The `doc_id` is the key everywhere: PDF and markdown are named `<doc_id>`, chunks carry it as
  `document`. `local_import.safe_id()` validates it (and the collection name) before either
  becomes a file name.
- A networked source goes through `ingestion/fetch.py` (retries, timeouts, `.part` + rename,
  sha256) and writes the same rows/files `make ingest` does, so parse/index need no change.
- Docling runs with **OCR off** (`PARSE_OCR=1` to enable) and a per-PDF `PARSE_TIMEOUT`: with OCR
  on, one born-digital PDF hung >10 min and a full `make parse` froze the user's WSL. Only a
  `success` status is saved; anything else becomes `parse_error`. Each PDF is converted in its
  own spawned process (`run_isolated`): docling leaks memory across documents. Don't move the
  conversion back into the main process.

## Communication

- Be concise.
- Call out trade-offs when multiple reasonable approaches exist.
- Push back if a safer or smaller approach would meet the goal.
- Be critical of suggestions posed by the user rather than agreeable.

## Git

Never write history. Do not run `git commit`, `git push`, `git merge`, `git rebase`,
`git reset --hard`, or `gh pr create`. Staging is fine: `git add`, `git status`, `git diff`.
When work is done, stage it and write the proposed commit message in the reply for the user to run
themselves. Commit messages use conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`, `ci:`).
Branch flow: feature branch → `dev` → `main` (`main` is protected).
