# SKILLS.md

Step-by-step recipes for common tasks in this repo. Each ends with the same gate: **`make ci` is
green, and tests/CI/README are updated to match** (see "Tests, CI/CD and docs" in `CLAUDE.md`).

## Quick orientation

```sh
make help                          # all targets
git ls-files | grep -v '^data/'    # tracked source (data/ is generated)
uv run pytest -q                   # fast test run
uv run ruff check . --fix && uv run ruff format .   # auto-fix lint + format
```

Key symbols:

- `ingestion/db.py` — tables `documents`, `chunks`, `paragraphs`; `engine()`, `upsert()`,
  `update()`, `get()`, `pending(stage)` (`STAGES`), `all_documents()`, `replace_index()`,
  `chunk_rows()`, `paragraph_rows()`, `shown_url()`.
- `ingestion/local_import.py` — `safe_id()`, `pdf_path()`, `file_sha256()`, `find_pdfs()`,
  `register()`, `main()` (`make ingest`).
- `ingestion/fetch.py` — `SESSION` (retries), `RETRY`, `get_json()`, `download_to()`;
  constants `TIMEOUT`, `DOWNLOAD_TIMEOUT`.
- `parsing/pdf_parser.py` — `targets()`, `needs_parse()`, `paged_markdown()`,
  `run_isolated()`, `PAGE_MARKER`; env `PARSE_LIMIT`, `PARSE_FORCE`, `PARSE_OCR`, `PARSE_TIMEOUT`.

---

## Add documents

1. Put PDFs in `data/<collection>/<doc_id>.pdf`. `collection` and `doc_id` must be safe file
   names (letters, digits, `.`, `_`, `-`); `make ingest` skips anything else and says so.
2. `make ingest`, then `make parse` (LIMIT=0 for all) and `make index`.
3. Titles, authors and years come from the `documents` table: a source that knows them sets
   them with `db.upsert()`/`db.update()`; they are embedded above every chunk and used in
   citations, so they improve both ranking and output.

## Add a networked source (API, website)

1. New module under `ingestion/` (e.g. `ingestion/<source>_api.py`) for the source's
   endpoints, built on `fetch.SESSION` / `fetch.get_json()` / `fetch.download_to()` — keep
   network calls there. Probe the live API manually first
   (`curl -s '<url>' | python -m json.tool | head`).
2. A second module maps the source's records to `documents` rows (`doc_id` through
   `local_import.safe_id()`, a `collection`, metadata, `source_url`, `source_modified`,
   `pdf_sha256` from `download_to()`), saves each PDF at `local_import.pdf_path()`, and
   skips records whose `source_modified` and file are unchanged. Reset `parse_error` to
   `None` when a new file is downloaded.
3. An entry point with `get_<thing>()` helpers reading `<SOURCE>_<THING>` env vars with
   defaults (an invalid value raises `ConfigError`, printed by `main()` as one line), and a
   `Makefile` target passing `Makefile` variables through.
4. Tests in `tests/test_ingestion.py`: monkeypatch `requests.Session.get` or the API module's
   functions — never real HTTP. Test the built params, the record mapping, and the
   skip/re-download logic. Follow "Add a new module or pipeline stage".

## Add a CLI/env option to a stage

1. Add a `get_<thing>()` helper in the stage's entry module reading `<STAGE>_<THING>` with a
   default (see `get_timeout()` in `parsing/pdf_parser.py`).
2. Add a `Makefile` variable (`THING ?= default`) and pass it through in the target.
3. Add two tests: default value and env override (copy the `PARSE_OCR`/`PARSE_TIMEOUT` pair).
   An invalid value raises `ConfigError` (from `search.config`).
4. Document it in README and the target's `##` help text.

## Change the database schema or queries

1. All of it lives in `ingestion/db.py`. New column → add it to the `Table`, with a portable
   type (`String`/`Text`/`Integer`/`BigInteger`/`JSON`). `create_all` does not alter an
   existing table, so tell the user to delete `data/documents.sqlite` and re-run
   `make ingest` (or add an Alembic migration once data is hosted).
2. Values only as bound parameters through SQLAlchemy Core; never `text(f"...")`, `%` or
   `.format`. A query whose shape depends on input looks it up in a fixed dict (see `STAGES`)
   or runs it through `checked()`.
3. Test on a `tmp_path` SQLite URL, including a hostile value
   (`test_untrusted_text_is_stored_as_data_not_sql`). If the upsert changes, keep
   `test_postgres_upsert_binds_every_value` passing.

## Add a new module or pipeline stage

1. Create the file under the right package (`ingestion/`, `parsing/`, `search/`, `app/`);
   add `__init__.py` if the package is new.
2. Add `import <pkg>.<module>` and an `assert` in `test_layers_import`.
3. If it writes generated files, put them in a gitignored folder, add the patterns to that
   folder's `.gitignore`, and add a `*_ignores_generated_files` test.
4. Update the Layout section of `README.md` and the pipeline table in `CLAUDE.md`.

## Add a dependency

1. `uv add <pkg>` (runtime) or `uv add --dev <pkg>` (tooling). Never hand-edit versions in
   `uv.lock`.
2. CI installs with `uv sync --all-groups`, so a new dependency group is picked up automatically.
   A new tool that must run in CI needs a `Makefile` target, a line in `make ci`, **and** a step
   in `.github/workflows/ci.yml`.
3. If you bump ruff, also bump `rev` in `.pre-commit-config.yaml`.

## Work on the parsing notebook

1. Kernel: `.venv` (installed by `uv sync`). Docling downloads models on the first run (slow once).
2. Headless run if needed (uses `nbclient`; `jupyter nbconvert` is not installed) — executes in
   memory and does not touch the committed file:
   `uv run python -c "import nbformat; from nbclient import NotebookClient; nb = nbformat.read('parsing/docling_prototype.ipynb', 4); NotebookClient(nb, kernel_name='python3').execute()"`
3. Before staging, clear outputs — `test_parsing_notebook_is_valid_and_unexecuted` enforces it:
   `uv run python -c "import json; p = 'parsing/docling_prototype.ipynb'; nb = json.load(open(p)); [c.update(outputs=[], execution_count=None) for c in nb['cells'] if c['cell_type'] == 'code']; open(p, 'w').write(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')"`
   (plain `json` on purpose — round-tripping through `nbformat` adds cell ids and a noisy diff).
4. When logic stabilises, move it to a `parsing/*.py` module and follow "Add a new module".

## Swap a Hugging Face model

1. Edit `model.yaml` only — no model id belongs in Python. The embedder uses `engine: embed`;
   set `max_length`, `pooling` (`mean` for BERT/e5-style encoders, `last` for decoder
   embedders such as Qwen3-Embedding), and `query_prefix` / `passage_prefix` to what the model
   card says (empty if none). The reranker uses `engine: rerank` and must load with
   `AutoModelForSequenceClassification` and output one relevance logit (bge-reranker style).
   The engine is fixed per role; a mismatch is a config error.
2. Check the limit still holds: `corpus.chunk_tokens` + 16 ≤ `embedder.max_length`
   (`search/options.py` enforces it; `make ci` runs the repo files through it).
3. User runs `make get`, then `make index` (the index refuses a different embedder; a new
   reranker needs no re-index, but re-check `search.min_relevance` against a few known
   on-topic and off-topic queries — its scores are model-specific). Don't download models
   yourself unless asked — they can be GBs.
4. Update the model name in README's Semantic search section.

## Tune chunking, skipped sections or search

1. Edit `search.yaml` (`corpus:` chunking, skipped headings, `drop_lines`,
   `min_section_tokens`; `search:` `top_k` documents, `passages_per_document`, `batch_size`).
   Corpus changes need `make index` to take effect.
2. A new setting → add it to `FIELDS` and the `Options` dataclass in `search/options.py`,
   plus a test in `tests/test_search.py`.
3. To see what a change does, run `make inspect T=<part of a file name>`: it prints every
   string exactly as embedded and each dropped section with its reason. It loads only the
   tokenizer — no weights, no network.

## Add a search command

1. Add `run_<name>(args)` and a subparser in `search/__main__.py`; register it in
   `COMMANDS`. Raise `ConfigError` for user mistakes — the CLI prints it as one line.
2. Add a Makefile target under `##@ Search` (and to `.PHONY`).
3. Keep the logic in a module with the model call injected (`count_tokens`, vectors) so it is
   testable without weights; test it in `tests/test_search.py`.

## Change CI/CD

1. Edit the `Makefile` `ci` target and `.github/workflows/ci.yml` together — they must run the
   same checks in the same order.
2. Validate YAML locally: `uv run pre-commit run check-yaml --all-files`.
3. Run `make ci`. Mention in the reply that the workflow change is only proven once pushed.

## Finish a task

1. `make ci` — fix anything red (`make format` for formatting).
2. `git status` / `git diff` — only intended files changed; no PDFs, CSVs, notebook outputs.
3. `git add` the changes and propose a conventional commit message (`feat:`, `fix:`, `test:`,
   `docs:`, `ci:`). Do not commit — see Git in `CLAUDE.md`.
