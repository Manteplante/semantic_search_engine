.DEFAULT_GOAL := help
.PHONY: sync lint format format-check test smoke hooks ci ingest export parse hardware get index inspect search search-paragraph help

##@ Setup

sync: ## Install/sync all dependencies (incl. dev group)
	uv sync --all-groups

hooks: ## Install the pre-commit git hook
	uv run pre-commit install

##@ Lint & Format

lint: ## Run ruff lint checks
	uv run ruff check .

format: ## Auto-format code with ruff
	uv run ruff format .

format-check: ## Check formatting without modifying files
	uv run ruff format --check .

##@ Test

test: ## Run the test suite
	uv run pytest

smoke: test ## Alias for test (smoke-test suite)

##@ CI

ci: lint format-check test ## Run the full local CI gate (same as GitHub Actions)

##@ Ingestion

FORMAT ?= csv

ingest: ## Register the PDFs in data/<collection>/ in the document database (new or changed only)
	uv run python -m ingestion.local_import

export: ## Write the document database to data/documents.csv (FORMAT=parquet needs pyarrow)
	EXPORT_FORMAT=$(FORMAT) uv run python -m ingestion.export

##@ Parsing

LIMIT ?= 20
FORCE ?= 0
OCR ?= 0
TIMEOUT ?= 300

parse: ## New or changed PDFs to markdown in parsing/output/: newest LIMIT per collection (0 = all). FORCE=1 redoes all, OCR=1 for scans, TIMEOUT=s per PDF
	PARSE_LIMIT=$(LIMIT) PARSE_FORCE=$(FORCE) PARSE_OCR=$(OCR) PARSE_TIMEOUT=$(TIMEOUT) uv run python -m parsing.pdf_parser

##@ Search

SEARCH = uv run python -m search
TOP_K_ARG = $(if $(TOP_K),--top-k $(TOP_K))

hardware: ## Show the GPU/CPU and which device the models will run on
	$(SEARCH) hardware

get: ## Download the embedder in model.yaml to models_local/ (the only networked step)
	$(SEARCH) get

index: ## Chunk the parsed documents and build the search index (only changed ones are re-embedded)
	$(SEARCH) index

inspect: ## Print exactly what the embedder reads for a document: make inspect T="part of a name"
	@test -n "$(T)" || { echo 'usage: make inspect T="part of an author, title or doc_id"'; exit 2; }
	$(SEARCH) inspect "$(T)"

search: ## Semantic search over the documents: make search Q="your question" [TOP_K=n]
	@test -n "$(Q)" || { echo 'usage: make search Q="your question"'; exit 2; }
	$(SEARCH) search "$(Q)" $(TOP_K_ARG)

search-paragraph: ## Whole paragraphs answering a question, with PDF page: make search-paragraph Q="..." [TOP_K=n]
	@test -n "$(Q)" || { echo 'usage: make search-paragraph Q="your question"'; exit 2; }
	$(SEARCH) paragraph "$(Q)" $(TOP_K_ARG)

##@ Help

help: ## Show this list of commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
