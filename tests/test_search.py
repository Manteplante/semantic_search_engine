"""The search stage without a model: config, chunking and ranking logic.

Nothing here downloads weights or touches the network - the model is replaced by
word counting (for tokens) and hand-made vectors (for search).
"""

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from search import config, corpus, hardware, model, options, search
from search.config import ConfigError


def count_words(text):
    return len(text.split())


@pytest.fixture(autouse=True)
def fresh_settings(monkeypatch):
    for name in ("EMBEDDER_ID", "EMBEDDER_ENGINE", "RERANKER_ID", "RERANKER_ENGINE"):
        monkeypatch.delenv(name, raising=False)
    config.reset()
    yield
    config.reset()


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def repo_options():
    return options.options(model.spec("embedder"))


# config


def test_torch_jit_off_only_without_a_compiler(monkeypatch):
    monkeypatch.delenv("TORCH_DISABLE_NATIVE_JIT", raising=False)
    monkeypatch.setattr(config.shutil, "which", lambda name: "/usr/bin/gcc")
    config.without_compiler_skip_torch_jit()
    assert "TORCH_DISABLE_NATIVE_JIT" not in os.environ
    monkeypatch.setattr(config.shutil, "which", lambda name: None)
    config.without_compiler_skip_torch_jit()
    assert os.environ["TORCH_DISABLE_NATIVE_JIT"] == "1"


def test_importing_config_goes_offline():
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_clean_strips_trailing_comment():
    assert config.clean("models  # where weights go") == "models"
    assert config.clean(5) == 5


def test_load_env_quoting_and_shell_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("ALREADY_SET", "shell")
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("PLAIN", raising=False)
    env_file = write(
        tmp_path / ".env", 'QUOTED="a # b"\nexport PLAIN=value  # note\nALREADY_SET=file\n'
    )
    loaded = config.load_env(env_file)
    assert loaded == {"QUOTED": "a # b", "PLAIN": "value"}
    assert os.environ["ALREADY_SET"] == "shell"


def test_settings_honours_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "weights"))
    config.reset()
    found = config.settings()
    assert found.models_dir == tmp_path / "weights"
    assert found.local_path("org/name") == tmp_path / "weights" / "org__name"
    assert found.input_dir.parts[-2:] == ("parsing", "output")


def test_read_yaml_rejects_unknown_keys(tmp_path):
    path = write(tmp_path / "x.yaml", "good: 1\ntypo: 2\n")
    with pytest.raises(ConfigError, match="unknown thing key 'typo'.*Valid keys: good"):
        config.read_yaml(path, {"good"}, "thing")


# model.yaml


def test_repo_model_yaml_has_embedder_and_reranker():
    found = model.specs()
    assert list(found) == ["embedder", "reranker"]
    assert found["embedder"].engine == "embed"
    assert found["embedder"].pooling in model.POOLINGS
    assert found["embedder"].query_prefix.endswith("\nQuery:")  # newline survives YAML
    assert found["reranker"].engine == "rerank"


@pytest.mark.parametrize("role", ["embedder", "reranker"])
def test_env_overrides_model_id(monkeypatch, role):
    monkeypatch.setenv(f"{role.upper()}_ID", "tiny/model")
    assert model.spec(role).id == "tiny/model"


def test_role_needs_its_own_engine():
    with pytest.raises(ConfigError, match="the reranker needs `engine: rerank`"):
        model.parse_block("reranker", {"id": "a/b", "engine": "embed"})
    with pytest.raises(ConfigError, match="the embedder needs `engine: embed`"):
        model.parse_block("embedder", {"id": "a/b", "engine": "rerank"})


@pytest.mark.parametrize(
    ("block", "message"),
    [
        ({"id": "a/b", "engine": "embed", "colour": "red"}, "unknown embedder key 'colour'"),
        ({"id": "a/b", "engine": "chat"}, "engine 'chat' is not one of"),
        ({"engine": "embed"}, "`embedder.id` is missing"),
        ({"id": "a/b", "engine": "embed", "pooling": "max"}, "pooling 'max' is not one of"),
        ("a/b", "needs a `embedder:` block"),
    ],
)
def test_bad_model_blocks_are_config_errors(block, message):
    with pytest.raises(ConfigError, match=message):
        model.parse_block("embedder", block)


def test_pooling_defaults_to_mean():
    assert model.parse_block("embedder", {"id": "a/b", "engine": "embed"}).pooling == "mean"


def test_last_token_pooling_finds_the_last_real_token_on_either_side():
    import torch

    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).view(2, 3, 2)
    right = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert search.last_token(hidden, right).tolist() == [
        hidden[0, 1].tolist(),
        hidden[1, 2].tolist(),
    ]
    left = torch.tensor([[0, 1, 1], [1, 1, 1]])
    assert search.last_token(hidden, left).tolist() == hidden[:, -1].tolist()


def test_missing_role_is_config_error(tmp_path):
    path = write(tmp_path / "model.yaml", "other: {}\n")
    with pytest.raises(ConfigError, match="unknown role key 'other'"):
        model.specs(path)
    path = write(tmp_path / "model.yaml", "")
    with pytest.raises(ConfigError, match="missing the `embedder:` block"):
        model.specs(path)
    path = write(tmp_path / "model.yaml", "embedder:\n  id: a/b\n  engine: embed\n")
    with pytest.raises(ConfigError, match="missing the `reranker:` block"):
        model.specs(path)


def test_access_advice_only_for_refusals():
    assert "HF_TOKEN" in model.access_advice("a/b", Exception("401 Client Error"))
    assert model.access_advice("a/b", Exception("connection reset")) is None


def test_missing_weights_say_make_get(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    config.reset()
    with pytest.raises(ConfigError, match="make get"):
        model.spec("embedder").require()


# search.yaml


def test_repo_search_yaml_is_valid():
    found = repo_options()
    assert found.chunk_tokens < model.spec("embedder").max_length
    assert "contents" in {name.lower() for name in found.skip_sections}


def test_chunk_too_big_for_embedder(tmp_path):
    source = Path("search.yaml").read_text()
    path = write(tmp_path / "s.yaml", source.replace("chunk_tokens: 400", "chunk_tokens: 600"))
    with pytest.raises(ConfigError, match="does not fit the embedder"):
        options.options(model.spec("embedder"), path)


def test_repo_drop_lines_remove_noise_and_keep_prose():
    drop = repo_options().drop_lines
    noise = [
        "Image 17. The harbour at dawn. Photo by the author, April 2023.",
        "Figure 3: Sample frequencies",
        "[Image 3. Google Maps, 2024. Available at: https://www.google.com/maps]",
        "Table 2 Survey responses by region",
        "| Region | A | 67.3 ± 4.4 |",
        "12",
    ]
    prose = [
        "The image of the harbour as a ruin is misleading.",
        "Figures from the census show a steady decline.",
        "In 2010 the team visited the site twice.",
    ]
    assert all(corpus.is_noise(line, drop) for line in noise)
    assert not any(corpus.is_noise(line, drop) for line in prose)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("min_section_tokens: 30", "min_section_tokens: -1", "min_section_tokens"),
        ("passages_per_document: 2", "passages_per_document: 0", "passages_per_document"),
        ("rerank_candidates: 50", "rerank_candidates: -1", "rerank_candidates"),
        ("min_paragraph_tokens: 20", "min_paragraph_tokens: -1", "min_paragraph_tokens"),
        ("paragraphs: 5", "paragraphs: 0", "search.paragraphs"),
        ("min_relevance: 0.", "min_relevance: 1.", "min_relevance"),
        ("  drop_lines:\n", "  drop_lines:\n    - '(unclosed'\n", "not a valid regex"),
    ],
)
def test_bad_corpus_and_search_values_are_config_errors(tmp_path, old, new, message):
    source = Path("search.yaml").read_text()
    assert old in source
    path = write(tmp_path / "s.yaml", source.replace(old, new))
    with pytest.raises(ConfigError, match=message):
        options.options(model.spec("embedder"), path)


def test_removed_summarize_section_is_rejected(tmp_path):
    path = write(tmp_path / "s.yaml", "summarize:\n  map_tokens: 8000\n")
    with pytest.raises(ConfigError, match="unknown section key 'summarize'"):
        options.options(model.spec("embedder"), path)


# hardware


def test_hardware_advice():
    assert hardware.advice(True, "13.0", True) == "using the GPU - nothing to do"
    assert "CPU build" in hardware.advice(True, None, False)
    assert "check the driver" in hardware.advice(True, "13.0", False)
    assert "no GPU" in hardware.advice(False, None, False)


def test_hardware_turns_gpu_errors_into_advice():
    assert "DEVICE=cpu" in hardware.as_advice(RuntimeError("CUDA out of memory"))
    assert "C compiler" in hardware.as_advice(RuntimeError("Failed to find C compiler"))
    assert hardware.as_advice(RuntimeError("something else")) is None


# corpus

DOCUMENT_MD = """\
## Caring for the sea

A report on coastal communities

## Contents

| Method ........................ 12 |

## 1. Introduction

Volunteers on the coast. They clean beaches &amp; harbours.

<!-- image -->

## Literature and chapter overview

Chapter two is about method.

## References

Smith, J. 2020. A study of coastal life.
"""


def test_split_sections_skips_front_and_back_matter():
    sections = corpus.split_sections(DOCUMENT_MD, ["Contents", "References", "Literature"])
    headings = [heading for heading, _ in sections]
    assert headings == ["Caring for the sea", "1. Introduction", "Literature and chapter overview"]
    text = dict(sections)["1. Introduction"]
    assert "&" in text and "&amp;" not in text
    assert "<!-- image -->" not in text


def test_skip_ignores_case_and_chapter_numbers():
    sections = corpus.split_sections("## 7. REFERENCES\n\nSmith 2020.\n", ["References"])
    assert sections == []


def test_text_before_first_heading_is_kept():
    assert corpus.split_sections("Cover\n\n## A\n\nb\n") == [("", "Cover"), ("A", "b")]


def test_control_characters_are_stripped():
    assert corpus.split_sections("## A\x00\n\nTe\x01xt\x13.\n") == [("A", "Text.")]


def test_short_sections_are_dropped_with_a_reason():
    markdown = "## Author\n\nJane\n\n## Summary\n\nA long enough text about the topic.\n"
    sections = corpus.sort_sections(markdown, min_tokens=3, count_tokens=count_words)
    assert [(section.heading, section.reason) for section in sections] == [
        ("Author", corpus.TOO_SHORT),
        ("Summary", None),
    ]
    assert corpus.split_sections(markdown, min_tokens=3) == [
        ("Summary", "A long enough text about the topic.")
    ]


def test_skipped_heading_reason_wins_over_length():
    [section] = corpus.sort_sections("## Thanks\n\nThanks.\n", ["Thanks"], min_tokens=5)
    assert section.reason == corpus.SKIPPED


PAGED_MD = """\
<!-- page 3 -->

## Introduction

The survey took place at the harbour. It lasted

<!-- page 4 -->

six months in the autumn of 2023.

This one stands alone.

<!-- page 5 -->

Footnote about method

which continues here.
"""


def test_paragraphs_carry_pdf_pages_and_rejoin_page_break_cuts():
    [section] = corpus.sort_sections(PAGED_MD)
    assert section.heading == "Introduction"
    assert [(p.text, p.page, p.last_page) for p in section.paragraphs] == [
        ("The survey took place at the harbour. It lasted six months in the autumn of 2023.", 3, 4),
        ("This one stands alone.", 4, 4),
        ("Footnote about method which continues here.", 5, 5),
    ]
    assert "<!--" not in section.text


def test_blocks_are_not_joined_after_a_full_stop_or_before_a_capital():
    text = "First paragraph ends here.\n\nsecond starts small.\n\nNo full stop\n\nNext capital."
    [section] = corpus.sort_sections(text)
    assert [p.text for p in section.paragraphs] == [
        "First paragraph ends here.",
        "second starts small.",
        "No full stop",
        "Next capital.",
    ]


def test_continuation_skips_footnote_scraps_to_the_body_paragraph():
    markdown = (
        "The report argues that the harbour was for decades a\n\n"
        "Smith, Rules\n\n, 17-18\n\n<!-- page 23 -->\n\n"
        "quiet place of work.\n"
    )
    [section] = corpus.sort_sections(markdown)
    assert [(p.text, p.page, p.last_page) for p in section.paragraphs] == [
        ("The report argues that the harbour was for decades a quiet place of work.", None, 23),
        ("Smith, Rules", None, None),
        (", 17-18", None, None),
    ]


def test_markdown_without_page_markers_has_no_pages():
    [section] = corpus.sort_sections("## A\n\nText here.\n")
    assert section.paragraphs == (corpus.Paragraph("Text here.", None, None),)


def test_page_regex_matches_the_parser_marker():
    from parsing.pdf_parser import PAGE_MARKER

    assert corpus.PAGE.match(PAGE_MARKER.format(12)).group(1) == "12"


def test_chunk_paragraphs_remembers_which_paragraphs_each_chunk_covers():
    texts = ["One two three. Four five six.", "Seven eight nine. Ten eleven twelve.", "Thirteen."]
    chunks = corpus.chunk_paragraphs(texts, count_words, size=6, overlap=0)
    assert chunks == [
        ("One two three. Four five six.", (0,)),
        ("Seven eight nine. Ten eleven twelve.", (1,)),
        ("Thirteen.", (2,)),
    ]
    overlapping = corpus.chunk_paragraphs(texts, count_words, size=9, overlap=3)
    assert overlapping == [  # the overlap sentence links both chunks to paragraph 1
        ("One two three. Four five six. Seven eight nine.", (0, 1)),
        ("Seven eight nine. Ten eleven twelve. Thirteen.", (1, 2)),
    ]


def test_chunk_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {n} is here." for n in range(20))  # 5 words each
    chunks = corpus.chunk(text, count_words, size=12, overlap=5)
    assert all(count_words(piece) <= 12 for piece in chunks)
    assert "Sentence number 19 is here." in chunks[-1]
    # the last sentence of one chunk opens the next
    assert chunks[1].startswith(chunks[0].split(". ")[-1].rstrip("."))


def test_chunk_splits_a_sentence_longer_than_size():
    chunks = corpus.chunk(" ".join(["word"] * 25), count_words, size=10)
    assert [count_words(piece) for piece in chunks] == [10, 10, 5]


def test_metadata_comes_from_database_rows_keyed_by_doc_id():
    rows = [{"doc_id": "report-2020", "title": "Coast", "authors": ["Jane", "John"], "year": 2020}]
    found = corpus.load_metadata(rows)
    assert found == {"report-2020": {"title": "Coast", "author": "Jane; John", "year": 2020}}
    assert corpus.document_metadata({"title": None, "authors": [], "year": None})["author"] is None


def test_document_rows_carry_metadata(tmp_path):
    path = write(tmp_path / "example" / "Jane_Doe_2022.md", DOCUMENT_MD)
    [document] = corpus.find_documents(tmp_path)
    assert (document.collection, document.stem) == ("example", path.stem)
    metadata = {"title": "Caring for\n the sea", "author": "Jane Doe", "year": "2022"}
    rows = corpus.document_rows(document, metadata, count_words, 50, 5, ["Contents"])
    assert rows[0]["id"] == "example/Jane_Doe_2022#0"
    assert rows[0]["title"] == "Caring for the sea"
    assert {row["section"] for row in rows} >= {"1. Introduction"}


def test_embedded_text_fits_size_with_a_long_title(tmp_path):
    body = " ".join(f"Sentence number {n} is here." for n in range(40))
    write(tmp_path / "example" / "Jane_Doe_2020.md", f"## 2. Method\n\n{body}\n")
    [document] = corpus.find_documents(tmp_path)
    metadata = {"title": " ".join(["Long"] * 10), "author": "Jane Doe", "year": "2020"}
    rows = corpus.document_rows(document, metadata, count_words, 30, 10, [])
    assert len(rows) > 1
    for row in rows:
        assert search.passage_text(row).startswith(metadata["title"] + "\n2. Method\n")
        assert count_words(search.passage_text(row)) <= 30


def test_document_rows_drop_noise_lines_and_short_sections(tmp_path):
    write(tmp_path / "example" / "Jane_Doe_2020.md", DOCUMENT_MD)
    [document] = corpus.find_documents(tmp_path)
    drop = repo_options().drop_lines
    rows = corpus.document_rows(document, None, count_words, 50, 5, [], drop, min_tokens=6)
    assert {row["section"] for row in rows} == {"1. Introduction", "References"}


def test_document_corpus_links_chunks_to_paragraph_rows(tmp_path):
    write(tmp_path / "example" / "Jane_Doe_2020.md", PAGED_MD)
    [document] = corpus.find_documents(tmp_path)
    metadata = {"title": "Coast", "author": "Jane Doe", "year": "2020"}
    chunks, paragraphs = corpus.document_corpus(
        document, metadata, count_words, 50, 5, [], min_paragraph_tokens=5
    )
    assert [row["id"] for row in paragraphs] == [
        "example/Jane_Doe_2020#p0",
        "example/Jane_Doe_2020#p1",
    ]  # "This one stands alone." is under 5 words: no row of its own
    assert paragraphs[0] | {"text": ""} == {
        "id": "example/Jane_Doe_2020#p0",
        "document": "Jane_Doe_2020",
        "collection": "example",
        "author": "Jane Doe",
        "year": "2020",
        "title": "Coast",
        "section": "Introduction",
        "page": 3,
        "last_page": 4,
        "text": "",
    }
    [only] = chunks
    assert "This one stands alone." in only["text"]  # still searchable inside the chunk
    assert only["paragraphs"] == [row["id"] for row in paragraphs]
    # without a minimum, the short paragraph gets a row too, and document_rows is the chunk half
    [row] = corpus.document_rows(document, metadata, count_words, 50, 5, [])
    assert row["paragraphs"] == [f"example/Jane_Doe_2020#p{n}" for n in range(3)]


def test_document_without_database_row_still_indexed(tmp_path):
    write(tmp_path / "example" / "Jane_Doe_2020.md", "## A\n\nText.\n")
    [document] = corpus.find_documents(tmp_path)
    [row] = corpus.document_rows(document, None, count_words, 50, 5, [])
    assert row["title"] is None and row["text"] == "Text."


def test_vector_key_changes_with_anything_embedded():
    embedder = model.spec("embedder")
    key = search.vector_key(embedder, ["a", "b"])
    assert search.vector_key(embedder, ["a", "b"]) == key
    assert search.vector_key(embedder, ["a", "c"]) != key
    other = model.parse_block("embedder", {"id": embedder.id, "engine": "embed"})  # mean pooling
    assert search.vector_key(other, ["a", "b"]) != key


def test_cached_vectors_embed_only_when_the_key_changes(tmp_path):
    calls = []

    def fake_embed(texts):
        calls.append(texts)
        return np.ones((len(texts), 4))

    document = "report-abcdef12"
    vectors, fresh = search.cached_vectors(tmp_path, document, "k1", ["a", "b"], fake_embed)
    assert fresh and vectors.shape == (2, 4) and vectors.dtype == np.float16
    _, fresh = search.cached_vectors(tmp_path, document, "k1", ["a", "b"], fake_embed)
    assert not fresh and len(calls) == 1
    # another document whose id merely starts like this one keeps its cache
    search.cached_vectors(tmp_path, document + "-x", "k9", ["z"], fake_embed)
    search.cached_vectors(tmp_path, document, "k2", ["a"], fake_embed)
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        f"{document}-k2.npy",
        f"{document}-x-k9.npy",
    ]
    search.prune_vectors(tmp_path, {document})
    assert [path.name for path in tmp_path.iterdir()] == [f"{document}-k2.npy"]


def test_join_vectors_keeps_order_and_skips_documents_without_chunks():
    blocks = [np.ones((2, 3)), np.zeros((0, 0)), np.full((1, 3), 2.0)]
    joined = search.join_vectors(blocks)
    assert joined.tolist() == [[1, 1, 1], [1, 1, 1], [2, 2, 2]]
    assert search.join_vectors([]).shape == (0, 0)


def test_inspect_matches_id_author_or_title():
    from search.__main__ import matches

    metadata = {"author": "Jane Doe", "title": "Harbours of the north"}
    assert matches("doe", "report-abcdef12", metadata)
    assert matches("HARBOURS", "report-abcdef12", metadata)
    assert matches("abcdef", "report-abcdef12", None)
    assert not matches("desert", "report-abcdef12", metadata)


# search


def test_top_k_ranks_by_cosine():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    hits = search.top_k(matrix, np.array([0.0, 1.0]), 2)
    assert [index for index, _ in hits] == [1, 2]
    assert hits[0][1] == pytest.approx(1.0)


def test_top_k_scores_a_float16_matrix_in_blocks(monkeypatch):
    monkeypatch.setattr(search, "SCORE_BLOCK", 2)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6], [0.0, -1.0]], np.float16)
    hits = search.top_k(matrix, np.array([0.0, 1.0], np.float32), 3)
    assert [index for index, _ in hits] == [1, 2, 3]


def test_index_refuses_a_different_embedder(tmp_path):
    embedder = model.spec("embedder")
    search.save_index(tmp_path, np.zeros((3, 4), dtype=np.float32), embedder)
    loaded = search.load_index(tmp_path, embedder)
    assert loaded.shape == (3, 4) and loaded.dtype == np.float16
    other = model.parse_block("embedder", {"id": "other/model", "engine": "embed"})
    with pytest.raises(ConfigError, match="make index"):
        search.load_index(tmp_path, other)


def test_index_refuses_a_different_pooling(tmp_path):
    embedder = model.spec("embedder")
    search.save_index(tmp_path, np.zeros((1, 4)), embedder)
    flipped = "mean" if embedder.pooling == "last" else "last"
    other = model.parse_block(
        "embedder", {"id": embedder.id, "engine": "embed", "pooling": flipped}
    )
    with pytest.raises(ConfigError, match="pooling"):
        search.load_index(tmp_path, other)


def test_no_index_says_make_index(tmp_path):
    with pytest.raises(ConfigError, match="make index"):
        search.load_index(tmp_path, model.spec("embedder"))


def test_passage_text_includes_title_and_section():
    row = {"title": "Coast", "section": "Method", "text": "Interviews."}
    assert search.passage_text(row) == "Coast\nMethod\nInterviews."
    assert search.passage_text(row | {"title": None}) == "Method\nInterviews."
    assert search.passage_text({"section": "", "text": "Cover."}) == "Cover."


def hit(document, score, section="S"):
    row = {"collection": "example", "document": document, "section": section}
    return (row, score)


def test_group_by_document_returns_distinct_documents_best_first():
    hits = [hit("A", 0.9, "a1"), hit("A", 0.8, "a2"), hit("A", 0.7, "a3"), hit("B", 0.6)]
    hits += [hit("C", 0.5), hit("B", 0.4)]
    grouped = search.group_by_document(hits, k=2, per_document=2)
    assert [best for best, _ in grouped] == [0.9, 0.6]
    assert [row["section"] for row, _ in grouped[0][1]] == ["a1", "a2"]
    assert [row["document"] for _, passages in grouped for row, _ in passages] == [
        "A",
        "A",
        "B",
        "B",
    ]


class FakeCrossEncoder:
    """Stands in for a tokenizer + sequence-classification model: logit = len(text) - 3."""

    device = "cpu"

    def __init__(self):
        self.batches = []

    def tokenizer(self, questions, texts, **kwargs):
        import torch

        assert kwargs["truncation"] == "only_second" and set(questions) == {"q"}
        self.batches.append(list(texts))
        lengths = torch.tensor([[len(text) - 3.0] for text in texts])
        return SimpleNamespace(to=lambda device: {"lengths": lengths})

    def __call__(self, lengths):
        return SimpleNamespace(logits=lengths)


def test_rerank_scores_each_text_in_order_across_batches():
    fake = FakeCrossEncoder()
    scores = search.rerank("q", ["abc", "abcdefghij", "a", "abcd"], fake.tokenizer, fake, 512, 3)
    assert fake.batches == [["abc", "abcdefghij", "a"], ["abcd"]]
    assert scores[0] == pytest.approx(0.5)  # logit 0
    assert list(np.argsort(-scores)) == [1, 3, 0, 2]
    assert ((scores > 0) & (scores < 1)).all()
    assert search.rerank("q", [], fake.tokenizer, fake).size == 0


def test_keep_relevant_sorts_and_drops_weak_hits():
    hits = [hit("A", 0.2), hit("B", 0.9), hit("C", 0.6)]
    assert [row["document"] for row, _ in search.keep_relevant(hits, 0.5)] == ["B", "C"]
    assert search.keep_relevant(hits, 0.95) == []


def test_inspect_report_shows_embedded_text_paragraphs_and_drops():
    row = {"id": "example/X#0", "title": "Coast", "section": "Method", "text": "Interviews."}
    paragraph = {"id": "example/X#p0", "page": 7, "last_page": 7, "text": "Interviews."}
    sections = [
        corpus.Section("", (corpus.Paragraph("Jane"),), corpus.TOO_SHORT),
        corpus.Section("Method", (corpus.Paragraph("Interviews."),)),
    ]
    report = search.inspect_report(sections, [row], "passage: ", count_words, [paragraph])
    assert "passage: Coast\nMethod\nInterviews." in report
    assert "--- example/X#0  (4 tokens)" in report
    assert "--- example/X#p0  PDF p. 7  (1 tokens)" in report
    assert "[dropped: too short] (title page): Jane" in report


def test_paragraph_reference_shows_pages_when_known(tmp_path):
    row = {"collection": "example", "document": "X_2020", "section": "Method"}
    assert search.cite_paragraph(row | {"page": 12, "last_page": 12}) == (
        "example · Method · PDF p. 12"
    )
    assert search.cite_paragraph(row | {"page": 12, "last_page": 13}).endswith("PDF p. 12-13")
    assert search.cite_paragraph(row | {"page": None, "last_page": None}) == "example · Method"
    assert search.cite_paragraph(row | {"section": ""}) == "example · (title page)"
    assert search.pdf_path(row, tmp_path) == tmp_path / "example" / "X_2020.pdf"


def test_describe_cites_title_author_year_and_section():
    row = {"document": "X_2020", "title": "Coast", "author": "X", "year": "2020"}
    row["section"] = "Method"
    assert search.describe(row) == "Coast (X, 2020) - Method"
    assert search.describe({"document": "X_2020", "section": "Method"}) == "X_2020 - Method"
    assert search.cite(row) == "Coast (X, 2020)"
