# patent-ate

[![CI](https://github.com/Qubut/patent-ate/actions/workflows/ci.yml/badge.svg)](https://github.com/Qubut/patent-ate/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**NLP** · **patents** · **automatic term extraction** · **C-value** · **spaCy** · **JATE** · **DuckDB** · **Ibis** · **Ray**

Automatic term extraction for patent claims, abstracts, and summaries. patent-ate
lists the multiword terms a corpus actually uses, scores nested frequency with
C-value, and commits an immutable **termhood** table: each row is a phrase, a
nested-frequency score, and how many documents contain that phrase.

`corpus` runs extract and score in one job. Stop after `extract` and run `score`
later if a parse is interrupted. A finished score writes a generation Parquet
file, then updates a small manifest.

## Install

```text
uv add patent-ate
```

Until the first PyPI release, depend on the git repository:

```text
uv add 'patent-ate @ git+https://github.com/Qubut/patent-ate'
```

Python 3.12, CPU only. The default spaCy model is `en_core_web_lg` (pulled as a
wheel). From a checkout:

```text
devenv shell -- uv sync --group dev --group test
devenv shell -- ruff check src tests
devenv shell -- ruff format --check src tests
devenv shell -- mypy src
devenv shell -- pytest
```

## Quick start

```text
patent-ate --help
python -m patent_ate --help
```

Sample JSON filings, extract candidates, score C-value, and commit termhood:

```text
patent-ate corpus \
  --output ./termhood \
  --input-dir ./patents \
  --limit 1000 \
  --extract-workers 0 \
  --extract-block-rows 256
```

`--extract-workers 0` leaves one CPU so Ray can coordinate. Pass a positive
count to size the spaCy worker pool. `--config` is optional YAML for the spaCy
model, DuckDB memory, and containment strategy. Package defaults apply when it
is omitted.

Extract only, then score without re-parsing:

```text
patent-ate extract --output ./termhood --input-dir ./patents --limit 1000
patent-ate extract-parts ./termhood/extract
patent-ate score --extract ./termhood/extract --output ./termhood
```

## Pipeline

1. **Choose filings.** `corpus` and `extract` sample JSON files under
   `--input-dir`, up to `--limit`. An optional `--index-cache` file can list
   those paths so the tree is not walked again.
2. **Extract candidates.** [Ray Data](https://docs.ray.io/en/latest/data/data.html)
   workers load a [spaCy](https://spacy.io) language model, split each filing
   into sentence groups, and run [JATE](https://github.com/ziqizhang/jate)
   noun-phrase and part-of-speech extractors. Each filing becomes one compact
   Parquet row: term keys, raw frequencies, and the surface strings as they
   appeared. Output lands under `extract/`. Filings already present are skipped.
3. **Score nested termhood.** [Ibis](https://ibis-project.org/) compiles C-value
   over [DuckDB](https://duckdb.org/). The scorer finds which longer candidates
   contain which shorter ones, subtracts nested frequency, and writes `c_value`
   plus document frequency (`df`).
4. **Commit a generation.** The scorer writes `termhood.<generation>.parquet`
   (columns `key`, `c_value`, `df`) and then replaces `termhood.meta.json`.
   Incomplete `.partial` files are leftover scratch, not a committed store.

Inspect, list, and reopen generations without running extract again.

## Input

Commands read a directory of JSON files, one patent object per file. Each object
should carry text in `claims`, `abstract`, and `summary` (missing fields are
skipped). Any directory with those fields works.

One optional corpus that already uses this layout is the **Harvard USPTO Patent
Dataset (HUPD)**: English-language US utility applications, released by Suzgun,
Melas-Kyriazi, Sarkar, Kominers, and Shieber (2022). Dataset card:
[huggingface.co/datasets/HUPD/hupd](https://huggingface.co/datasets/HUPD/hupd).
Paper: [arXiv:2207.04043](https://arxiv.org/abs/2207.04043). Site:
[patentdataset.org](https://patentdataset.org). Point `--input-dir` at a local
tree of those JSON files, or at any other directory with the same fields.

Hugging Face JSONL or Parquet shards are not ingested directly. Convert or dump
to one JSON object per file first.

## How scoring works

Automatic term extraction lists the domain phrases a corpus actually uses.
Patent prose nests phrases ("vehicle interior panel" contains "interior panel"
and "panel"). Counting raw frequency ranks the short leftovers highest.

**C-value** is the nested-frequency statistic this package scores, from Frantzi,
Ananiadou, and Mima, *Automatic recognition of multi-word terms: the
C-value/NC-value method* (2000, *International Journal on Digital Libraries*;
[Springer record](https://link.springer.com/article/10.1007/s007999900023)).
For each candidate phrase:

- Longer phrases get a length factor (base-2 log of word count). JATE 3.3 uses
  `log2(length + 0.1)`, matching its published C-value, so a one-word candidate
  is down-weighted relative to multiword phrases.
- Nested frequency is subtracted. If "panel" only occurs inside longer
  candidates, those parent counts are averaged and taken off "panel"'s own
  count, so generic heads lose when they are leftover fragments.
- The committed `c_value` column is that nested score.

**JATE** is the Python toolkit whose candidate shapes and C-value formula this
scorer matches ([github.com/ziqizhang/jate](https://github.com/ziqizhang/jate)).
Surfaces are the strings as written (hyphens, plural endings). Keys are
normalized forms. `score` unions both into the committed table so lookups hit
either spelling.

**Containment** is the nested-parent search. Shorter candidates use original-string
windows with Unicode word-boundary checks (the same JATE word class on both
paths). Longer candidates can switch to an indexed generalized suffix array
over concatenated keys, built with
[pydivsufsort](https://github.com/louisabraham/pydivsufsort) (`divsufsort`).
Both paths implement the same JATE containment predicate; they are execution
strategies for the same score.

**Document frequency (`df`)** is how many filings contain the key. Inverse
document frequency (IDF) down-weights phrases that appear in most of the
corpus (legal boilerplate such as "another aspect"). See Sparck Jones,
*A statistical interpretation of term specificity and its application in
retrieval* (1972). The fact table stores raw `c_value` and `df`. The package
already forms a single number per key: `product_score_expr` is `log1p` of
non-negative C-value times a [Lucene](https://lucene.apache.org/) BM25-style
IDF, and is zero when `df * df` exceeds the document count (while `df` is
still less than that count). That expression lives in this library. It is not
a separate product.

## Python

```python
from pathlib import Path

from patent_ate import AteSpec, TermhoodStore, corpus_termhood

root = corpus_termhood(
    tuple(sorted(Path('patents').glob('*.json')))[:1000],
    ate=AteSpec(),
    chunk_size=64,
    workers=0,
    artifact_dir=Path('termhood'),
)
store = TermhoodStore.open(root)
by_score, by_df, n_positive = store.report_ranks(20)
```

`write-termhood` / `TermhoodStore.write_frame` publishes a generation from a
`key` / `c_value` / `df` Parquet file and replaces the manifest last.

## Generations

```text
patent-ate write-termhood --input ./keys.parquet --output ./termhood --total-docs 1000
patent-ate read-termhood ./termhood
patent-ate inspect ./termhood
patent-ate inspect ./termhood/termhood.<generation>.parquet
patent-ate list-generations ./termhood
```

`read-termhood` prints the manifest, and fails if the sidecar is missing or
mismatches the fact table. `inspect` prints the row count after schema checks
(unique keys, finite C-values, non-negative `df`). `list-generations` names
every `termhood.<generation>.parquet` and marks the file the manifest currently
points at.

## Container

devenv builds a CPU image whose entrypoint is `patent-ate` (no CUDA). After a
`main` or `v*` CI run:

```text
devenv container --registry docker://ghcr.io/qubut/ copy prod
```

The image name is `ghcr.io/qubut/patent-ate`.

## Changelog and publish

Conventional commits. On `v*` tags, git-cliff regenerates `CHANGELOG.md` before
the publish workflow uploads the wheel:

```text
devenv shell -- git-cliff -o CHANGELOG.md
```

Wheels go to pypi.org through GitHub Actions OIDC (Trusted Publishing). Register
a pending publisher on pypi.org for project `patent-ate`, repository
`Qubut/patent-ate`, workflow `publish.yml`, environment `pypi`.

## License

MIT
