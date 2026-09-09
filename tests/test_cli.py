"""Corpus ATE package CLI: help and request forwarding."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError
from tests.cli_text import compact_help, help_text

from patent_ate import __main__ as ate_main
from patent_ate.extract import write_termhood
from patent_ate.run import AteRunRequest, run_corpus_ate
from patent_ate.spec import AteSpec
from patent_ate.termhood import TERMHOOD_META_NAME, TermhoodStore, TermhoodTable

_FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'hupd'
_SMOKE_YAML = Path(__file__).resolve().parent / 'fixtures' / 'ate.smoke.yaml'
_LEAKS = ('ip-claim', 'ip_claim', 'occupy', 'occupancy', 'ssv', 'gnp')


def test_ate_module_help_lists_commands_not_corpus_flags() -> None:
    text = help_text('--help')
    assert 'corpus' in text
    assert 'extract' in text
    assert 'score' in text
    lowered = text.lower()
    assert all(token not in lowered for token in _LEAKS)


def test_corpus_help_lists_sample_options() -> None:
    text = help_text('corpus', '--help')
    compact = compact_help(text)
    assert '--output' in compact
    assert '--input-dir' in compact
    assert '--limit' in compact
    assert '--hupd-dir' not in compact
    assert '--hupd-limit' not in compact
    assert '--extract-workers' in compact
    assert '--extract-block-rows' in compact
    plain = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    assert 'leaves one CPU so Ray can coordinate' in plain
    lowered = text.lower()
    assert all(token not in lowered for token in _LEAKS)


def test_ate_cli_forwards_args_to_corpus_termhood(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_termhood(
        paths: Sequence[Path],
        *,
        ate: AteSpec,
        chunk_size: int,
        workers: int,
        artifact_dir: Path | None = None,
    ) -> Path:
        captured['n_paths'] = len(paths)
        captured['ate'] = ate
        captured['chunk_size'] = chunk_size
        captured['workers'] = workers
        captured['artifact_dir'] = artifact_dir
        dest = artifact_dir if artifact_dir is not None else tmp_path
        dest.mkdir(parents=True, exist_ok=True)
        return write_termhood(TermhoodTable(total_docs=len(paths)), dest)

    monkeypatch.setattr('patent_ate.run.corpus_termhood', fake_termhood)
    code = ate_main.main([
        'corpus',
        '--output',
        str(tmp_path / 'out'),
        '--config',
        str(_SMOKE_YAML),
        '--input-dir',
        str(_FIXTURES),
        '--limit',
        '2',
        '--extract-chunk-size',
        '5',
        '--extract-workers',
        '3',
        '--seed',
        '11',
    ])
    assert code == 0
    captured_ate = captured['ate']
    assert isinstance(captured_ate, AteSpec)
    assert captured_ate.spacy_model == 'en_core_web_lg'
    assert captured_ate.extract_block_rows == 256
    assert captured['chunk_size'] == 5
    assert captured['workers'] == 3
    assert captured['artifact_dir'] == tmp_path / 'out'
    assert captured['n_paths'] == 2
    store = TermhoodStore.open(tmp_path / 'out')
    assert store.meta.n_keys == 0
    assert (tmp_path / 'out' / TERMHOOD_META_NAME).is_file()
    assert not (tmp_path / 'out' / 'termhood.json').exists()


def test_ate_cli_overrides_extract_block_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_termhood(
        paths: Sequence[Path],
        *,
        ate: AteSpec,
        chunk_size: int,
        workers: int,
        artifact_dir: Path | None = None,
    ) -> Path:
        captured['ate'] = ate
        dest = artifact_dir if artifact_dir is not None else tmp_path
        dest.mkdir(parents=True, exist_ok=True)
        return write_termhood(TermhoodTable(total_docs=len(paths)), dest)

    monkeypatch.setattr('patent_ate.run.corpus_termhood', fake_termhood)
    code = ate_main.main([
        'corpus',
        '--output',
        str(tmp_path / 'out'),
        '--config',
        str(_SMOKE_YAML),
        '--input-dir',
        str(_FIXTURES),
        '--limit',
        '1',
        '--extract-block-rows',
        '32',
    ])
    assert code == 0
    captured_ate = captured['ate']
    assert isinstance(captured_ate, AteSpec)
    assert captured_ate.extract_block_rows == 32
    assert AteSpec.from_yaml(_SMOKE_YAML).extract_block_rows == 256


def test_ate_run_rejects_extract_block_rows_below_cpu_job_width(tmp_path: Path) -> None:
    request = AteRunRequest(
        output_dir=tmp_path / 'out',
        ate=AteSpec(),
        input_dir=_FIXTURES,
        limit=1,
        extract_block_rows=4,
    )
    with pytest.raises(ValidationError, match='extract_block_rows must be at least cpu_job_width'):
        run_corpus_ate(request)


def test_ate_cli_rejects_extract_block_rows_below_cpu_job_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        'patent_ate.run.corpus_termhood',
        lambda *_args, **_kwargs: tmp_path,
    )
    with pytest.raises(ValidationError, match='extract_block_rows must be at least cpu_job_width'):
        ate_main.main([
            'corpus',
            '--output',
            str(tmp_path / 'out'),
            '--config',
            str(_SMOKE_YAML),
            '--input-dir',
            str(_FIXTURES),
            '--limit',
            '1',
            '--extract-block-rows',
            '4',
        ])


def test_ate_run_request_uses_injected_spec(tmp_path: Path) -> None:
    ate = AteSpec(spacy_model='en_core_web_lg', extract_block_rows=256)
    request = AteRunRequest(
        output_dir=tmp_path / 'out',
        ate=ate,
        input_dir=_FIXTURES,
        limit=1,
        extract_workers=1,
    )
    assert request.ate is not None
    assert request.ate.spacy_model == ate.spacy_model
    assert request.input_dir == _FIXTURES


def test_extract_help_matches_corpus_sample_flags() -> None:
    text = help_text('extract', '--help')
    compact = compact_help(text)
    assert '--input-dir' in compact
    assert '--limit' in compact
    assert '--hupd-dir' not in compact
    assert '--output' in compact
    lowered = text.lower()
    assert all(token not in lowered for token in _LEAKS)
