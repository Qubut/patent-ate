"""CLI help lists commands and corpus options."""

import re

from tests.cli_text import compact_help, help_text


def test_module_help_lists_commands() -> None:
    text = help_text('--help')
    assert 'corpus' in text
    assert 'extract' in text
    lowered = text.lower()
    assert 'occupy' not in lowered
    assert 'ip-claim' not in lowered
    assert 'ssv' not in lowered


def test_corpus_help_lists_sample_options() -> None:
    text = help_text('corpus', '--help')
    compact = compact_help(text)
    assert '--output' in compact
    assert '--input-dir' in compact
    assert '--limit' in compact
    assert '--hupd-dir' not in compact
    assert '--extract-workers' in compact
    assert '--extract-block-rows' in compact
    plain = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    assert 'leaves one CPU so Ray can coordinate' in plain
    assert 'occupy' not in text.lower()
