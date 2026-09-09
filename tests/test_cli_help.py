"""CLI help lists commands and corpus options."""

import re
import subprocess
import sys


def test_module_help_lists_commands() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'patent_ate', '--help'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert 'corpus' in result.stdout
    assert 'extract' in result.stdout
    lowered = result.stdout.lower()
    assert 'occupy' not in lowered
    assert 'ip-claim' not in lowered
    assert 'ssv' not in lowered


def test_corpus_help_lists_sample_options() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'patent_ate', 'corpus', '--help'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '--output' in result.stdout
    assert '--input-dir' in result.stdout
    assert '--limit' in result.stdout
    assert '--hupd-dir' not in result.stdout
    assert '--extract-workers' in result.stdout
    assert '--extract-block-rows' in result.stdout
    plain = re.sub(r'[^A-Za-z0-9]+', ' ', result.stdout)
    assert 'leaves one CPU so Ray can coordinate' in plain
    assert 'occupy' not in result.stdout.lower()
