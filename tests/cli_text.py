"""Plain CLI help text for assertions."""

from __future__ import annotations

import os
import re
import subprocess
import sys

_ANSI = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_COLOR_KEYS = frozenset({'FORCE_COLOR', 'CLICOLOR_FORCE', 'CLICOLOR'})


def help_text(*args: str) -> str:
    """Return ``patent-ate`` help with color and wrapping stripped."""
    env = {key: value for key, value in os.environ.items() if key.upper() not in _COLOR_KEYS}
    env['NO_COLOR'] = '1'
    env['TERM'] = 'dumb'
    env['COLUMNS'] = '200'
    result = subprocess.run(
        [sys.executable, '-m', 'patent_ate', *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    return _ANSI.sub('', result.stdout)


def compact_help(text: str) -> str:
    """Collapse whitespace so wrapped or styled flags stay one token."""
    return re.sub(r'\s+', '', text)
