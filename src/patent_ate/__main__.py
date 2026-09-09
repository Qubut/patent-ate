"""Automatic term extraction CLI: ``patent-ate`` or ``python -m patent_ate``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import polars as pl
import structlog
import typer

from patent_ate.extract import (
    DUCKDB_TMP_DIRNAME,
    extract_parquet_parts,
    score_term_parquet,
)
from patent_ate.run import AteRunRequest, run_corpus_ate, run_extract
from patent_ate.spec import AteSpec
from patent_ate.termhood import TERMHOOD_META_NAME, TermhoodStore

_log = structlog.get_logger(__name__)

app = typer.Typer(
    name='patent-ate',
    help=(
        'Extract ranked multiword terms from patent text and commit an immutable termhood table.'
    ),
    no_args_is_help=True,
    add_completion=False,
)

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        '--config',
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help='YAML of extract and scoring settings. Package defaults apply when omitted.',
    ),
]
OutputOption = Annotated[
    Path,
    typer.Option('--output', help='Directory of the committed artifact.'),
]
InputDirOption = Annotated[
    Path,
    typer.Option(
        '--input-dir',
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help='Directory of JSON patent files, one object per file.',
    ),
]


@app.command()
def corpus(
    *,
    output: OutputOption,
    input_dir: InputDirOption,
    config: ConfigOption = None,
    limit: Annotated[
        int,
        typer.Option('--limit', min=1, help='Maximum number of JSON patent files to sample.'),
    ] = 1000,
    seed: Annotated[int, typer.Option('--seed', help='RNG seed of the sample.')] = 20260830,
    extract_chunk_size: Annotated[
        int,
        typer.Option('--extract-chunk-size', min=1, help='Patents per Ray Data extract batch.'),
    ] = 64,
    extract_workers: Annotated[
        int,
        typer.Option(
            '--extract-workers',
            min=0,
            help='Extract actor count. 0 leaves one CPU so Ray can coordinate.',
        ),
    ] = 0,
    extract_block_rows: Annotated[
        int | None,
        typer.Option(
            '--extract-block-rows',
            min=1,
            help='Patent rows per Ray Data extract block. Unset uses the spec default.',
        ),
    ] = None,
    index_cache: Annotated[
        Path | None,
        typer.Option(
            '--index-cache',
            help=(
                'Optional newline-separated list of JSON paths. '
                'When omitted, the directory is walked.'
            ),
        ),
    ] = None,
) -> None:
    """Sample patent JSON, extract candidates, score C-value, and commit termhood."""
    path = run_corpus_ate(
        AteRunRequest(
            output_dir=output,
            config_path=config,
            input_dir=input_dir,
            index_cache=index_cache,
            limit=limit,
            extract_chunk_size=extract_chunk_size,
            extract_workers=extract_workers,
            extract_block_rows=extract_block_rows,
            seed=seed,
        )
    )
    typer.echo(str(path))
    _log.info('patent_ate.corpus.finished', output=str(path), limit=limit)
    raise typer.Exit(0)


@app.command()
def extract(
    *,
    output: OutputOption,
    input_dir: InputDirOption,
    config: ConfigOption = None,
    limit: Annotated[
        int,
        typer.Option('--limit', min=1, help='Maximum number of JSON patent files to sample.'),
    ] = 1000,
    seed: Annotated[int, typer.Option('--seed', help='RNG seed of the sample.')] = 20260830,
    extract_chunk_size: Annotated[
        int,
        typer.Option('--extract-chunk-size', min=1, help='Patents per Ray Data extract batch.'),
    ] = 64,
    extract_workers: Annotated[
        int,
        typer.Option(
            '--extract-workers',
            min=0,
            help='Extract actor count. 0 leaves one CPU so Ray can coordinate.',
        ),
    ] = 0,
    extract_block_rows: Annotated[
        int | None,
        typer.Option(
            '--extract-block-rows',
            min=1,
            help='Patent rows per Ray Data extract block. Unset uses the spec default.',
        ),
    ] = None,
    index_cache: Annotated[
        Path | None,
        typer.Option(
            '--index-cache',
            help=(
                'Optional newline-separated list of JSON paths. '
                'When omitted, the directory is walked.'
            ),
        ),
    ] = None,
) -> None:
    """Write compact per-patent candidate parquet. Patents already in the sink are skipped."""
    path = run_extract(
        AteRunRequest(
            output_dir=output,
            config_path=config,
            input_dir=input_dir,
            index_cache=index_cache,
            limit=limit,
            extract_chunk_size=extract_chunk_size,
            extract_workers=extract_workers,
            extract_block_rows=extract_block_rows,
            seed=seed,
        )
    )
    typer.echo(str(path))
    _log.info('patent_ate.extract.finished', output=str(path), limit=limit)
    raise typer.Exit(0)


@app.command()
def score(
    *,
    extract_dir: Annotated[
        Path,
        typer.Option(
            '--extract',
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            help='Directory of compact extract parquet parts.',
        ),
    ],
    output: OutputOption,
    config: ConfigOption = None,
) -> None:
    """Score extract parquet into a committed termhood generation without re-parsing."""
    ate = AteSpec.from_yaml(config)
    path = score_term_parquet(
        extract_dir,
        ate=ate,
        temp_dir=output / DUCKDB_TMP_DIRNAME,
        artifact_dir=output,
    )
    typer.echo(str(path))
    _log.info('patent_ate.score.finished', output=str(path))
    raise typer.Exit(0)


@app.command('write-termhood')
def write_termhood_cmd(
    *,
    input_parquet: Annotated[
        Path,
        typer.Option(
            '--input',
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            help='Parquet with columns key, c_value, and df.',
        ),
    ],
    output: OutputOption,
    total_docs: Annotated[
        int,
        typer.Option(
            '--total-docs',
            min=0,
            help='Document count stored on the generation manifest.',
        ),
    ],
) -> None:
    """Publish a generation from key/c_value/df parquet and replace the manifest last."""
    store = TermhoodStore.write_frame(
        pl.read_parquet(input_parquet),
        output,
        total_docs=total_docs,
    )
    typer.echo(str(store.root))
    _log.info(
        'patent_ate.write_termhood.finished',
        output=str(store.root),
        generation=store.meta.generation,
        n_keys=store.meta.n_keys,
    )
    raise typer.Exit(0)


@app.command('read-termhood')
def read_termhood_cmd(
    path: Annotated[
        Path,
        typer.Argument(help='Termhood directory or termhood.meta.json.'),
    ],
) -> None:
    """Print the committed manifest. A missing or mismatched sidecar fails."""
    store = TermhoodStore.open(path)
    typer.echo(store.meta.model_dump_json())
    raise typer.Exit(0)


@app.command()
def inspect(
    path: Annotated[
        Path,
        typer.Argument(help='Termhood directory, manifest, or generation parquet.'),
    ],
    *,
    recast: Annotated[
        bool,
        typer.Option(
            '--recast',
            help=(
                'Rewrite columns to string/float64/int64 when types differ. '
                'Leave the file unchanged when omitted.'
            ),
        ),
    ] = False,
) -> None:
    """Validate a termhood parquet or store and print the row count."""
    if path.is_dir() or path.name == TERMHOOD_META_NAME:
        store = TermhoodStore.open(path)
        typer.echo(str(store.meta.n_keys))
        raise typer.Exit(0)
    typer.echo(str(TermhoodStore.inspect_parquet(path, recast=recast)))
    raise typer.Exit(0)


@app.command('list-generations')
def list_generations_cmd(
    path: Annotated[
        Path,
        typer.Argument(help='Directory that holds termhood generations and the manifest.'),
    ],
) -> None:
    """List generation parquet files and mark the file the manifest currently names."""
    files = TermhoodStore.generations(path)
    committed = (
        TermhoodStore.open(path).meta.data_file if (path / TERMHOOD_META_NAME).is_file() else None
    )

    def line(file: Path) -> str:
        mark = ' committed' if file.name == committed else ''
        return f'{file.name}{mark}'

    if files:
        typer.echo('\n'.join(map(line, files)))
    raise typer.Exit(0)


@app.command('extract-parts')
def extract_parts_cmd(
    path: Annotated[
        Path,
        typer.Argument(help='Directory of compact extract parquet parts.'),
    ],
) -> None:
    """List compact extract parquet files. Files whose names start with termhood are omitted."""
    parts = extract_parquet_parts(path)
    if parts:
        typer.echo('\n'.join(map(str, parts)))
    raise typer.Exit(0)


def main(argv: list[str] | None = None) -> int:
    """Invoke the Typer app; return a process exit code."""
    try:
        app(prog_name='patent-ate', args=argv)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
