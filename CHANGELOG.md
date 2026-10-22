# Changelog

All notable changes to this project are documented in this file.
## [1.1.2] - 2026-10-22

### Bug Fixes

- *(nlp)* Keep surface variants and spell conjunct pieces with their head
## [1.1.1] - 2026-10-22

### Bug Fixes

- *(nlp)* Read fragment keys and keep figure labels whole
- *(cvalue)* Honour the configured DuckDB thread count
## [1.1.0] - 2026-10-22

### Bug Fixes

- *(nlp)* Expand coordinated modifiers and reject fragment keys

### Documentation

- Describe phrase admission and the required table rebuild

### Styling

- *(tests)* Sort the CLI test imports
## [1.0.0.post1] - 2026-09-12

### Documentation

- Link HUPD cutoff and document library usage
- Rewrite library section and trim README redundancy
## [1.0.0] - 2026-09-12

### Bug Fixes

- *(types)* Satisfy mypy on score manifest, Ray, and CLI imports
- *(ci)* Collect the pytest suite on GitHub Actions
- *(ci)* Stop assuming inodes and raw Rich help on Ubuntu

### CI

- Lint, typecheck, and test on push
- *(release)* Publish wheels to PyPI on v* tags
- *(image)* Publish patent-ate container to GHCR

### Documentation

- Add changelog stub for unreleased work
- Describe extract, score, and termhood usage
- Lead the README with tags, install, and a Python start
- Explain C-value on HUPD and add evidence figures
- Cite C-value mass by length and the DF tail

### Features

- *(lexicon)* Ship patent closed-class stop surfaces
- *(cvalue)* Add Ibis C-value algebra
- *(cvalue)* Validate score manifests and extract fingerprints
- *(cvalue)* Share Unicode JATE word-class predicates
- *(cvalue)* Export plan and score entrypoints
- *(spec)* Declare extract and C-value settings
- *(cvalue)* Declare parent work-plan schemas
- *(cvalue)* Plan units then score term Parquet
- *(extract)* Run Ray Data candidate extract
- *(extract)* Start a local Ray session
- *(cvalue)* Index containment with a suffix array
- *(corpus)* Load claim, abstract, and summary JSON
- *(cvalue)* Materialize Ibis stages to Parquet
- *(run)* Validate extract, score, and corpus envelopes
- Export the public patent-ate API
- *(extract)* Write a one-row-per-patent Parquet plan
- *(nlp)* Host spaCy and draw JATE candidates
- *(termhood)* Store generation-addressed fact tables
- *(cli)* Add extract, score, and corpus commands

### Miscellaneous Tasks

- Ignore caches, venvs, and parquet spills
- *(devenv)* Pin nixpkgs and container flake inputs
- Add MIT license
- *(devenv)* Lock flake input revisions
- Lock Python dependencies for uv sync
- Load devenv through direnv
- *(devenv)* Define the Nix Python and container shell

### Testing

- *(fixtures)* Add HUPD sample US20130296823A1
- *(fixtures)* Add HUPD sample US20180191139A1
- Assert package version is importable
- *(cvalue)* Gate 30k mixed-punctuation scoring time
- *(cvalue)* Cover GSA containment and resume
- *(fixtures)* Add smoke YAML for extract settings
- *(cvalue)* Cover staged algebra and resume
- *(cli)* Assert module help lists commands
- *(cli)* Cover help and request forwarding
- *(extract)* Cover candidate rows and compact Parquet
- *(termhood)* Cover generation Parquet and manifest

### Build

- Declare patent-ate package metadata and scripts

