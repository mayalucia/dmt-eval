# DMT-Eval — Claude Code Instructions

## What This Is

DMT-Eval (Data, Models, Tests) is a universal validation framework. Its
architectural insight — decoupling analyses from models through formal adapter
interfaces, with structured scientific report generation — was proven over seven
years at the Blue Brain Project (EPFL, 2017-2024) and is now being rebuilt for
any domain where computational models need systematic evaluation.

The guiding principle: **validation is structured argumentation**. A validation
report is a scientific document — abstract, introduction, methods, results,
discussion — produced programmatically from (adapter, model) pairs.

Part of the [MayaLucIA](https://github.com/mayalucia) organisation.

## The Cardinal Rule

**Source of truth is `codev/*.org`, not `.py` files.**

The `codev/` directory contains numbered Org-mode files that weave narrative
explanation with executable code blocks. These files tangle (via `emacs --batch`
or `make tangle`) to produce the Python source files. If you need to modify code
that is managed by a `codev/*.org` file, modify the org file and re-tangle — do
not edit the `.py` file directly.

## Project Topology

```
dmt-eval/                         <- git root
├── codev/                        <- literate source (org files) — SOURCE OF TRUTH
│   └── 00-document-builder.org
├── develop/                      <- strategic documents, project charter
│   └── project-charter.org
├── src/dmt/                      <- Python package (tangled output + new code)
│   ├── __init__.py               <- dmt.evaluate(), dmt.compare()
│   ├── interface.py              <- Protocol-based interfaces
│   ├── adapter.py                <- @dmt.adapter decorator + registry
│   ├── measurement.py            <- @dmt.measurement + Measurement
│   ├── verdict.py                <- statistical verdict
│   ├── dataset.py                <- dmt.Dataset, dmt.Case
│   ├── document/                 <- report generation
│   │   ├── builder.py            <- DocumentBuilder (from codev/00)
│   │   └── section.py
│   ├── metrics/                  <- built-in metrics
│   ├── plot/                     <- plotting utilities
│   └── domains/                  <- domain-specific packs
│       └── brainscore/           <- Brain-Score domain adapter
│           ├── interface.py      <- Interface base (via __init_subclass__)
│           ├── brain_model.py    <- BrainModelInterface, BenchmarkInterface
│           ├── registry.py       <- PluginRegistry (validated, per-interface)
│           ├── adapter.py        <- @adapts, @implements decorators
│           ├── plugin.py         <- ModelPlugin, BenchmarkPlugin + registries
│           ├── compat.py         <- Bridge to Brain-Score's own registries
│           ├── literate/         <- Org source of truth (00-07)
│           └── examples/         <- AlexNet adapter, benchmark runner
├── test/                         <- test files (tangled from codev/ + standalone)
│   └── brainscore/               <- brain-score domain tests
├── Makefile                      <- tangle, test, clean
├── pyproject.toml
└── CLAUDE.md                     <- this file
```

## The Human (mu2tau)

PhD-level theoretical statistical physicist. 20 years across academia and industry.
Works from Emacs. Reads the Org narrative, not the Python.

Do not over-explain. Do not be sycophantic. Push back on flawed reasoning.

## Core Conventions

### Literate Programming

- Org files in `codev/` are the primary artifact for new development
- Each org file tangles to one or more `.py` files via `:tangle` headers
- Narrative sections explain *why*, code blocks show *what*
- Tests are tangled from the same org file as the implementation
- `make tangle` regenerates all Python files from org sources

### Python Environment

- Python 3.11+
- Use `uv` for virtual environments and package management
- Run tests with: `make test` or `uv run pytest`

### Key Architectural Concepts

- **Three-Party Architecture**: Data Interface Authors, Model Adapters, Validation Writers
- **Seven-Level Interface Gradient**: From `dmt.evaluate()` (zero concepts) to Protocol-based interfaces
- **Adapter Pattern**: `typing.Protocol` + `@dmt.adapter` decorator
- **Document System**: DocumentBuilder decorator API producing LabReports
- **Parameterized Measurement**: `@dmt.measurement` with parameter sweeps

### Git

- Do not commit unless asked
- Do not push unless asked
- Commit messages: imperative mood, concise

### Brain-Score Domain (`src/dmt/domains/brainscore/`)

The first domain adapter, implementing DMT's interface/adapter pattern for
[Brain-Score](https://www.brain-score.org/) — a neuroscience model validation
platform. Key design choices:

- **`__init_subclass__` instead of metaclasses**: The original DMT (BBP era) used
  `InterfaceMeta` — a custom metaclass that caused MRO conflicts. This version uses
  `__init_subclass__` hooks: simpler, no metaclass composition needed.
- **Per-interface registries**: DMT's original `InterfaceMeta` had a shared mutable
  `__implementation_registry__` across all interfaces (bug). Fixed here.
- **`@implements` decorator**: Validates compliance at registration time, not runtime.
- **Compatibility bridge** (`compat.py`): `sync_to_brainscore()` maps DMT-registered
  plugins to Brain-Score's own registry format.

Literate sources for this domain live in `src/dmt/domains/brainscore/literate/`
(00-07). Tutorials about Brain-Score itself are in `codev/` (08-12).

### Relationship to BBP-era DMT

The original framework lives at `visood/dmt` (private). This repo inherits the
proven architectural ideas and rebuilds with modern Python:

| BBP Era (2017-2024)             | dmt-eval (2026)                    |
|---------------------------------|------------------------------------|
| Metaclasses (AIMeta)            | `typing.Protocol` + runtime check  |
| WithFields descriptor system    | `dataclasses` or `attrs`           |
| Cheetah3 templates              | Jinja2 templates                   |
| Neuroscience-coupled            | Domain-agnostic core + domain packs|
| No CLI                          | `typer` CLI                        |
| No packaging                    | `pyproject.toml`, `uv`, PyPI       |

### Organisational Context

This module belongs to the **bravli** guild (computational neuroscience)
within the MāyāLucIA organisation. Its guardian spirit is `dmt-eval-guardian`
(see `aburaya/spirits/dmt-eval-guardian/` in the parent repo).

The module is a git submodule at `modules/dmt-eval` in the parent repo.

**Sūtra relay**: The organisational relay is `github.com/mayalucia/sutra`.
Clone locally to `.sutra/` (gitignored) if absent. Use the relay-read
skill to fetch and filter messages. The local HEAD in `.sutra/` is your
read cursor.

**The relay is heard.** If you have organisational needs — wishes about
how things should work, dependencies on other modules, questions for
other projects — write them into the sūtra with appropriate tags.
Messages go to the universe, not to a recipient. The organisation
listens.
