# dmt-eval

**Data, Models, Tests** — universal validation framework for the age of AI agents.

*10 lines of code. Any model. Auditable report.*

```python
import dmt

result = dmt.evaluate(
    model="openai/gpt-4o",
    dataset=[
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "Capital of France?", "expected": "Paris"},
    ],
    metrics=["accuracy", "latency"],
)
result.report("./out")  # Generates structured Markdown report
```

## What This Solves

Every domain where computational models compete faces the same problem:
multiple models, reference data, regulatory pressure for documentation,
and no clean way to connect them.

DMT provides three capabilities nobody else combines:

1. **Model-agnostic adapter interfaces** — the same analysis evaluates LLMs,
   weather models, drug discovery pipelines, or brain simulations by swapping adapters
2. **Structured narrative reports** — LabReports with introduction, methods,
   results, discussion. Not dashboards. Not JSON. Documents.
3. **Parameterized measurement** — systematic parameter sweeps with collection
   policies, standard in computational science, absent from AI evaluation

## Install

```bash
pip install dmt-eval            # core
pip install dmt-eval[llm]       # + OpenAI, Anthropic adapters
pip install dmt-eval[weather]   # + xarray, weather metrics
```

## CLI

After installation, the `dmt` command is available:

```bash
# Evaluate all built-in weather models
dmt eval --scenario weather

# Evaluate drug efficacy models with custom output
dmt eval --scenario drug_efficacy --output ./results

# Compare specific models with a reference baseline
dmt compare --scenario weather --models Persistence,Climatology --reference Climatology

# Re-render an existing report
dmt report ./dmt_report --format md

# List available scenarios and metrics
dmt list scenarios
dmt list metrics
```

## Status

Pre-alpha. Architecture proven over 7 years at EPFL Blue Brain Project
(2017-2024), now being rebuilt with modern Python for universal use.

Part of the [MayaLucIA](https://github.com/mayalucia) organisation.

## License

MIT
