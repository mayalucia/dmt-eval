"""Make brain_score_dmt importable as an alias for dmt.domains.brainscore.

The literate sources (07-init.org) define brain_score_dmt as the public
namespace, but pyproject.toml packages src/dmt. Until the org files are
updated, this shim bridges the gap at test time.
"""
import importlib
import sys

import dmt.domains.brainscore as _bs

# Register the package
sys.modules["brain_score_dmt"] = _bs

# Register submodules so 'from brain_score_dmt.interface import X' works
_submodules = [
    "interface", "brain_model", "registry", "adapter",
    "plugin", "compat",
]
for name in _submodules:
    full = f"dmt.domains.brainscore.{name}"
    mod = importlib.import_module(full)
    sys.modules[f"brain_score_dmt.{name}"] = mod

# Register examples subpackage
_examples = importlib.import_module("dmt.domains.brainscore.examples")
sys.modules["brain_score_dmt.examples"] = _examples
for name in ["alexnet", "benchmark"]:
    full = f"dmt.domains.brainscore.examples.{name}"
    mod = importlib.import_module(full)
    sys.modules[f"brain_score_dmt.examples.{name}"] = mod
