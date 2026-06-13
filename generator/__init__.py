"""Type Cast generation driver.

Offline video generation pipeline for the Type Cast installation. Loads
an experiment YAML, iterates a prompt × sweep-band matrix, drives the
scope-attention-bender pipeline once per cell, encodes each result to
Pi-optimal MP4, and optionally concat-merges per prompt.

Public entry points:

  * :func:`generator.cli.main` — ``type-cast`` command-line interface
  * :class:`generator.experiment.ExperimentSpec` — the parsed YAML model
  * :func:`generator.runner.run_experiment` — programmatic execution

The package is split into single-purpose modules to keep each
responsibility small and individually testable. The runner glues them
together and is the only module that knows the full pipeline.
"""
from __future__ import annotations

__version__ = "0.1.0"
