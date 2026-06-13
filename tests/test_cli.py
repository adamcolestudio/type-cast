"""Tests for the CLI's resolution helpers.

The CLI commands themselves are thin glue around the loader + runner,
both of which are covered separately. What's worth pinning here is the
three-way precedence on the output root path — the only piece of CLI
logic that has real branching, and the easiest place for a silent bug
("my run wrote to the wrong disk") to land.
"""
from __future__ import annotations

from pathlib import Path

from generator.cli import resolve_output_root


class TestResolveOutputRoot:
    """CLI > YAML ``output_dir`` > sibling-of-yaml default."""

    def test_cli_flag_wins_over_yaml(self, tmp_path):
        """When both CLI ``--output`` and YAML ``output_dir`` are set,
        the CLI invocation's value wins. Standard CLI > config precedence.
        Operator's intent at invoke time supersedes the file."""
        yaml = tmp_path / "x.yaml"
        out = resolve_output_root(
            yaml_path=yaml,
            cli_output=str(tmp_path / "from_cli"),
            spec_output_dir="/somewhere/from/yaml",
        )
        assert out == (tmp_path / "from_cli").resolve()

    def test_yaml_absolute_path_used_when_no_cli_flag(self, tmp_path):
        """YAML wins when CLI absent. Absolute paths used verbatim —
        the SSD use case (``output_dir: /mnt/ssd/runs``)."""
        yaml = tmp_path / "x.yaml"
        out = resolve_output_root(
            yaml_path=yaml,
            cli_output=None,
            spec_output_dir=str(tmp_path / "absolute_yaml"),
        )
        assert out == (tmp_path / "absolute_yaml").resolve()

    def test_yaml_relative_path_resolves_against_yaml_dir(self, tmp_path):
        """Relative paths in YAML are relative to the YAML FILE, not
        the CWD. Keeps experiments portable: moving the YAML to another
        repo doesn't change where it writes (relative to itself)."""
        yaml_dir = tmp_path / "nested" / "experiments"
        yaml_dir.mkdir(parents=True)
        yaml = yaml_dir / "x.yaml"
        out = resolve_output_root(
            yaml_path=yaml,
            cli_output=None,
            spec_output_dir="../runs",
        )
        assert out == (tmp_path / "nested" / "runs").resolve()

    def test_default_used_when_both_absent(self, tmp_path):
        """No CLI flag, no YAML field → sibling-of-yaml default. Same
        layout the README documents."""
        yaml_dir = tmp_path / "experiments"
        yaml_dir.mkdir()
        yaml = yaml_dir / "x.yaml"
        out = resolve_output_root(
            yaml_path=yaml,
            cli_output=None,
            spec_output_dir=None,
        )
        assert out == (tmp_path / "output").resolve()

    def test_tilde_expansion_on_yaml(self):
        """``~/foo`` in YAML expands to the user's home — relied on
        when an operator types a friendly path. ``Path.expanduser`` is
        a no-op on absolute non-tilde paths so the absolute-path test
        above already covers the no-op case."""
        yaml = Path("/tmp/some.yaml")
        out = resolve_output_root(
            yaml_path=yaml,
            cli_output=None,
            spec_output_dir="~/typecast_runs",
        )
        assert "~" not in str(out)
        assert out == (Path.home() / "typecast_runs").resolve()

    def test_tilde_expansion_on_cli(self):
        yaml = Path("/tmp/some.yaml")
        out = resolve_output_root(
            yaml_path=yaml,
            cli_output="~/cli_runs",
            spec_output_dir=None,
        )
        assert out == (Path.home() / "cli_runs").resolve()
