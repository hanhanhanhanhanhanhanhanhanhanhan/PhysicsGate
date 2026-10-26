from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "physicsgate" / "submitted"
MODULES = sorted(
    "physicsgate." + ".".join(p.relative_to(ROOT / "physicsgate").with_suffix("").parts)
    for p in PACKAGE.rglob("*.py")
    if p.name != "__init__.py"
)


@pytest.mark.parametrize("module", MODULES)
def test_submitted_modules_import(module: str) -> None:
    importlib.import_module(module)


def test_submitted_code_has_no_local_paths() -> None:
    for path in PACKAGE.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users" not in text and "sys.path.insert" not in text, path


def test_workdir_defaults_to_repository_root() -> None:
    from physicsgate.submitted.paths import WORKDIR

    assert WORKDIR == ROOT.resolve() or "PHYSICSGATE_WORKDIR" in __import__("os").environ


def test_reproduction_steps_use_evaluated_splits() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import reproduce_submitted as rs

    config = json.loads((ROOT / "configs" / "evaluated_splits.json").read_text(encoding="utf-8"))
    for dataset, spec in config["datasets"].items():
        expected = [str(s) for s in spec["split_ids"]]
        for stages in (rs.submitted_stages(dataset), rs.revised_stages(dataset)):
            commands = [cmd for stage in stages for cmd in stage]
            assert commands, dataset
            for cmd in commands:
                assert importlib.util.find_spec(cmd[0]) is not None, cmd[0]
            args = " ".join(" ".join(cmd[1:]) for cmd in commands)
            for seed in expected:
                assert seed in args, (dataset, seed)


def test_repeated_split_configuration() -> None:
    repeated = json.loads((ROOT / "configs" / "repeated_splits.json").read_text(encoding="utf-8"))
    evaluated = json.loads((ROOT / "configs" / "evaluated_splits.json").read_text(encoding="utf-8"))
    for dataset, spec in repeated["datasets"].items():
        ids = spec["split_ids"]
        assert len(ids) == 50 and len(set(ids)) == 50, dataset
        assert spec["selection_rule"], dataset
    assert repeated["datasets"]["ESTM"]["split_ids"][:5] == evaluated["datasets"]["ESTM"]["split_ids"]


def test_figure_steps_use_existing_modules() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import reproduce_figures as rf

    for protocol in ("submitted", "revised"):
        runner = rf.Runner.__new__(rf.Runner)
        runner.protocol = protocol
        for name in ("physicsgate.submitted.pv.pipeline", "physicsgate.submitted.pv.stabilization",
                     "physicsgate.submitted.mechanism.estm", "physicsgate.submitted.mechanism.sse",
                     "physicsgate.submitted.mechanism.lmb", "physicsgate.submitted.mechanism.run_splits"):
            assert importlib.util.find_spec(runner.module(name)) is not None, (protocol, name)
    for step in rf.STEPS:
        assert callable(getattr(rf, f"step_{step}")), step
