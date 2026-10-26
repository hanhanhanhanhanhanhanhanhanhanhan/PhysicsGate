"""Recompute the additional analyses of the revision (Supplementary Tables S10 and S12-S19).

Each analysis runs the revised pipelines of ``scripts/reproduce_submitted.py``
through a wrapper module, in its own working directory
``results/<analysis>/workdir``; the matching ``scripts/analyze_*.py`` script
then summarizes the records (see README). The PV model selection is taken from
the revised run, so run ``python scripts/reproduce_submitted.py --protocol revised``
first. The five evaluated outer splits of each dataset are used throughout.

    python scripts/reproduce_revision.py ANALYSIS [--datasets ...]

ANALYSIS                      Table  wrapper
r3_audit                      S10    physicsgate.revised.r3_audit
cost_audit                    S13    physicsgate.revised.cost_audit
bagged_direct                 S14    physicsgate.revised.bagged_direct
mtl_baseline                  S15    physicsgate.revised.mtl_baseline
residual_ablation             S16    physicsgate.revised.residual_ablation (inputs: Physics-only and conditions)
residual_nodirect             S16    ... without Direct
residual_direct_target        S16    ... target fitted directly
residual_no_physics           S16    ... target fitted without Physics-only
residual_sse_direct           S16    physicsgate.revised.sse_direct_residual: SSE residual model with the Direct prediction added
estm_tuned_base               S18    physicsgate.revised.estm_diagnostics, first stage of the auxiliary tuning
estm_source_settings          S18    ... Direct at the source authors' setting, tuned auxiliary and residual models
estm_source_settings_blend    S18    the same with the gate fitted to the blend error (physicsgate.revised.gate)
estm_element_stats            S19    ... element-property representation
estm_formula_group            S19    ... composition-held-out outer splits

Table S12 is the benchmark with the gate fitted to the blend error,
``python scripts/reproduce_submitted.py --protocol blend_gate --parallel``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce_submitted import RAW_FILES, revised_stages, seeds  # noqa: E402

PIPELINES = ("physicsgate.submitted.sse.pipeline", "physicsgate.revised.estm_pipeline",
             "physicsgate.submitted.lmb.pipeline", "physicsgate.revised.pv_pipeline")
COST_STAGES = PIPELINES + ("physicsgate.submitted.sse.ordinary", "physicsgate.submitted.lmb.complete",
                           "physicsgate.submitted.pv.selection")
BAGGED_M = {"SSE": 116, "ESTM": 32, "LMB": 51, "PV": 1248}  # refits matching the training time of PhysicsGate
ESTM = {
    "estm_tuned_base": ["--base", "tuned"],
    "estm_source_settings": ["--base", "github_all", "--aux-params", str(ROOT / "configs" / "estm_auxiliary_settings.csv"),
                             "--tune-residual", "--base-cache", "--tune-workers", "4"],
    "estm_element_stats": ["--representation", "element_stats"],
    "estm_formula_group": ["--split", "formula_group"],
}
ESTM["estm_source_settings_blend"] = ESTM["estm_source_settings"]
ANALYSES = ["r3_audit", "cost_audit", "bagged_direct", "mtl_baseline", "residual_ablation", "residual_nodirect",
            "residual_direct_target", "residual_no_physics", "residual_sse_direct", *ESTM]


def prepare(name: str) -> Path:
    workdir = ROOT / "results" / name / "workdir"
    raw = workdir / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (workdir / "results" / "submitted").mkdir(parents=True, exist_ok=True)
    for file in RAW_FILES:
        if not (raw / file).exists():
            try:
                os.link(ROOT / "data" / "raw" / file, raw / file)
            except OSError:
                shutil.copy2(ROOT / "data" / "raw" / file, raw / file)
    return workdir


def copy_pv_selection(workdir: Path) -> None:
    source = ROOT / "results" / "revised" / "workdir" / "results" / "submitted"
    files = sorted(source.glob("submitted_pv_selection_*.csv"))
    if not files:
        raise SystemExit("PV model selection not found; run scripts/reproduce_submitted.py --protocol revised first")
    for file in files:
        shutil.copy2(file, workdir / "results" / "submitted" / file.name)


def wrapper(name: str, dataset: str) -> list[str]:
    if name == "r3_audit":
        return ["physicsgate.revised.r3_audit"]
    if name == "cost_audit":
        return ["physicsgate.revised.cost_audit"]
    if name == "bagged_direct":
        return ["physicsgate.revised.bagged_direct", "--m", str(BAGGED_M[dataset])]
    inputs = {"residual_ablation": "minimal", "residual_nodirect": "no_direct",
              "residual_direct_target": "direct_target", "residual_no_physics": "no_physics"}[name]
    return ["physicsgate.revised.residual_ablation", "--inputs", inputs]


def pipeline_commands(name: str, dataset: str) -> list[list[str]]:
    """The revised stages of a dataset, with the model-fitting stages run through the wrapper of the analysis."""
    wrapped = COST_STAGES if name == "cost_audit" else PIPELINES
    commands = []
    for stage in revised_stages(dataset):
        for cmd in stage:
            if cmd[0] == "physicsgate.submitted.pv.selection" and name != "cost_audit":
                continue  # taken from the revised run
            if cmd[0] == "physicsgate.submitted.sse.ordinary" and name != "cost_audit":
                continue
            if cmd[0] == "physicsgate.submitted.lmb.complete" and name != "cost_audit":
                continue
            commands.append([*wrapper(name, dataset), *cmd] if cmd[0] in wrapped else cmd)
    return commands


DRY_RUN = False


def run(commands: list[list[str]], workdir: Path, extra_env: dict | None = None) -> None:
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               PHYSICSGATE_WORKDIR=str(workdir), **(extra_env or {}))
    for cmd in commands:
        print("+", " ".join(f"{k}={v}" for k, v in (extra_env or {}).items()), "python -m", " ".join(cmd), flush=True)
        if not DRY_RUN:
            subprocess.check_call([sys.executable, "-m", *cmd], cwd=ROOT, env=env)


def main() -> None:
    global DRY_RUN
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("analysis", choices=ANALYSES)
    parser.add_argument("--datasets", nargs="+", default=["SSE", "ESTM", "LMB", "PV"],
                        choices=["SSE", "ESTM", "LMB", "PV"])
    parser.add_argument("--workers", type=int, default=16, help="parallel fits of mtl_baseline")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args()
    DRY_RUN = args.dry_run
    workdir = prepare(args.analysis)

    if args.analysis in ESTM:
        command = ["physicsgate.revised.estm_diagnostics", *ESTM[args.analysis], "--", "--seeds", ",".join(seeds("ESTM"))]
        if args.analysis == "estm_source_settings_blend":
            command = ["physicsgate.revised.gate", *command]
        env = {"LOKY_MAX_CPU_COUNT": "32"} if "source_settings" in args.analysis else None
        run([["physicsgate.submitted.estm.data"], command], workdir, env)
        return

    if args.analysis == "residual_sse_direct":
        run([["physicsgate.revised.sse_direct_residual", "--seeds", ",".join(seeds("SSE")), "--include-physresstack",
              "--output-tag", "submitted"]], workdir)
        return

    if args.analysis == "mtl_baseline":
        run([["physicsgate.submitted.estm.data"], ["physicsgate.submitted.lmb.data"], ["physicsgate.submitted.pv.data"]],
            workdir)
        run([["physicsgate.revised.mtl_baseline", "--workers", str(args.workers), "--dataset", d, "--seeds",
              *seeds(d)] for d in args.datasets], workdir, {"LOKY_MAX_CPU_COUNT": str(args.workers)})
        return

    if "PV" in args.datasets and args.analysis != "cost_audit" and not DRY_RUN:
        copy_pv_selection(workdir)
    if args.analysis == "residual_nodirect" and not DRY_RUN:  # the SSE residual model has no Direct input: as revised
        source = ROOT / "results" / "revised" / "workdir" / "results" / "submitted"
        for file in ("tmp_sse_mnr_data_submitted_multiseed_stability_all_seeds_detailed.csv", "sse_ordinary_stacking.csv"):
            shutil.copy2(source / file, workdir / "results" / "submitted" / file)
    datasets = [d for d in args.datasets if not (args.analysis == "residual_nodirect" and d == "SSE")]  # SSE h has no Direct input
    for dataset in datasets:
        env = {"R3_AUDIT_TAG": dataset.lower()} if args.analysis == "r3_audit" else None
        run(pipeline_commands(args.analysis, dataset), workdir, env)


if __name__ == "__main__":
    main()
