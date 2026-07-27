"""Draw the Nature-style 50-seed ESTM/PV PhysicsGate mechanism figure.

Panel a shows target-dependent gate allocation. Panel b compares predicted
weights with a post hoc sample-optimal diagnostic. Panel c reports reliability
feature ablations. Panel d compares the adaptive gate with a train-OOF-tuned
fixed scalar blend.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd

from plot_style import (
    CI_WIDTH,
    DOMAIN_COLORS,
    FONT_LABEL,
    FONT_PANEL,
    MARKER_EDGE_WIDTH,
    MEAN_MARKER_SIZE,
    METHOD_COLORS,
    METHOD_LIGHT,
    PAPER_BAND,
    SPLIT_MARKER_SIZE,
    TEXT,
    add_group_separator,
    add_right_group_bracket,
    apply_manuscript_rcparams,
    centered_title,
    style_box,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "source_data" / "figure4"
RESULTS_DIR = ROOT / "results" / "figures"

WEIGHT_FILE = DATA_DIR / "Figure4a_Weight_BySeedTarget.csv"
CALIBRATION_FILE = DATA_DIR / "Figure4b_Calibration_ByDatasetSummary.csv"
ABLATION_FILE = DATA_DIR / "Figure4c_Ablation_TargetSummary.csv"
GATE_FIXED_FILE = (
    DATA_DIR / "Figure4d_PhysicsGate_vs_FixedBlend_BySeedTarget.csv"
)

BLUE = METHOD_COLORS["PhysicsGate"]
BLUE_LIGHT = METHOD_LIGHT["PhysicsGate"]
HEATMAP_BLUE = "#3C5488"
ESTM_COLOR = DOMAIN_COLORS["ESTM"]
PV_COLOR = DOMAIN_COLORS["PV"]
NEUTRAL = "#777772"
NEUTRAL_LIGHT = "#C8C7C1"
BAND = PAPER_BAND
ZERO = "#333333"

TARGET_ORDER = [
    ("ESTM", "Seebeck coefficient", r"$S$"),
    ("ESTM", r"$ZT$", r"$ZT$"),
    (
        "ESTM",
        r"$\lg \sigma_{\rm elec}$",
        r"$\mathit{lg}\,\sigma_{\mathit{elec}}$",
    ),
    (
        "ESTM",
        r"$\lg \kappa_{\rm thermal}$",
        r"$\mathit{lg}\,\kappa_{\mathit{thermal}}$",
    ),
    ("PV", "FF", "FF"),
    ("PV", r"$J_{\rm SC}$", r"$J_{\mathit{SC}}$"),
    ("PV", "PCE", "PCE"),
    ("PV", r"$V_{\rm OC}$", r"$V_{\mathit{OC}}$"),
]
TARGET_INDEX = {
    (dataset, source): index
    for index, (dataset, source, _display) in enumerate(TARGET_ORDER)
}
TARGET_LABELS = [display for _, _, display in TARGET_ORDER]

FEATURE_ORDER = [
    "R1",
    "R2",
    "R3",
    "R1+R2",
    "R1+R3",
    "R2+R3",
    "R1+R2+R3",
]
FEATURE_LABELS = [
    r"$R_1$",
    r"$R_2$",
    r"$R_3$",
    r"$R_1+R_2$",
    r"$R_1+R_3$",
    r"$R_2+R_3$",
    r"$R_1+R_2+R_3$",
]
FULL_FEATURE_SET = "R1+R2+R3"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20_260_726
PANEL_TITLE_SIZE = 8.0
PANEL_TITLE_PAD = 2.5


apply_manuscript_rcparams()


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_bootstrap: int = BOOTSTRAP_SAMPLES,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap values must be a non-empty vector")
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = draws.mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax._physicsgate_panel_label = label


def place_panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Align labels with each panel's leftmost element and title midline."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = fig.transFigure.inverted()
    for index, ax in enumerate(axes):
        label = getattr(ax, "_physicsgate_panel_label", None)
        if not label:
            continue
        left_edges = [ax.get_window_extent(renderer).x0]
        left_edges.extend(
            tick.get_window_extent(renderer).x0
            for tick in ax.get_yticklabels()
            if tick.get_visible() and tick.get_text()
        )
        if ax.yaxis.label.get_visible() and ax.yaxis.label.get_text():
            left_edges.append(ax.yaxis.label.get_window_extent(renderer).x0)
        title_bbox = ax.title.get_window_extent(renderer)
        if index == 0:
            x_left = inverse.transform((min(left_edges), title_bbox.y0))[0]
        else:
            axes_left = inverse.transform(
                (ax.get_window_extent(renderer).x0, title_bbox.y0)
            )[0]
            x_left = axes_left - 0.020
        title_center_y = inverse.transform(
            (title_bbox.x0, (title_bbox.y0 + title_bbox.y1) / 2.0)
        )[1]
        fig.text(
            x_left,
            title_center_y,
            label,
            fontsize=FONT_PANEL,
            fontweight="bold",
            ha="left",
            va="center",
            color="#111111",
        )


def add_target_bands(ax: plt.Axes) -> None:
    ax.axhspan(3.5, 7.5, color=BAND, ec="none", zorder=0)
    ax.axhline(3.5, color=TEXT, linewidth=0.9, zorder=1)
    add_right_group_bracket(ax, 3.55, 7.45, "ESTM")
    add_right_group_bracket(ax, -0.45, 3.45, "PV")


def prepare_weight_data(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["target_index"] = [
        TARGET_INDEX[(dataset, target)]
        for dataset, target in zip(frame["dataset"], frame["target_display"])
    ]
    frame["y"] = 7 - frame["target_index"]
    summary_rows: list[dict[str, object]] = []
    for (dataset, target, target_index), group in frame.groupby(
        ["dataset", "target_display", "target_index"], sort=False
    ):
        mean, low, high = bootstrap_mean_interval(
            group["mean_weight"].to_numpy(dtype=float), rng=rng
        )
        summary_rows.append(
            {
                "dataset": dataset,
                "target_display": target,
                "target_index": int(target_index),
                "n_seeds": int(group["seed"].nunique()),
                "mean_weight": mean,
                "bootstrap_CI_low": low,
                "bootstrap_CI_high": high,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("target_index")
    return frame, summary


def draw_weight_panel(
    ax: plt.Axes, units: pd.DataFrame, summary: pd.DataFrame
) -> None:
    add_target_bands(ax)
    for target_index in range(8):
        group = units.loc[units["target_index"].eq(target_index)]
        y = 7 - target_index
        values = group["mean_weight"].to_numpy(dtype=float)
        violin = ax.violinplot(
            values,
            positions=[y],
            vert=False,
            widths=0.58,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.35,
        )
        for body in violin["bodies"]:
            body.set_facecolor(BLUE_LIGHT)
            body.set_edgecolor("none")
            body.set_alpha(0.55)
            body.set_zorder(2)
        ordered = group.sort_values("seed")
        jitter = np.linspace(-0.18, 0.18, len(ordered))
        ax.scatter(
            ordered["mean_weight"],
            y + jitter,
            s=SPLIT_MARKER_SIZE,
            color=BLUE,
            alpha=0.24,
            edgecolor=TEXT,
            linewidth=0.18,
            zorder=3,
        )
        row = summary.loc[summary["target_index"].eq(target_index)].iloc[0]
        ax.plot(
            [row["bootstrap_CI_low"], row["bootstrap_CI_high"]],
            [y, y],
            color=BLUE,
            linewidth=CI_WIDTH,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            row["mean_weight"],
            y,
            s=MEAN_MARKER_SIZE,
            color=BLUE,
            edgecolor=TEXT,
            linewidth=MARKER_EDGE_WIDTH,
            zorder=5,
        )
    ax.axvline(0.5, color="#9B9A96", linestyle=(0, (2, 2)), linewidth=0.7, zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks(range(8))
    ax.set_yticklabels(TARGET_LABELS[::-1])
    ax.set_xlabel(r"Gate weight on residual branch, $w$")
    centered_title(
        ax,
        "Target-dependent allocation",
        pad=PANEL_TITLE_PAD,
        fontsize=PANEL_TITLE_SIZE,
    )
    ax.text(
        0.5,
        0.985,
        "Equal blend",
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=5.8,
        color=TEXT,
    )
    style_box(ax)
    add_panel_label(ax, "a")


def draw_calibration_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    ax.plot([0.25, 0.9], [0.25, 0.9], color="#A7A6A1", linewidth=0.75, linestyle="--")
    for dataset, color in (("ESTM", ESTM_COLOR), ("PV", PV_COLOR)):
        group = frame.loc[frame["dataset"].eq(dataset)].sort_values("weight_quintile")
        x = group["mean_weight"].to_numpy(dtype=float)
        y = group["mean_oracle_weight"].to_numpy(dtype=float)
        xerr = 1.96 * group["sem_weight"].to_numpy(dtype=float)
        yerr = 1.96 * group["oracle_weight_sem"].to_numpy(dtype=float)
        ax.plot(x, y, color=color, linewidth=1.0, alpha=0.85, zorder=2)
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            markersize=5.0,
            color=color,
            markeredgecolor=TEXT,
            markeredgewidth=MARKER_EDGE_WIDTH,
            elinewidth=CI_WIDTH,
            capsize=1.5,
            label=dataset,
            zorder=3,
        )
        label_positions = {
            ("ESTM", 1): {"xytext": (3, 7), "ha": "left", "va": "bottom"},
            ("ESTM", 5): {"xytext": (-4, -5), "ha": "right", "va": "top"},
            ("PV", 1): {"xytext": (-3, -7), "ha": "right", "va": "top"},
            ("PV", 5): {"xytext": (4, -5), "ha": "left", "va": "top"},
        }
        for _, row in group.iterrows():
            quintile = int(row["weight_quintile"])
            if quintile in (1, 5):
                label_position = label_positions[(dataset, quintile)]
                ax.annotate(
                    f"Q{quintile}",
                    (row["mean_weight"], row["mean_oracle_weight"]),
                    xytext=label_position["xytext"],
                    textcoords="offset points",
                    fontsize=5.6,
                    color=TEXT,
                    ha=label_position["ha"],
                    va=label_position["va"],
                    bbox={
                        "boxstyle": "round,pad=0.08",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.88,
                    },
                    arrowprops={
                        "arrowstyle": "-",
                        "color": TEXT,
                        "linewidth": 0.40,
                        "shrinkA": 1.0,
                        "shrinkB": 4.0,
                    },
                    zorder=6,
                )
    ax.set_xlim(0.25, 0.9)
    ax.set_ylim(0.25, 0.9)
    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_yticks([0.3, 0.5, 0.7, 0.9])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted weight")
    ax.set_ylabel("Mean post hoc optimal weight")
    centered_title(
        ax,
        "Gate-oracle alignment",
        pad=PANEL_TITLE_PAD,
        fontsize=PANEL_TITLE_SIZE,
    )
    ax.legend(
        loc="lower right",
        fontsize=6.5,
        handletextpad=0.35,
        borderaxespad=0.2,
    )
    ax.text(
        0.02,
        0.98,
        "*Q1-Q5, predicted-weight quintiles\nerror bars, 95% CI",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.6,
        color=TEXT,
    )
    style_box(ax)
    add_panel_label(ax, "b")


def prepare_ablation_data(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["target_index"] = [
        TARGET_INDEX[(dataset, target)]
        for dataset, target in zip(frame["dataset"], frame["target_display"])
    ]
    full = (
        frame.loc[frame["feature_set"].eq(FULL_FEATURE_SET)]
        .set_index(["dataset", "target_display"])["mean_log2_gain"]
        .to_dict()
    )
    frame["full_mean_log2_gain"] = [
        full[(dataset, target)]
        for dataset, target in zip(frame["dataset"], frame["target_display"])
    ]
    frame["full_minus_ablation_log2_gain"] = (
        frame["full_mean_log2_gain"] - frame["mean_log2_gain"]
    )
    return (
        frame.loc[frame["feature_set"].isin(FEATURE_ORDER)]
        .sort_values(["target_index", "feature_set"])
        .reset_index(drop=True)
    )


def draw_ablation_panel(ax: plt.Axes, frame: pd.DataFrame) -> None:
    matrix = np.full((8, len(FEATURE_ORDER)), np.nan, dtype=float)
    for _, row in frame.iterrows():
        matrix[int(row["target_index"]), FEATURE_ORDER.index(row["feature_set"])] = row[
            "full_minus_ablation_log2_gain"
        ]
    maximum = max(0.03, float(np.nanmax(np.abs(matrix))))
    cmap = LinearSegmentedColormap.from_list(
        "ablation_delta",
        ["#E4E7ED", "#F6F5F1", "#A8B4CC", "#2D4573"],
    )
    image = ax.imshow(
        matrix,
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-maximum, vcenter=0.0, vmax=maximum),
        aspect="auto",
        interpolation="none",
    )
    ax.set_xticks(range(len(FEATURE_LABELS)))
    ax.set_xticklabels(FEATURE_LABELS, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(8))
    ax.set_yticklabels(TARGET_LABELS)
    ax.set_xticks(np.arange(-0.5, len(FEATURE_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.axhline(3.5, color=TEXT, linewidth=1.15)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            shown_value = 0.0 if abs(value) < 5e-5 else value
            color = "white" if abs(value) > maximum * 0.52 else TEXT
            ax.text(
                column_index,
                row_index,
                f"{shown_value:+.4f}",
                ha="center",
                va="center",
                fontsize=4.45,
                color=color,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    centered_title(
        ax,
        "Reliability-feature ablation",
        pad=PANEL_TITLE_PAD,
        fontsize=PANEL_TITLE_SIZE,
    )
    add_right_group_bracket(ax, -0.45, 3.45, "ESTM")
    add_right_group_bracket(ax, 3.55, 7.45, "PV")
    colorbar = ax.figure.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.055,
        pad=0.20,
        aspect=28,
    )
    colorbar.set_label(
        r"$G_{\mathit{full}}-G_{\mathit{ablated}}$",
        fontsize=FONT_LABEL,
        labelpad=2,
    )
    colorbar.ax.tick_params(labelsize=6.2, width=0.65, length=2.5)
    colorbar.outline.set_linewidth(0.7)
    add_panel_label(ax, "c")


def prepare_gate_fixed_data(
    frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    frame["target_index"] = [
        TARGET_INDEX[(dataset, target)]
        for dataset, target in zip(frame["dataset"], frame["target_display"])
    ]
    frame["y"] = 7 - frame["target_index"]
    rows: list[dict[str, object]] = []
    for (dataset, target, target_index), group in frame.groupby(
        ["dataset", "target_display", "target_index"], sort=False
    ):
        mean, low, high = bootstrap_mean_interval(
            group["gate_minus_fixed_log2_gain"].to_numpy(dtype=float), rng=rng
        )
        rows.append(
            {
                "dataset": dataset,
                "target_display": target,
                "target_index": int(target_index),
                "n_seeds": int(group["seed"].nunique()),
                "mean_gate_minus_fixed_log2_gain": mean,
                "bootstrap_CI_low": low,
                "bootstrap_CI_high": high,
                "gate_win_fraction": float(group["gate_beats_fixed"].mean()),
            }
        )
    summary = pd.DataFrame(rows).sort_values("target_index")
    return frame, summary


def draw_gate_fixed_panel(
    ax: plt.Axes, units: pd.DataFrame, summary: pd.DataFrame
) -> None:
    add_target_bands(ax)
    for target_index in range(8):
        group = units.loc[units["target_index"].eq(target_index)].sort_values("seed")
        y = 7 - target_index
        jitter = np.linspace(-0.19, 0.19, len(group))
        ax.scatter(
            group["gate_minus_fixed_log2_gain"],
            y + jitter,
            s=SPLIT_MARKER_SIZE,
            color=NEUTRAL,
            alpha=0.22,
            edgecolor=TEXT,
            linewidth=0.18,
            zorder=2,
        )
        row = summary.loc[summary["target_index"].eq(target_index)].iloc[0]
        ax.plot(
            [row["bootstrap_CI_low"], row["bootstrap_CI_high"]],
            [y, y],
            color=BLUE,
            linewidth=CI_WIDTH,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            row["mean_gate_minus_fixed_log2_gain"],
            y,
            s=MEAN_MARKER_SIZE,
            color=BLUE,
            edgecolor=TEXT,
            linewidth=MARKER_EDGE_WIDTH,
            zorder=4,
        )
    ax.axvline(0, color=ZERO, linewidth=0.8, zorder=1)
    ax.set_xlim(-0.035, 0.09)
    ax.set_ylim(-0.5, 7.5)
    ax.set_xticks([-0.025, 0.0, 0.025, 0.05, 0.075])
    ax.set_yticks(range(8))
    ax.set_yticklabels(TARGET_LABELS[::-1])
    ax.set_xlabel(r"$\Delta G=G_{\mathit{adaptive}}-G_{\mathit{fixed}}$")
    centered_title(
        ax,
        "Adaptive versus fixed blending",
        pad=PANEL_TITLE_PAD,
        fontsize=PANEL_TITLE_SIZE,
    )
    ax.text(
        0.98,
        0.515,
        "ESTM: adaptive gain (+1.16%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color=BLUE,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": BAND,
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=7,
    )
    ax.text(
        0.98,
        0.475,
        "PV: no clear advantage",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.4,
        color=PV_COLOR,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.12",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.92,
        },
        zorder=7,
    )
    ax.text(
        0.98,
        0.018,
        "*Positive values favour sample-adaptive weighting",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.2,
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.10",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
        zorder=7,
    )
    style_box(ax)
    add_panel_label(ax, "d")


def main() -> None:
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    weight_units, weight_summary = prepare_weight_data(
        pd.read_csv(WEIGHT_FILE), rng
    )
    calibration = pd.read_csv(CALIBRATION_FILE)
    ablation = prepare_ablation_data(pd.read_csv(ABLATION_FILE))
    gate_fixed_units, gate_fixed_summary = prepare_gate_fixed_data(
        pd.read_csv(GATE_FIXED_FILE), rng
    )

    expected_units = 8 * 50
    if len(weight_units) != expected_units or len(gate_fixed_units) != expected_units:
        raise RuntimeError("Figure 4 requires exactly 50 seeds for each of 8 targets")
    for name, frame in {
        "panel a": weight_units,
        "panel d": gate_fixed_units,
    }.items():
        seeds_per_dataset = frame.groupby("dataset")["seed"].nunique()
        units_per_target = frame.groupby(["dataset", "target_id"]).size()
        if not (seeds_per_dataset == 50).all() or not (units_per_target == 50).all():
            raise RuntimeError(f"Figure 4 seed manifest is incomplete for {name}")
    if not np.isfinite(weight_units["mean_weight"]).all():
        raise RuntimeError("Panel a contains non-finite weights")
    if not np.isfinite(gate_fixed_units["gate_minus_fixed_log2_gain"]).all():
        raise RuntimeError("Panel d contains non-finite paired gains")

    width_in = 183.0 / 25.4
    height_in = 166.0 / 25.4
    fig = plt.figure(figsize=(width_in, height_in))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.12,
        right=0.965,
        bottom=0.085,
        top=0.955,
        wspace=0.43,
        hspace=0.22,
        width_ratios=[1.04, 0.96],
        height_ratios=[1.0, 1.04],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_weight_panel(ax_a, weight_units, weight_summary)
    draw_calibration_panel(ax_b, calibration)
    draw_ablation_panel(ax_c, ablation)
    draw_gate_fixed_panel(ax_d, gate_fixed_units, gate_fixed_summary)
    place_panel_labels(fig, [ax_a, ax_b, ax_c, ax_d])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for base in (RESULTS_DIR / "Figure4",):
        fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote Figure 4 to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
