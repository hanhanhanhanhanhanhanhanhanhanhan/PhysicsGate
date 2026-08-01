"""Draw the aligned four-panel ESTM/PV mechanism chain for Figure 5."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_style import (
    CI_WIDTH,
    FONT_PANEL,
    FONT_LEGEND,
    MARKER_EDGE_WIDTH,
    MEAN_MARKER_SIZE,
    METHOD_COLORS,
    METHOD_LIGHT,
    PAPER_BAND,
    SPLIT_MARKER_SIZE,
    TEXT,
    WHITE_BAND,
    add_group_separator,
    add_right_group_bracket,
    apply_manuscript_rcparams,
    centered_title,
    style_box,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "source_data" / "figure5"
RESULTS_DIR = ROOT / "results" / "figures"

UNIT_FILE = DATA_DIR / "Figure5_SeedTarget_Diagnostics.csv"
SUMMARY_FILE = DATA_DIR / "Figure5_Target_Summary.csv"
LITERATURE_FILE = DATA_DIR / "Figure5_Literature_Direct_R2.csv"

ESTM_BAND = PAPER_BAND
PV_BAND = WHITE_BAND
LITERATURE = "#171717"

DATASET_BANDS = {"ESTM": ESTM_BAND, "PV": PV_BAND}

TARGET_LABELS = {
    "seebeck_uV_per_K": r"$S$",
    "ZT": r"$ZT$",
    "log10_electrical_conductivity": r"$\mathit{lg}\,\sigma_{\mathit{elec}}$",
    "log10_thermal_conductivity": r"$\mathit{lg}\,\kappa_{\mathit{thermal}}$",
    "FF_fraction": "FF",
    "Jsc_mA_cm2": r"$J_{\mathit{SC}}$",
    "PCE_percent": "PCE",
    "Voc_V": r"$V_{\mathit{OC}}$",
}

PANELS = [
    {
        "letter": "a",
        "metric": "direct_R2",
        "title": "Direct-model fidelity",
        "xlabel": r"$R^2_{\mathit{Direct}}$",
        "xlim": (0.38, 1.01),
        "xticks": [0.4, 0.6, 0.8, 1.0],
        "zero": None,
        "style_method": "Direct",
    },
    {
        "letter": "b",
        "metric": "R3_p90",
        "title": "Propagation burden",
        "xlabel": r"$R_3$ (P90)",
        "xlim": (0.60, 3.30),
        "xticks": [1, 2, 3],
        "zero": None,
        "style_method": "Residual-only",
    },
    {
        "letter": "c",
        "metric": "physics_penalty_log2",
        "title": "Physics-branch distortion",
        "xlabel": (
            r"$\log_2(\mathrm{RMSE}_{\mathit{Physics}}/$"
            "\n"
            r"$\mathrm{RMSE}_{\mathit{Direct}})$"
        ),
        "xlim": (-0.20, 6.60),
        "xticks": [0, 2, 4, 6],
        "zero": 0.0,
        "style_method": "Physics-only",
    },
    {
        "letter": "d",
        "metric": "PhysicsGate_gain_log2",
        "title": "PhysicsGate gain",
        "xlabel": (
            r"$G=\log_2(\mathrm{RMSE}_{\mathit{Direct}}/$"
            "\n"
            r"$\mathrm{RMSE}_{\mathit{Gate}})$"
        ),
        "xlim": (-0.015, 0.235),
        "xticks": [0, 0.1, 0.2],
        "zero": 0.0,
        "style_method": "PhysicsGate",
    },
]

PANEL_TITLE_SIZE = 6.6
PANEL_TITLE_PAD = 2.0
TICK_SIZE = 6.2
AXIS_LABEL_SIZE = 6.4
LEGEND_SIZE = 5.8
PANEL_LETTER_SIZE = 12.0


apply_manuscript_rcparams()


def target_y_positions(summary: pd.DataFrame) -> dict[str, float]:
    ordered = summary.sort_values("target_order")
    return {
        code: float(len(ordered) - 1 - index)
        for index, code in enumerate(ordered["target_code"])
    }


def add_dataset_bands(
    ax: plt.Axes,
    positions: dict[str, float],
    summary: pd.DataFrame,
    *,
    show_labels: bool,
) -> None:
    for dataset in ("ESTM", "PV"):
        codes = summary.loc[summary["dataset"].eq(dataset), "target_code"]
        ys = [positions[code] for code in codes]
        ax.axhspan(
            min(ys) - 0.5,
            max(ys) + 0.5,
            facecolor=DATASET_BANDS[dataset],
            edgecolor="none",
            zorder=0,
        )
        if show_labels:
            add_right_group_bracket(
                ax,
                min(ys) - 0.45,
                max(ys) + 0.45,
                dataset,
                fontsize=PANEL_TITLE_SIZE,
            )
    estm_bottom = min(
        positions[code]
        for code in summary.loc[summary["dataset"].eq("ESTM"), "target_code"]
    )
    pv_top = max(
        positions[code]
        for code in summary.loc[summary["dataset"].eq("PV"), "target_code"]
    )
    add_group_separator(ax, (estm_bottom + pv_top) / 2)


def seed_offsets(target_order: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(31_415 + int(target_order))
    return rng.uniform(-0.16, 0.16, size=n)


def draw_split_distribution(
    ax: plt.Axes,
    units: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    metric: str,
    style_method: str,
    positions: dict[str, float],
) -> None:
    for row in summary.itertuples(index=False):
        subset = units.loc[
            units["dataset"].eq(row.dataset)
            & units["target_code"].eq(row.target_code)
        ].sort_values("seed")
        values = subset[metric].to_numpy(dtype=float)
        y = positions[row.target_code]
        color = METHOD_COLORS[style_method]
        light_color = METHOD_LIGHT[style_method]
        ax.scatter(
            values,
            y + seed_offsets(int(row.target_order), len(values)),
            s=SPLIT_MARKER_SIZE,
            marker="o",
            facecolor=light_color,
            edgecolor=TEXT,
            linewidth=0.16,
            alpha=0.42,
            rasterized=False,
            zorder=2,
        )
        mean = float(getattr(row, f"{metric}_mean"))
        low = float(getattr(row, f"{metric}_CI_low"))
        high = float(getattr(row, f"{metric}_CI_high"))
        ax.plot(
            [low, high],
            [y, y],
            color="#222222",
            linewidth=CI_WIDTH,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            mean,
            y,
            s=MEAN_MARKER_SIZE,
            marker="o",
            facecolor=color,
            edgecolor=TEXT,
            linewidth=MARKER_EDGE_WIDTH,
            zorder=5,
        )


def draw_literature_references(
    ax: plt.Axes,
    literature: pd.DataFrame,
    positions: dict[str, float],
) -> None:
    for row in literature.itertuples(index=False):
        y = positions[row.target_code]
        x = float(row.published_direct_R2)
        spread = float(row.published_direct_R2_spread)
        if np.isfinite(spread):
            ax.plot(
                [x - spread, x + spread],
                [y, y],
                color=LITERATURE,
                linewidth=0.75,
                solid_capstyle="round",
                zorder=5,
            )
        ax.scatter(
            x,
            y,
            s=MEAN_MARKER_SIZE,
            marker="o",
            facecolor="white",
            edgecolor=LITERATURE,
            linewidth=MARKER_EDGE_WIDTH,
            zorder=6,
        )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax._physicsgate_panel_label = label


def place_panel_labels(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    """Align panel letters with the leftmost element and title midline."""
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
            fontsize=PANEL_LETTER_SIZE,
            fontweight="bold",
            ha="left",
            va="center",
            color="#111111",
        )


def style_panel(
    ax: plt.Axes,
    panel: dict[str, object],
    positions: dict[str, float],
    summary: pd.DataFrame,
    *,
    show_labels: bool,
    show_dataset_labels: bool,
) -> None:
    add_dataset_bands(
        ax,
        positions,
        summary,
        show_labels=show_dataset_labels,
    )
    zero = panel["zero"]
    if zero is not None:
        ax.axvline(
            float(zero),
            color="#4A4A47",
            linewidth=0.8,
            linestyle=(0, (2.2, 2.0)),
            zorder=1,
        )
    ordered = summary.sort_values("target_order")
    tick_positions = [positions[code] for code in ordered["target_code"]]
    ax.set_yticks(tick_positions)
    if show_labels:
        ax.set_yticklabels([TARGET_LABELS[code] for code in ordered["target_code"]])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.5, 7.5)
    ax.set_xlim(*panel["xlim"])
    ax.set_xticks(panel["xticks"])
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    ax.set_xlabel(
        str(panel["xlabel"]),
        fontsize=AXIS_LABEL_SIZE,
        labelpad=4.0,
    )
    centered_title(
        ax,
        str(panel["title"]),
        fontsize=PANEL_TITLE_SIZE,
        pad=PANEL_TITLE_PAD,
    )
    ax.grid(False)
    style_box(ax)
    add_panel_label(ax, str(panel["letter"]))


def shared_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_COLORS["Direct"],
            markeredgecolor=TEXT,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=5.2,
            label="50-split mean",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=LITERATURE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=5.2,
            label="Literature-reported\n" + r"Direct $R^2$",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=1,
        fontsize=LEGEND_SIZE,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


def main() -> None:
    for path in (UNIT_FILE, SUMMARY_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; run build_figure5_error_propagation_analysis.py first"
            )
    units = pd.read_csv(UNIT_FILE)
    summary = pd.read_csv(SUMMARY_FILE).sort_values("target_order")
    if len(summary) != 8:
        raise RuntimeError("Figure 5 requires exactly eight ESTM/PV targets")
    seed_counts = units.groupby(["dataset", "target_code"])["seed"].nunique()
    if not seed_counts.eq(50).all():
        raise RuntimeError("Figure 5 requires exactly 50 outer splits per target")

    literature = pd.read_csv(LITERATURE_FILE)
    if set(literature["target_code"]) != set(summary["target_code"]):
        raise RuntimeError("Published Direct references do not cover all eight targets")

    positions = target_y_positions(summary)
    figure = plt.figure(figsize=(183 / 25.4, 98 / 25.4))
    grid = figure.add_gridspec(
        1,
        4,
        left=0.185,
        right=0.985,
        bottom=0.19,
        top=0.94,
        wspace=0.23,
        width_ratios=[1.22, 1.0, 1.08, 1.08],
    )
    axes = [
        figure.add_subplot(grid[0, 0]),
        figure.add_subplot(grid[0, 1]),
        figure.add_subplot(grid[0, 2]),
        figure.add_subplot(grid[0, 3]),
    ]

    for index, (ax, panel) in enumerate(zip(axes, PANELS)):
        metric = str(panel["metric"])
        draw_split_distribution(
            ax,
            units,
            summary,
            metric=metric,
            style_method=str(panel["style_method"]),
            positions=positions,
        )
        if metric == "direct_R2":
            draw_literature_references(ax, literature, positions)
        style_panel(
            ax,
            panel,
            positions,
            summary,
            show_labels=index == 0,
            show_dataset_labels=index == 3,
        )
    shared_legend(axes[0])
    place_panel_labels(figure, axes)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stems = [RESULTS_DIR / "Figure5"]
    for stem in stems:
        figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(f"Wrote Figure 5 to {RESULTS_DIR}")


if __name__ == "__main__":
    main()






