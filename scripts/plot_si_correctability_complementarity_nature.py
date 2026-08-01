"""Plot the SI correctability and complementarity diagnostic figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from plot_style import (
    CI_WIDTH,
    FONT_LEGEND,
    FONT_PANEL,
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
DATA_DIR = ROOT / "source_data" / "figureS2"
RESULTS_DIR = ROOT / "results" / "figures"

DETAIL_FILE = DATA_DIR / "SI_Correctability_Complementarity_SeedTarget.csv"
SUMMARY_FILE = DATA_DIR / "SI_Correctability_Complementarity_TargetSummary.csv"

DATASET_BANDS = {"ESTM": PAPER_BAND, "PV": WHITE_BAND}

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
        "metric": "residual_correctability_log2",
        "title": "Residual correctability",
        "xlabel": r"$C_{\mathit{res}}=\log_2(\mathrm{RMSE}_{\mathit{Physics}}/\mathrm{RMSE}_{\mathit{Residual}})$",
        "xlim": (-0.15, 6.65),
        "xticks": [0, 2, 4, 6],
        "reference": 0.0,
        "style_method": "Residual-only",
    },
    {
        "letter": "b",
        "metric": "branch_error_correlation",
        "title": "Branch-error correlation",
        "xlabel": r"$\rho_e=\mathrm{corr}(e_{\mathit{Direct}},e_{\mathit{Residual}})$",
        "xlim": (0.62, 1.01),
        "xticks": [0.7, 0.8, 0.9, 1.0],
        "reference": 1.0,
        "style_method": "Residual-only",
    },
    {
        "letter": "c",
        "metric": "delta_w_low_minus_high_R3",
        "title": "Gate risk response",
        "xlabel": (
            r"$\Delta w=\overline{w}_{\mathit{low}\,\mathit{R}_3}"
            r"-\overline{w}_{\mathit{high}\,\mathit{R}_3}$"
        ),
        "xlim": (-0.10, 0.36),
        "xticks": [-0.1, 0, 0.1, 0.2, 0.3],
        "reference": 0.0,
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


def positions(summary: pd.DataFrame) -> dict[str, float]:
    ordered = summary.sort_values("target_order")
    return {
        code: float(len(ordered) - 1 - index)
        for index, code in enumerate(ordered["target_code"])
    }


def add_bands(
    ax: plt.Axes,
    pos: dict[str, float],
    summary: pd.DataFrame,
    *,
    show_labels: bool,
) -> None:
    for dataset in ("ESTM", "PV"):
        codes = summary.loc[summary["dataset"].eq(dataset), "target_code"]
        ys = [pos[code] for code in codes]
        ax.axhspan(
            min(ys) - 0.5,
            max(ys) + 0.5,
            color=DATASET_BANDS[dataset],
            ec="none",
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
    add_group_separator(ax, 3.5)


def seed_offsets(order: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(27_182 + int(order))
    return rng.uniform(-0.16, 0.16, n)


def draw_panel(
    ax: plt.Axes,
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    panel: dict[str, object],
    *,
    show_labels: bool,
    show_dataset_labels: bool,
) -> None:
    pos = positions(summary)
    add_bands(ax, pos, summary, show_labels=show_dataset_labels)
    metric = str(panel["metric"])
    style_method = str(panel["style_method"])
    for row in summary.itertuples(index=False):
        subset = detailed.loc[
            detailed["dataset"].eq(row.dataset)
            & detailed["target_code"].eq(row.target_code)
        ].sort_values("seed")
        values = subset[metric].to_numpy(dtype=float)
        y = pos[row.target_code]
        color = METHOD_COLORS[style_method]
        light_color = METHOD_LIGHT[style_method]
        ax.scatter(
            values,
            y + seed_offsets(int(row.target_order), len(values)),
            s=SPLIT_MARKER_SIZE,
            marker="o",
            color=light_color,
            edgecolor=TEXT,
            linewidth=0.16,
            alpha=0.42,
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
            color=color,
            edgecolor=TEXT,
            linewidth=MARKER_EDGE_WIDTH,
            zorder=5,
        )

    ax.axvline(
        float(panel["reference"]),
        color="#4A4A47",
        linestyle=(0, (2.2, 2.0)),
        linewidth=0.8,
        zorder=1,
    )
    ordered = summary.sort_values("target_order")
    ys = [pos[code] for code in ordered["target_code"]]
    ax.set_yticks(ys)
    if show_labels:
        ax.set_yticklabels([TARGET_LABELS[code] for code in ordered["target_code"]])
    else:
        ax.set_yticklabels([])
    ax.set_ylim(-0.5, 7.5)
    ax.set_xlim(*panel["xlim"])
    ax.set_xticks(panel["xticks"])
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_xlabel(
        str(panel["xlabel"]),
        fontsize=AXIS_LABEL_SIZE,
        labelpad=4.5,
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
            markerfacecolor=METHOD_COLORS["Residual-only"],
            markeredgecolor=TEXT,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=5.2,
            label="Residual diagnostics",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_COLORS["PhysicsGate"],
            markeredgecolor=TEXT,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=5.2,
            label="PhysicsGate response",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        fontsize=LEGEND_SIZE,
        handletextpad=0.35,
        borderaxespad=0.0,
    )


def main() -> None:
    if not DETAIL_FILE.exists() or not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            "Run build_si_correctability_complementarity.py before plotting"
        )
    detailed = pd.read_csv(DETAIL_FILE)
    summary = pd.read_csv(SUMMARY_FILE).sort_values("target_order")
    if len(detailed) != 400 or len(summary) != 8:
        raise RuntimeError("SI figure requires 400 seed-target units and 8 targets")

    figure = plt.figure(figsize=(183 / 25.4, 88 / 25.4))
    grid = figure.add_gridspec(
        1,
        3,
        left=0.185,
        right=0.985,
        bottom=0.21,
        top=0.94,
        wspace=0.23,
        width_ratios=[1.18, 1.0, 1.0],
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    for index, (ax, panel) in enumerate(zip(axes, PANELS)):
        draw_panel(
            ax,
            detailed,
            summary,
            panel,
            show_labels=index == 0,
            show_dataset_labels=index == 2,
        )
    shared_legend(axes[0])
    place_panel_labels(figure, axes)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stems = [RESULTS_DIR / "FigureS2"]
    for stem in stems:
        figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote Figure S2 to {RESULTS_DIR}")


if __name__ == "__main__":
    main()

