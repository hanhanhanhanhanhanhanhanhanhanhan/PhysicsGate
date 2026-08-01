"""Plot the paired ESTM/PV stabilization sensitivity as an SI figure."""

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
DATA = ROOT / "source_data" / "figureS3"
RESULTS = ROOT / "results" / "figures"
DETAIL = DATA / "SI_Stabilization_PairedSeedTarget.csv"
SUMMARY = DATA / "SI_Stabilization_TargetSummary.csv"

BLUE = "#3C5488"
ORANGE = "#D9822B"
BANDS = {"ESTM": PAPER_BAND, "PV": WHITE_BAND}
ROUTES = {
    "seebeck_uV_per_K": "inverse",
    "ZT": "forward",
    "log10_electrical_conductivity": "inverse",
    "log10_thermal_conductivity": "inverse",
    "FF_fraction": "inverse",
    "Jsc_mA_cm2": "inverse",
    "PCE_percent": "forward",
    "Voc_V": "inverse",
}
LABELS = {
    "seebeck_uV_per_K": r"$S$",
    "ZT": r"$ZT$",
    "log10_electrical_conductivity": r"$\mathit{lg}\,\sigma_{\mathit{elec}}$",
    "log10_thermal_conductivity": r"$\mathit{lg}\,\kappa_{\mathit{thermal}}$",
    "FF_fraction": "FF",
    "Jsc_mA_cm2": r"$J_{\mathit{SC}}$",
    "PCE_percent": "PCE",
    "Voc_V": r"$V_{\mathit{OC}}$",
}
PANELS = (
    {
        "letter": "a",
        "method": "Physics-only",
        "title": "Physics-only",
        "xlim": (-0.35, 8.0),
        "xticks": [0, 2, 4, 6, 8],
    },
    {
        "letter": "b",
        "method": "Residual-only",
        "title": "Residual-only",
        "xlim": (-0.28, 6.5),
        "xticks": [0, 2, 4, 6],
    },
    {
        "letter": "c",
        "method": "PhysicsGate",
        "title": "PhysicsGate",
        "xlim": (-0.025, 0.35),
        "xticks": [0, 0.1, 0.2, 0.3],
    },
)

PANEL_TITLE_SIZE = 6.6
PANEL_TITLE_PAD = 2.0
TICK_SIZE = 6.2
AXIS_LABEL_SIZE = 6.4
LEGEND_SIZE = 5.8
PANEL_LETTER_SIZE = 12.0

SENSITIVITY_XLABEL = (
    r"$S=\log_2(\mathrm{RMSE}_{\mathit{no\,range}}/"
    r"\mathrm{RMSE}_{\mathit{train\,range}})$"
)


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


def add_bands(ax: plt.Axes, show_labels: bool) -> None:
    ax.axhspan(3.5, 7.5, color=BANDS["ESTM"], zorder=0)
    ax.axhspan(-0.5, 3.5, color=BANDS["PV"], zorder=0)
    add_group_separator(ax, 3.5)
    if show_labels:
        add_right_group_bracket(
            ax, 3.55, 7.45, "ESTM", fontsize=PANEL_TITLE_SIZE
        )
        add_right_group_bracket(
            ax, -0.45, 3.45, "PV", fontsize=PANEL_TITLE_SIZE
        )


def plot_panel(
    ax: plt.Axes,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    panel: dict[str, object],
    show_labels: bool,
    show_dataset_labels: bool,
) -> None:
    letter = str(panel["letter"])
    method = str(panel["method"])
    add_bands(ax, show_dataset_labels)
    for row in summary.loc[summary["method"].eq(method)].itertuples(index=False):
        subset = detail.loc[
            detail["dataset"].eq(row.dataset)
            & detail["target_code"].eq(row.target_code)
            & detail["method"].eq(method)
        ].sort_values("seed")
        y = 7 - int(row.target_order)
        color = METHOD_COLORS[method]
        light_color = METHOD_LIGHT[method]
        rng = np.random.default_rng(82_000 + int(row.target_order) * 17)
        jitter = rng.uniform(-0.13, 0.13, len(subset))
        ax.scatter(
            subset["log2_gain_from_train_range"],
            y + jitter,
            s=SPLIT_MARKER_SIZE,
            marker="o",
            color=light_color,
            edgecolor=TEXT,
            linewidth=0.16,
            alpha=0.42,
            zorder=2,
        )
        ax.plot(
            [row.log2_gain_CI_low, row.log2_gain_CI_high],
            [y, y],
            color="#222222",
            linewidth=CI_WIDTH,
            solid_capstyle="round",
            zorder=4,
        )
        ax.scatter(
            row.log2_gain_mean,
            y,
            s=MEAN_MARKER_SIZE,
            marker="o",
            color=color,
            edgecolor=TEXT,
            linewidth=MARKER_EDGE_WIDTH,
            zorder=5,
        )
    ax.axvline(0, color="#4A4A47", linestyle=(0, (2.2, 2.0)), linewidth=0.8)
    ax.set_xlim(*panel["xlim"])
    ax.set_xticks(panel["xticks"])
    ax.set_ylim(-0.5, 7.5)
    ax.set_yticks(range(7, -1, -1))
    if show_labels:
        ordered = (
            summary[["target_code", "target_order"]]
            .drop_duplicates()
            .sort_values("target_order")
        )
        ax.set_yticklabels([LABELS[value] for value in ordered["target_code"]])
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_xlabel(
        SENSITIVITY_XLABEL,
        fontsize=AXIS_LABEL_SIZE,
        labelpad=3.0,
    )
    ax.grid(False)
    centered_title(
        ax,
        str(panel["title"]),
        fontsize=PANEL_TITLE_SIZE,
        pad=PANEL_TITLE_PAD,
    )
    style_box(ax)
    add_panel_label(ax, letter)


def add_encoding_legend(ax: plt.Axes) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=METHOD_LIGHT["Physics-only"],
            markeredgecolor=TEXT,
            markeredgewidth=0.25,
            markersize=3.8,
            label="Outer split",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            color=TEXT,
            markerfacecolor=METHOD_COLORS["Physics-only"],
            markeredgecolor=TEXT,
            markeredgewidth=MARKER_EDGE_WIDTH,
            linewidth=CI_WIDTH,
            markersize=5.2,
            label="Mean and 95% CI",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        fontsize=LEGEND_SIZE,
        handletextpad=0.4,
        borderaxespad=0.0,
    )


def main() -> None:
    detail = pd.read_csv(DETAIL)
    summary = pd.read_csv(SUMMARY)
    figure = plt.figure(figsize=(183 / 25.4, 88 / 25.4))
    grid = figure.add_gridspec(
        1, 3, left=0.185, right=0.985, bottom=0.19, top=0.94,
        wspace=0.23,
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(3)]
    for index, (panel, ax) in enumerate(zip(PANELS, axes)):
        plot_panel(
            ax,
            detail,
            summary,
            panel,
            show_labels=index == 0,
            show_dataset_labels=index == 2,
        )
    add_encoding_legend(axes[0])
    place_panel_labels(figure, axes)
    RESULTS.mkdir(parents=True, exist_ok=True)
    for stem in (RESULTS / "FigureS3",):
        figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
        figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Wrote stabilization SI figure to {RESULTS}")


if __name__ == "__main__":
    main()

