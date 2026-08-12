"""Shared visual language for manuscript figures generated with Python."""

from __future__ import annotations

import matplotlib as mpl


METHOD_COLORS = {
    "Direct": "#3CC47C",
    "Physics-only": "#9B9B97",
    "Residual-only": "#FFC21C",
    "Ordinary stacking": "#6F6DA8",
    "PhysResStack": "#2F75B5",
    "PhysicsGate": "#F0180A",
}

METHOD_LIGHT = {
    "Direct": "#CBEFDD",
    "Physics-only": "#DEDEDB",
    "Residual-only": "#FFE6A0",
    "Ordinary stacking": "#D6D5E7",
    "PhysResStack": "#B9D2E8",
    "PhysicsGate": "#F9B7B1",
}

DOMAIN_COLORS = {
    "SSE": "#3C5488",
    "ESTM": "#3C5488",
    "LMB": "#8A6FA5",
    "PV": "#D98A28",
}

TEXT = "#151515"
MUTED = "#6E6D68"
GRID = "#D8D6CF"
PAPER_BAND = "#F4F2EC"
WHITE_BAND = "#FFFFFF"
ZERO = "#222222"

FONT_TICK = 7.0
FONT_LABEL = 8.2
FONT_TITLE = 9.0
FONT_LEGEND = 6.8
FONT_PANEL = 10.5
SPINE_WIDTH = 0.85
TICK_WIDTH = 0.75
CI_WIDTH = 1.15
MEAN_MARKER_SIZE = 34
SPLIT_MARKER_SIZE = 7.5
MARKER_EDGE_WIDTH = 0.65
GROUP_LINESTYLE = (0, (2.0, 1.8))
GROUP_LINEWIDTH = 0.75


def apply_manuscript_rcparams() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_TICK,
            "axes.titlesize": FONT_TITLE,
            "axes.titleweight": "bold",
            "axes.labelsize": FONT_LABEL,
            "axes.linewidth": SPINE_WIDTH,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "legend.fontsize": FONT_LEGEND,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bf": "Arial:bold",
            "mathtext.sf": "Arial",
            "mathtext.cal": "Arial:italic",
            "svg.fonttype": "none",
            "svg.hashsalt": "physicsgate",
            "pdf.fonttype": 42,
        }
    )


def style_box(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color(TEXT)


def centered_title(
    ax,
    title: str,
    *,
    pad: float = 7.0,
    fontsize: float = FONT_TITLE,
) -> None:
    ax.set_title(
        title,
        loc="center",
        fontsize=fontsize,
        fontweight="bold",
        pad=pad,
        color=TEXT,
    )


def add_group_separator(ax, y: float) -> None:
    ax.axhline(
        y,
        color=TEXT,
        linewidth=GROUP_LINEWIDTH,
        linestyle=GROUP_LINESTYLE,
        zorder=1,
    )


def add_right_group_bracket(
    ax,
    y_low: float,
    y_high: float,
    label: str,
    *,
    x_inner: float = 1.018,
    x_outer: float = 1.055,
    label_x: float = 1.082,
    fontsize: float | None = None,
) -> None:
    """Draw a Figure 2/3-style long bracket outside the right plot edge."""
    transform = ax.get_yaxis_transform()
    line_kwargs = {
        "transform": transform,
        "color": TEXT,
        "linewidth": SPINE_WIDTH,
        "clip_on": False,
        "solid_capstyle": "butt",
        "zorder": 10,
    }
    ax.plot([x_inner, x_outer], [y_low, y_low], **line_kwargs)
    ax.plot([x_outer, x_outer], [y_low, y_high], **line_kwargs)
    ax.plot([x_inner, x_outer], [y_high, y_high], **line_kwargs)
    ax.text(
        label_x,
        (y_low + y_high) / 2,
        label,
        transform=transform,
        rotation=-90,
        ha="center",
        va="center",
        fontsize=FONT_LABEL if fontsize is None else fontsize,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )
