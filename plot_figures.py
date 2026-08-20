"""Figures for the LPC OPC data.

Figure 1: time series of cumulative aerosol concentration for particles
larger than 300, 500, 1000, and 2000 nm.

Figure 2: differential size distribution (dN/dD), averaged over each
episodic measurement, one subplot per measurement in a dynamic grid.

Figure 3: housekeeping time series (temperatures, voltages, currents,
flow), one panel per group, stacked vertically.
"""

from itertools import groupby
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from read_lpcopc import (
    CSV_PATH,
    add_ambient_volume,
    add_measurement_id,
    add_sample_volume,
    fill_position,
    flag_bad_flow,
    load_lpcopc,
    load_lpcrs41,
    matching_rs41_path,
    merge_rs41,
)

OUT_DIR = Path(__file__).parent / "figures"

# Measurements are separated by hibernation gaps of several minutes; a gap
# longer than this (seconds) breaks the plotted line rather than
# connecting straight across the hibernation period.
GAP_BREAK_S = 10.0

# Categorical palette (light mode), fixed order — dataviz skill reference palette.
COLORS = {
    300: "#2a78d6",   # blue
    500: "#eb6834",   # orange
    1000: "#1baf7a",  # aqua
    2000: "#eda100",  # yellow
}

# Same palette, as an ordered list for figures with an arbitrary number of
# series (assign in this fixed order — never cycle/re-sort by rank).
CATEGORICAL_COLORS = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Reference conditions the flow meter reports flow_SLPM at — must match
# add_ambient_volume's defaults in read_lpcopc.py.
P_STD_MB = 1013.0
T_STD_K = 295.0

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def bin_columns(df) -> list[str]:
    """All aerosol bin count columns, in their original (increasing
    diameter) CSV order."""
    return [c for c in df.columns if c.startswith(("hg_", "lg_"))]


def cumulative_concentration(df, diameter_nm: int) -> "pd.Series":
    """Cumulative aerosol concentration (# / cm^3) for particles with
    diameter >= diameter_nm: the sum of all bin-count columns from the
    matching bin edge rightward, divided by the sampled volume
    (sample_volume_L converted from liters to cm^3: 1 L = 1000 cm^3)."""
    cols = bin_columns(df)
    start_col = next(
        c for c in cols if c in (f"hg_{diameter_nm}", f"lg_{diameter_nm}")
    )
    start = cols.index(start_col)
    counts = df[cols[start:]].sum(axis=1)
    return counts / (df["sample_volume_L"] * 1000.0)


def bin_diameters_and_widths(df):
    """Upper-edge diameters (nm) and bin widths (nm) for each aerosol bin
    column, in increasing-diameter order.

    Column names give each bin's *upper* edge (e.g. hg_300 -> 300 nm); a
    bin's width is taken as the gap to the previous bin's upper edge. The
    first bin has no previous edge to measure from, so its width is
    assumed equal to the second bin's (constant spacing at the low end).
    The data ends with two columns sharing the same upper edge
    (lg_24000, lg_24000_1) — the second has no defined width, so it's
    dropped from the distribution.

    Returns (cols, diameters, widths) — the surviving column names and
    two float arrays of equal length.
    """
    cols = bin_columns(df)
    diameters = [int(c.split("_")[1]) for c in cols]

    while len(diameters) >= 2 and diameters[-1] == diameters[-2]:
        cols, diameters = cols[:-1], diameters[:-1]

    widths = [diameters[1] - diameters[0]]
    widths += [diameters[i] - diameters[i - 1] for i in range(1, len(diameters))]

    return cols, np.array(diameters, dtype=float), np.array(widths, dtype=float)


def break_on_gaps(x, y, gap_mask):
    """Insert a NaN just before each gap-start point so the plotted line
    lifts the pen across a hibernation gap instead of connecting straight
    across it, while leaving every real (x, y) point intact."""
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    gap_mask = np.asarray(gap_mask)

    insert_at = np.flatnonzero(gap_mask)
    x_out = np.insert(x, insert_at, x[insert_at])
    y_out = np.insert(y, insert_at, np.nan)
    return x_out, y_out


def _style_spines(ax):
    """Heavy box-border chrome (all four spines, thick and neutral)."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(BASELINE)
        spine.set_linewidth(1.8)


def _style_ticks(ax, axis="both", color=INK_PRIMARY):
    """Heavy, bold tick chrome on the given axis ('x', 'y', or 'both')."""
    ax.tick_params(axis=axis, colors=color, width=1.8, length=7, labelsize=11)
    labels = []
    if axis in ("x", "both"):
        labels += ax.get_xticklabels()
    if axis in ("y", "both"):
        labels += ax.get_yticklabels()
    for tick_label in labels:
        tick_label.set_fontweight("bold")


def plot_figure1(df, save_path: Path = None):
    diameters = [300, 500, 1000, 2000]

    # Rows starting a new batch after a hibernation gap (used only to
    # break the plotted line; the data values themselves are unaffected).
    gap_mask = (df["epoch"].diff() > GAP_BREAK_S).to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 8.25),
        facecolor=SURFACE,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 0.5], "hspace": 0.08},
    )
    ax1.set_facecolor(SURFACE)
    ax2.set_facecolor(SURFACE)

    # --- Top: cumulative aerosol concentration ---------------------------
    for d in diameters:
        conc = cumulative_concentration(df, d)
        x, y = break_on_gaps(df["epoch_utc"], conc, gap_mask)
        ax1.plot(
            x,
            y,
            color=COLORS[d],
            linewidth=2,
            solid_capstyle="round",
            label=f"d > {d} nm",
        )

    ax1.set_yscale("log")
    ax1.set_ylabel(
        "Cumulative concentration (# cm$^{-3}$)",
        color=INK_PRIMARY,
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_title(
        "Cumulative aerosol concentration", color=INK_PRIMARY, fontsize=14, fontweight="bold"
    )
    ax1.grid(True, which="major", axis="y", color=GRIDLINE, linewidth=1.2)
    ax1.grid(False, axis="x")
    _style_spines(ax1)
    _style_ticks(ax1)

    legend = ax1.legend(
        #title="Diameter threshold",
        frameon=True,
        labelcolor=INK_PRIMARY,
    )
    legend.get_title().set_color(INK_PRIMARY)

    # --- Bottom: temperature and relative humidity ------------------------
    temp_color = COLORS[500]  # orange
    rh_color = COLORS[300]  # blue

    x_temp, y_temp = break_on_gaps(df["epoch_utc"], df["air_temp_degC"], gap_mask)
    ax2.plot(x_temp, y_temp, color=temp_color, linewidth=2, solid_capstyle="round")
    ax2.set_ylabel("Temperature (°C)", color=temp_color, fontsize=12, fontweight="bold")

    ax2_rh = ax2.twinx()
    x_rh, y_rh = break_on_gaps(df["epoch_utc"], df["rs41_rh_percent"], gap_mask)
    ax2_rh.plot(x_rh, y_rh, color=rh_color, linewidth=2, solid_capstyle="round")
    ax2_rh.set_ylabel("Relative humidity (%)", color=rh_color, fontsize=12, fontweight="bold")

    ax2.set_xlabel("Time (UTC)", color=INK_PRIMARY, fontsize=12, fontweight="bold")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for tick_label in ax2.get_xticklabels():
        tick_label.set_rotation(0)
        tick_label.set_horizontalalignment("center")

    ax2.grid(True, which="major", axis="y", color=GRIDLINE, linewidth=1.2)
    ax2.grid(False, axis="x")
    _style_spines(ax2)
    _style_ticks(ax2, axis="x", color=INK_PRIMARY)
    _style_ticks(ax2, axis="y", color=temp_color)

    # twinx doesn't own its own spines, so style them separately, in the
    # RH line's color for consistency (a dual-axis mitigation: each
    # axis's ink matches the series it measures instead of a legend).
    ax2_rh.spines["right"].set_color(BASELINE)
    ax2_rh.spines["right"].set_linewidth(1.8)
    _style_ticks(ax2_rh, axis="y", color=rh_color)

    fig.align_ylabels([ax1, ax2])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)

    return fig, (ax1, ax2)


MAX_PANELS_PER_FIGURE = 24
GRID_NCOLS = 4  # panel grid is always this many columns wide; rows scale with measurement count


def _grid_shape(n_panels: int) -> tuple[int, int]:
    """(nrows, ncols) for a grid of n_panels (<= MAX_PANELS_PER_FIGURE)
    that's exactly as tall as needed, fixed at GRID_NCOLS wide."""
    ncols = min(GRID_NCOLS, max(n_panels, 1))
    nrows = -(-n_panels // ncols)  # ceil division
    return nrows, ncols


def _paginate_measurements(df, measurement_ids, max_panels: int = MAX_PANELS_PER_FIGURE):
    """Split measurement_ids into pages of at most max_panels each.

    If everything fits in one page, returns [(None, measurement_ids)]
    (no page label needed). Otherwise splits by calendar date (each date
    its own page); a single date with more measurements than max_panels
    is further split into numbered parts.
    """
    if len(measurement_ids) <= max_panels:
        return [(None, measurement_ids)]

    measurement_date = {
        mid: df.loc[df["measurement_id"] == mid, "epoch_utc"].iloc[0].date()
        for mid in measurement_ids
    }

    pages = []
    for date, group in groupby(measurement_ids, key=lambda mid: measurement_date[mid]):
        day_measurements = list(group)
        n_parts = -(-len(day_measurements) // max_panels)  # ceil division
        for part in range(n_parts):
            sub = day_measurements[part * max_panels : (part + 1) * max_panels]
            label = str(date) if n_parts == 1 else f"{date}_part{part + 1}"
            pages.append((label, sub))
    return pages


def _page_save_path(save_path: Path, label: str) -> Path:
    if label is None:
        return save_path
    return save_path.with_name(f"{save_path.stem}_{label}{save_path.suffix}")


def _plot_distribution_page(df, cols, left_edges, widths, measurement_ids, title):
    valid = df["sample_volume_L"].notna()
    nrows, ncols = _grid_shape(len(measurement_ids))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 3.7 * nrows + 1.6),
        facecolor=SURFACE,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    for i, ax in enumerate(axes.flat):
        ax.set_facecolor(SURFACE)

        if i >= len(measurement_ids):
            ax.axis("off")
            continue

        measurement = df[(df["measurement_id"] == measurement_ids[i]) & valid]

        # Measurement-averaged concentration per bin: total counts over
        # total sampled volume (not a row-by-row mean), then normalized
        # by bin width to get the differential distribution dN/dD.
        total_counts = measurement[cols].sum(axis=0).to_numpy(dtype=float)
        total_volume_cm3 = measurement["sample_volume_L"].sum() * 1000.0
        dN_dD = total_counts / (total_volume_cm3 * widths)

        ax.bar(
            left_edges,
            dN_dD,
            width=widths,
            align="edge",
            color=COLORS[300],
            edgecolor=SURFACE,
            linewidth=0.3,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        start_time = measurement["epoch_utc"].iloc[0]
        ax.set_title(
            start_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            fontsize=10,
            color=INK_PRIMARY,
        )

        ax.grid(True, which="major", color=GRIDLINE, linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(BASELINE)
        # sharex/sharey only auto-label the grid's bottom/left edge, but
        # with unused panels hidden that isn't necessarily this panel's
        # row/column — force labels on every populated panel instead.
        ax.tick_params(colors=INK_MUTED, labelsize=8, labelbottom=True, labelleft=True)

    fig.supxlabel("Diameter (nm)", color=INK_SECONDARY, fontsize=12, fontweight="bold")
    fig.supylabel(
        "dN/dD (# cm$^{-3}$ nm$^{-1}$)",
        color=INK_SECONDARY,
        fontsize=12,
        fontweight="bold",
    )
    fig.suptitle(title, color=INK_PRIMARY, fontsize=15, fontweight="bold")

    fig.tight_layout(rect=[0.02, 0.02, 1, 0.96])

    return fig, axes


def plot_figure2(df, save_path: Path = None, gap_break_s: float = GAP_BREAK_S):
    """Differential size distribution dN/dD, averaged over each episodic
    measurement, one subplot per measurement.

    The panel grid is sized to exactly fit the number of measurements
    (fixed at 4 columns wide, rows scale with measurement count), capped
    at MAX_PANELS_PER_FIGURE (24) per figure. A dataset with more
    measurements than that is split into multiple pages by calendar
    date, each returned/saved separately.

    Returns a list of (fig, axes, save_path_used) tuples — one entry
    unless the data needed to be paginated.
    """
    if "measurement_id" not in df.columns:
        df = add_measurement_id(df, gap_break_s=gap_break_s)

    cols, diameters, widths = bin_diameters_and_widths(df)
    left_edges = diameters - widths

    measurement_ids = sorted(df["measurement_id"].unique())
    pages = _paginate_measurements(df, measurement_ids)

    results = []
    for label, page_measurement_ids in pages:
        title = "Differential size distribution by measurement"
        if label is not None:
            title += f" — {label.replace('_', ' ')}"

        fig, axes = _plot_distribution_page(
            df, cols, left_edges, widths, page_measurement_ids, title
        )

        page_path = _page_save_path(save_path, label) if save_path else None
        if page_path:
            fig.savefig(page_path, dpi=150, facecolor=SURFACE)

        results.append((fig, axes, page_path))

    return results


def ambient_flow_slpm(df, p_std_mb: float = P_STD_MB, t_std_k: float = T_STD_K) -> "pd.Series":
    """Ambient-equivalent volumetric flow rate (L/min): flow_SLPM (at the
    flow meter's standard reference conditions) converted to the
    balloon's local pressure/temperature via the ideal gas law — the same
    factor add_ambient_volume applies to sample_volume_L, but applied to
    the flow rate directly rather than an already-integrated volume, so
    it isn't affected by that column's hibernation-gap/startup masking.

    Requires pres_mb and air_temp_degC (see merge_rs41 in read_lpcopc.py).
    """
    t_ambient_k = df["air_temp_degC"] + 273.15
    return df["flow_SLPM"] * (p_std_mb / df["pres_mb"]) * (t_ambient_k / t_std_k)


def plot_figure3(df, save_path: Path = None):
    """Housekeeping time series: temperatures, voltages, currents, and
    flow (standard vs. ambient-equivalent), one panel per group."""
    gap_mask = (df["epoch"].diff() > GAP_BREAK_S).to_numpy()

    panels = [
        (
            "Temperatures",
            "Temperature (°C)",
            [
                ("pump1_T_degC", "pump 1"),
                ("pump2_T_degC", "pump 2"),
                ("laser_T_degC", "laser"),
                ("pcb_T_degC", "PCB"),
                ("inlet_T_degC", "inlet"),
            ],
        ),
        (
            "Voltages",
            "Voltage (V)",
            [
                ("pha_12V_V", "PHA 12V"),
                ("pha_3V3_V", "PHA 3.3V"),
                ("cpu_V_V", "CPU"),
                ("input_V_V", "input"),
            ],
        ),
        (
            "Currents",
            "Current (mA)",
            [
                ("pump1_I_mA", "pump 1"),
                ("pump2_I_mA", "pump 2"),
                ("pha_I_mA", "PHA"),
            ],
        ),
    ]

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 13),
        facecolor=SURFACE,
        sharex=True,
        gridspec_kw={"hspace": 0.45},
    )

    for ax, (title, ylabel, series) in zip(axes[:3], panels):
        ax.set_facecolor(SURFACE)
        for i, (col, label) in enumerate(series):
            x, y = break_on_gaps(df["epoch_utc"], df[col], gap_mask)
            ax.plot(
                x,
                y,
                color=CATEGORICAL_COLORS[i],
                linewidth=2,
                solid_capstyle="round",
                label=label,
            )
        ax.set_ylabel(ylabel, color=INK_PRIMARY, fontsize=12, fontweight="bold")
        # pad clears the panel's own top spine and (with the increased
        # hspace above) the previous panel's bottom spine, which
        # otherwise cuts through the title's letterforms.
        ax.set_title(title, color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=10)
        ax.grid(True, which="major", axis="y", color=GRIDLINE, linewidth=1.2)
        ax.grid(False, axis="x")
        _style_spines(ax)
        _style_ticks(ax)
        legend = ax.legend(
            frameon=True, labelcolor=INK_PRIMARY, fontsize=9, ncols=len(series), loc="upper left"
        )
        legend.get_frame().set_edgecolor(BASELINE)

    # --- Flow: standard (as reported) vs. ambient-equivalent -------------
    ax_flow = axes[3]
    ax_flow.set_facecolor(SURFACE)

    x_std, y_std = break_on_gaps(df["epoch_utc"], df["flow_SLPM"], gap_mask)
    ax_flow.plot(
        x_std,
        y_std,
        color=CATEGORICAL_COLORS[0],
        linewidth=2,
        solid_capstyle="round",
        label="standard (SLPM)",
    )

    x_amb, y_amb = break_on_gaps(df["epoch_utc"], ambient_flow_slpm(df), gap_mask)
    ax_flow.plot(
        x_amb,
        y_amb,
        color=CATEGORICAL_COLORS[1],
        linewidth=2,
        solid_capstyle="round",
        label="ambient (L/min)",
    )

    ax_flow.set_ylabel("Flow (L/min)", color=INK_PRIMARY, fontsize=12, fontweight="bold")
    ax_flow.set_title("Flow", color=INK_PRIMARY, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax_flow.set_xlabel("Time (UTC)", color=INK_PRIMARY, fontsize=12, fontweight="bold")
    ax_flow.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for tick_label in ax_flow.get_xticklabels():
        tick_label.set_rotation(0)
        tick_label.set_horizontalalignment("center")

    ax_flow.grid(True, which="major", axis="y", color=GRIDLINE, linewidth=1.2)
    ax_flow.grid(False, axis="x")
    _style_spines(ax_flow)
    _style_ticks(ax_flow)
    legend = ax_flow.legend(
        frameon=True, labelcolor=INK_PRIMARY, fontsize=9, ncols=2, loc="upper left"
    )
    legend.get_frame().set_edgecolor(BASELINE)

    fig.suptitle("Instrument housekeeping", color=INK_PRIMARY, fontsize=15, fontweight="bold")
    fig.align_ylabels(axes)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        fig.savefig(save_path, dpi=150, facecolor=SURFACE)

    return fig, axes


if __name__ == "__main__":
    df = load_lpcopc()
    df = fill_position(df)
    df = flag_bad_flow(df)
    df = add_sample_volume(df)
    df = add_measurement_id(df)

    df_rs41 = load_lpcrs41(matching_rs41_path(CSV_PATH))
    df = merge_rs41(df, df_rs41)
    df = add_ambient_volume(df)

    OUT_DIR.mkdir(exist_ok=True)

    fig1_path = OUT_DIR / "figure1_cumulative_concentration.png"
    plot_figure1(df, save_path=fig1_path)
    print(f"Saved {fig1_path}")

    fig2_path = OUT_DIR / "figure2_differential_distribution.png"
    for _fig, _axes, saved_path in plot_figure2(df, save_path=fig2_path):
        print(f"Saved {saved_path}")

    fig3_path = OUT_DIR / "figure3_housekeeping.png"
    plot_figure3(df, save_path=fig3_path)
    print(f"Saved {fig3_path}")

    plt.show()  # open each figure in its own interactive window
