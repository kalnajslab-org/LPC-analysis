# LPC Python

Scripts for loading LPC optical particle counter (OPC) data, merging in the
paired RS41 radiosonde record, deriving aerosol concentrations, and plotting
quicklook figures.

## Files

| File | Purpose |
|---|---|
| [read_lpcopc.py](read_lpcopc.py) | Load an `lpcopc-N.csv` file, merge in its paired `lpcrs41-N.csv`, and derive the columns needed for concentration calculations. Saves the result to NetCDF. |
| [plot_figures.py](plot_figures.py) | Build two quicklook figures from the processed data: a cumulative-concentration time series and a per-measurement differential size distribution. |

## Requirements

```
pandas
numpy
matplotlib
xarray
netCDF4
```

## Input data

Data lives in `lpc.csv/`, as pairs of CSV files sharing a run number, e.g.:

- `lpcopc-2.csv` — OPC housekeeping and aerosol bin counts
- `lpcrs41-2.csv` — RS41 radiosonde pressure/temperature/humidity, sampled ~1 Hz

Both files share the same layout: a header row, a units row (e.g. `[mA]`,
`[deg]`), then data. `read_lpcopc.py` skips the units row automatically.

The OPC reports 32 aerosol bin-count columns, `hg_*` and `lg_*` (high-gain /
low-gain channels), named by each bin's *upper* diameter edge in nm (e.g.
`hg_300` = the bin with an upper edge at 300 nm). The instrument samples
**episodically**: batches of 30-60 rows at a ~4 s cadence, separated by
multi-minute hibernation gaps.

## Processing pipeline (`read_lpcopc.py`)

Run directly (`python3 read_lpcopc.py`) or import the functions and call them
in order — each takes and returns a DataFrame:

1. **`load_lpcopc(csv_path)`** — read the OPC CSV, parse `epoch_utc` as a UTC datetime.
2. **`fill_position(df)`** — forward-fill `lat_deg` / `lon_deg` / `alt_m`, which the OPC only reports on periodic GPS updates.
3. **`add_sample_volume(df, max_dt_s=10.0, min_dt_s=2.0)`** — add `sample_volume_L`, the volume of air sampled by each row (`flow_SLPM x dt_minutes`, `dt` from consecutive `epoch` values). Rows whose `dt` is too long (a hibernation gap) or too short (the startup-transient rows right after waking, which carry an unreasonably high count relative to their short interval) are masked to `NaN` rather than estimated.
4. **`add_measurement_id(df, gap_break_s=10.0)`** — add `measurement_id`, an integer identifying which episodic batch each row belongs to (increments whenever the gap since the previous row exceeds `gap_break_s`).
5. **`matching_rs41_path(opc_csv_path)`** / **`load_lpcrs41(csv_path)`** — locate and load the paired RS41 file.
6. **`merge_rs41(df, df_rs41)`** — attach the nearest-in-time `pres_mb`, `air_temp_degC`, `rs41_rh_percent`, and `wv_mixing_ratio_ppmv` to each OPC row (`pd.merge_asof`, nearest match on `epoch`). RS41 rows marked `valid == False` (zero-filled placeholders) are excluded before matching.
7. **`add_ambient_volume(df, p_std_mb=1013.0, t_std_k=295.0)`** — convert `sample_volume_L` from the flow meter's standard reference conditions to the actual (ambient) volume sampled, via the ideal gas law: `V_ambient = V_standard * (P_std / P_ambient) * (T_ambient / T_std)`. Requires step 6 to have run first.
8. **`save_netcdf(df, csv_path)`** — write the processed DataFrame to `lpcopc-N.nc` (indexed by `epoch`; the tz-aware `epoch_utc` column is dropped since NetCDF/CF datetimes can't carry a UTC offset — `epoch` already encodes the same instant unambiguously).

Aerosol concentration (counts per volume) for any bin is then just
`df["hg_300"] / df["sample_volume_L"]` (liters) or divide by
`sample_volume_L * 1000` for cm⁻³ — this is what `plot_figures.py` does.

## Figures (`plot_figures.py`)

Run directly (`python3 plot_figures.py`) to regenerate both figures into
`figures/` and open them in interactive windows, or import `plot_figure1` /
`plot_figure2` to call them on your own DataFrame.

### Figure 1 — cumulative concentration + temperature/RH

`plot_figure1(df, save_path=...)` → `figures/figure1_cumulative_concentration.png`

Two stacked panels sharing a time axis:

- **Top:** cumulative aerosol concentration (# cm⁻³) for particles larger
  than 300, 500, 1000, and 2000 nm — the sum of all bin columns from that
  diameter rightward, divided by the ambient sampled volume.
- **Bottom** (half height): temperature and relative humidity, on a
  twin y-axis (temperature left/orange, RH right/blue — each axis's ink
  matches its line so identity doesn't depend on a legend).

Both panels break the plotted line at hibernation gaps (`break_on_gaps`)
instead of drawing a straight line across a period with no data.

### Figure 2 — differential size distribution by measurement

`plot_figure2(df, save_path=...)` → one or more
`figures/figure2_differential_distribution*.png`

One bar-histogram subplot per episodic measurement, showing dN/dD (# cm⁻³
per nm) vs. diameter (log-log axes), titled with that measurement's start
date/time. For each measurement, dN/dD is computed from the *totals*
(summed counts over summed sampled volume across the measurement's rows),
not a row-by-row average.

The panel grid is always 4 columns wide with just enough rows for the
number of measurements, capped at 24 panels per figure
(`MAX_PANELS_PER_FIGURE`). A dataset with more than 24 measurements is
split into multiple pages by calendar date (`_paginate_measurements`); a
single date with more than 24 measurements of its own is further split into
numbered parts. `plot_figure2` returns a list of `(fig, axes, save_path)`
tuples — normally one entry, more if paginated.

Bin widths (`bin_diameters_and_widths`) are derived from the gap between
each bin's upper edge and the previous one; the first bin borrows the
second bin's width (no lower edge is recorded), and the trailing duplicate
column (`lg_24000_1`, sharing `lg_24000`'s edge) is dropped since its width
is undefined.

## Example

```python
from read_lpcopc import (
    load_lpcopc, fill_position, add_sample_volume, add_measurement_id,
    matching_rs41_path, load_lpcrs41, merge_rs41, add_ambient_volume,
    save_netcdf, CSV_PATH,
)
from plot_figures import plot_figure1, plot_figure2

df = load_lpcopc(CSV_PATH)
df = fill_position(df)
df = add_sample_volume(df)
df = add_measurement_id(df)

df_rs41 = load_lpcrs41(matching_rs41_path(CSV_PATH))
df = merge_rs41(df, df_rs41)
df = add_ambient_volume(df)

save_netcdf(df, CSV_PATH)

plot_figure1(df, save_path="figure1.png")
plot_figure2(df, save_path="figure2.png")
```
