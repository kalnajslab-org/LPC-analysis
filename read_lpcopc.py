"""Read an LPC OPC housekeeping/data CSV file into a pandas DataFrame.

The CSV files (e.g. lpc.csv/lpcopc-2.csv) have the format:
    row 1: column names
    row 2: units for each column, e.g. "[mA]", "[deg]"
    row 3+: data

This script skips the units row and parses the CSV into a DataFrame,
converting epoch_utc to a proper datetime.
"""

from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).parent / "lpc.csv" / "lpcopc-2.csv"


def load_lpcopc(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Load an lpcopc CSV file into a DataFrame, skipping the units row."""
    df = pd.read_csv(csv_path, skiprows=[1])

    if "epoch_utc" in df.columns:
        df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], errors="coerce")

    return df


def matching_rs41_path(opc_csv_path: Path = CSV_PATH) -> Path:
    """Given an lpcopc-N.csv path, return the path to its paired
    lpcrs41-N.csv file (same directory, same run number)."""
    return opc_csv_path.with_name(opc_csv_path.name.replace("lpcopc", "lpcrs41"))


def load_lpcrs41(csv_path: Path) -> pd.DataFrame:
    """Load an lpcrs41 (RS41 radiosonde) CSV file into a DataFrame,
    skipping the units row."""
    df = pd.read_csv(csv_path, skiprows=[1])

    if "epoch_utc" in df.columns:
        df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], errors="coerce")

    return df


def merge_rs41(df: pd.DataFrame, df_rs41: pd.DataFrame) -> pd.DataFrame:
    """Add the closest-in-time pressure (pres_mb), temperature
    (air_temp_degC), relative humidity (rs41_rh_percent), and water
    vapor mixing ratio (wv_mixing_ratio_ppmv) from an RS41 radiosonde
    DataFrame to each row of an LPC OPC DataFrame, matched by nearest
    epoch (whole seconds).

    Rows where the RS41 reading is marked invalid (valid == False) are
    zero-filled placeholders in this data and are excluded before
    matching, so a real OPC row is never paired with bogus zeros.
    """
    rs41_fields = ["epoch", "pres_mb", "air_temp_degC", "rs41_rh_percent", "wv_mixing_ratio_ppmv"]
    rs41_valid = (
        df_rs41.loc[df_rs41["valid"], rs41_fields]
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    merged = pd.merge_asof(
        df.sort_values("epoch"),
        rs41_valid,
        on="epoch",
        direction="nearest",
    )
    return merged.sort_index()


def fill_position(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill lat_deg/lon_deg/alt_m so every row carries the most
    recent GPS fix (these columns only update periodically, leaving NaN
    in between updates)."""
    position_cols = ["lat_deg", "lon_deg", "alt_m"]
    cols = [c for c in position_cols if c in df.columns]
    df[cols] = df[cols].ffill()
    return df


def add_sample_volume(
    df: pd.DataFrame, max_dt_s: float = 10.0, min_dt_s: float = 2.0
) -> pd.DataFrame:
    """Add a "sample_volume_L" column: the volume of air (liters) sampled
    for each row, so aerosol bin counts (hg_xxxx, lg_xxxx) can later be
    converted to concentration via counts / sample_volume_L.

    Each row's counts were accumulated since the previous row, over
    ``dt = epoch[i] - epoch[i-1]`` seconds. Sampled volume for that
    interval is ``flow_SLPM * dt_minutes`` liters (flow is already in
    standard liters per minute). The first row has no preceding row to
    diff against, so its sample_volume_L is NaN.

    The instrument samples episodically: 30-60 samples at the normal
    ~4s cadence, then it hibernates for several minutes before waking for
    the next batch. Two artifacts show up around each hibernation gap,
    both excluded (masked NaN) rather than estimated:

    - The row after the gap has dt of several minutes, but its counts
      are a normal single-cycle reading, not counts accumulated over the
      whole gap — dt at face value would hugely overestimate the sampled
      volume. Any row with dt > max_dt_s is masked.
    - The 1-2 rows right after that come back at an unreasonably short
      dt (~1s vs. the normal ~4s), carrying startup-transient counts
      that aren't actually scaled down to that short interval — dividing
      by their tiny volume produces an implausible concentration spike.
      Any row with dt < min_dt_s is masked.
    """
    dt_s = df["epoch"].diff()
    volume_L = df["flow_SLPM"] * (dt_s / 60.0) 
    volume_L = volume_L.mask((dt_s > max_dt_s) | (dt_s < min_dt_s))
    df["sample_volume_L"] = volume_L.replace(0, pd.NA)

    return df


def add_ambient_volume(
    df: pd.DataFrame, p_std_mb: float = 1013.0, t_std_k: float = 295.0
) -> pd.DataFrame:
    """Convert "sample_volume_L" from standard liters (the flow meter's
    reference conditions) to ambient liters — the actual physical volume
    of air sampled at the balloon's local pressure and temperature —
    via the ideal gas law:

        V_ambient = V_standard * (P_std / P_ambient) * (T_ambient / T_std)

    Requires pres_mb and air_temp_degC already merged in (see
    merge_rs41). Aerosol concentration should be computed from the
    ambient volume, not the standard one, to reflect the actual local
    particle density.
    """
    t_ambient_k = df["air_temp_degC"] + 273.15
    df["sample_volume_L"] = (
        df["sample_volume_L"] * (p_std_mb / df["pres_mb"]) * (t_ambient_k / t_std_k)
    )
    return df


def add_measurement_id(df: pd.DataFrame, gap_break_s: float = 10.0) -> pd.DataFrame:
    """Add a "measurement_id" column (0, 1, 2, ...) identifying which
    episodic measurement batch each row belongs to. The instrument
    samples in batches of 30-60 rows separated by multi-minute
    hibernation gaps; a new measurement starts whenever the gap since
    the previous row's epoch exceeds gap_break_s."""
    gap = df["epoch"].diff() > gap_break_s
    df["measurement_id"] = gap.cumsum()
    return df


def save_netcdf(df: pd.DataFrame, csv_path: Path = CSV_PATH) -> Path:
    """Save a DataFrame to a NetCDF file next to the source CSV, indexed
    by epoch (whole seconds since 1970-01-01 UTC)."""
    df = df.copy()
    # NetCDF/CF datetimes can't carry a UTC offset; epoch (already UTC
    # seconds) is kept as the index, so drop the redundant tz-aware copy.
    df = df.drop(columns=["epoch_utc"], errors="ignore")

    ds = df.set_index("epoch").to_xarray()
    netcdf_path = csv_path.with_suffix(".nc")
    ds.to_netcdf(netcdf_path)
    return netcdf_path


if __name__ == "__main__":
    df = load_lpcopc()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {CSV_PATH}")
    print(df.head())
    print(df.dtypes)

    df = fill_position(df)
    df = add_sample_volume(df)
    df = add_measurement_id(df)

    rs41_path = matching_rs41_path(CSV_PATH)
    df_rs41 = load_lpcrs41(rs41_path)
    df = merge_rs41(df, df_rs41)
    print(f"Merged RS41 data from {rs41_path}")

    df = add_ambient_volume(df)

    netcdf_path = save_netcdf(df)
    print(f"Saved to {netcdf_path}")
