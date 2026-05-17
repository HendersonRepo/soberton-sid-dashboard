"""
Soberton SID — master database ingestion script.

Usage:
    python ingest.py                         # process inputs/, update master.csv
    python ingest.py --inputs path/to/dir    # custom inputs folder
    python ingest.py --master path/to/master.csv

Naming convention for new files:
    siteN_M.csv              e.g. site1_3.csv  (N = site number, M = download index)
    minmaxavg_siteN_M.csv    e.g. minmaxavg_site2_1.csv

How site assignment works:
    - Rows dated within the campaign period (CAMPAIGN_END) are always assigned
      via the deployment map (deployments_template.csv), regardless of filename.
      This handles devices that were moved between sites during that period.
    - Rows dated after the campaign period are assigned the site from the filename,
      which is reliable (the device was genuinely at that site when downloaded).
"""

import argparse
import glob
import re
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
INPUTS_DIR = BASE_DIR / "inputs"
MASTER_CSV = BASE_DIR / "inputs" / "master.csv"
MINMAX_CSV = BASE_DIR / "inputs" / "master_minmaxavg.csv"
DEPLOY_CSV = BASE_DIR / "deployments_template.csv"

MASTER_COLS = ["Date", "Direction", "Number of measurements",
               "Number of vehicles", "Average speed", "Maximum speed", "Location"]

SITE_LABELS = {
    1: "Site1", 2: "Site2", 3: "Site3", 4: "Site4",
    5: "Site5", 6: "Site6", 7: "Site7", 8: "Site8", 9: "Site9",
}

# Last date of the multi-site campaign period. Rows on or before this date
# are assigned via the deployment map; rows after this date use the filename.
CAMPAIGN_END = pd.Timestamp("2025-08-07")


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", dayfirst=True, parse_dates=["Date"],
                     on_bad_lines="skip")
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


def is_minmaxavg(name: str) -> bool:
    return name.startswith("minmaxavg_")


def site_from_filename(name: str) -> int | None:
    """Return site number from filename, or None if not determinable."""
    n = name.lower()
    if n.startswith("sobertonhighstreet"):
        return 4
    m = re.match(r"site(\d+)_", n)
    return int(m.group(1)) if m else None


def load_deployment_map(path: Path) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path, parse_dates=["period_start", "period_end"])
    df = df[df["site_number"].notna()]
    df["site_number"] = df["site_number"].astype(int)
    return df[["period_start", "period_end", "site_number"]].to_dict("records")


def assign_by_map(df: pd.DataFrame, deploy_map: list[dict]) -> pd.DataFrame:
    """Assign Location from date-range map. Rows outside every range are dropped."""
    parts = []
    for entry in deploy_map:
        mask = (df["Date"] >= entry["period_start"]) & (df["Date"] <= entry["period_end"])
        chunk = df[mask].copy()
        chunk["Location"] = SITE_LABELS[entry["site_number"]]
        parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=df.columns.tolist() + ["Location"])
    return pd.concat(parts, ignore_index=True)


def assign_site(df: pd.DataFrame, filename_site: int | None,
                deploy_map: list[dict]) -> pd.DataFrame:
    """
    Split df into campaign-period rows (use map) and post-campaign rows (use filename).
    Returns combined result with Location assigned, or empty df if site unknown.
    """
    campaign_rows = df[df["Date"] <= CAMPAIGN_END]
    post_rows     = df[df["Date"] >  CAMPAIGN_END]

    parts = []

    # Campaign period — always use deployment map
    if not campaign_rows.empty and deploy_map:
        parts.append(assign_by_map(campaign_rows, deploy_map))
    elif not campaign_rows.empty:
        # No map available — skip campaign rows (site unknown)
        pass

    # Post-campaign — use filename
    if not post_rows.empty:
        if filename_site is None:
            pass  # Cannot determine site — skip
        else:
            chunk = post_rows.copy()
            chunk["Location"] = SITE_LABELS[filename_site]
            parts.append(chunk)

    if not parts:
        return pd.DataFrame(columns=df.columns.tolist() + ["Location"])
    return pd.concat(parts, ignore_index=True)


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=MASTER_COLS)
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip()
    # Ensure no pre-existing duplicates
    return df.drop_duplicates(subset=["Date", "Direction", "Location"], keep="first")


def load_existing_mm(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["Date", "Min", "Max", "Average", "Location"])
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = df.columns.str.strip()
    return df.drop_duplicates(subset=["Date", "Location"], keep="first")


# ── Main ingestion ─────────────────────────────────────────────────────────────

def ingest(inputs_dir: Path, master_path: Path, minmax_path: Path):
    deploy_map  = load_deployment_map(DEPLOY_CSV)
    existing    = load_existing(master_path)
    existing_mm = load_existing_mm(minmax_path)

    new_rows    = []
    new_mm_rows = []
    skipped     = []
    added_counts = {}

    skip_files = {"master.csv", "master_minmaxavg.csv", "14thJuly2025.csv",
                  "deployments_template.csv"}

    for fpath in sorted(glob.glob(str(inputs_dir / "*.csv"))):
        name = Path(fpath).name

        if name in skip_files:
            continue

        # Reject files without a recognisable site in the filename
        if name.startswith("unknownsite") or name.startswith("multiple"):
            msg = (
                "NAMING CONVENTION: 'multiple_' files are no longer accepted — "
                "new files must include a site number, e.g. site1_3.csv"
                if name.startswith("multiple") else
                "NAMING CONVENTION: filename must include a site number, e.g. site1_3.csv"
            )
            skipped.append((name, msg))
            continue

        try:
            df = read_raw(Path(fpath))
        except Exception as e:
            skipped.append((name, f"read error: {e}"))
            continue

        # ── minmaxavg files (no Direction column) ────────────────────────
        if is_minmaxavg(name):
            inner = name[len("minmaxavg_"):]
            if inner.startswith(("unknownsite", "unknown", "multiple")):
                skipped.append((name, "NAMING CONVENTION: filename must include a site number"))
                continue
            site_num = site_from_filename(inner)
            if site_num is None:
                skipped.append((name, "cannot determine site from filename"))
                continue
            # minmaxavg files are single-site downloads — no campaign-period ambiguity
            df_mm = df.copy()
            df_mm["Location"] = SITE_LABELS[site_num]
            if df_mm.empty:
                continue
            new_mm_rows.append(df_mm[["Date", "Min", "Max", "Average", "Location"]])
            added_counts[name] = len(df_mm)
            continue

        # ── Standard direction files ──────────────────────────────────────
        site_num = site_from_filename(name)
        # site_num may be None for files we can't parse — assign_site handles that
        df_site = assign_site(df, site_num, deploy_map)

        if df_site.empty:
            skipped.append((name, "no rows after site assignment (check deployment map)"))
            continue

        cols = [c for c in MASTER_COLS if c in df_site.columns]
        new_rows.append(df_site[cols])
        added_counts[name] = len(df_site)

    # ── Write master ──────────────────────────────────────────────────────
    if new_rows:
        combined = pd.concat([existing] + new_rows, ignore_index=True)
        before   = len(combined)
        combined = combined.drop_duplicates(subset=["Date", "Direction", "Location"], keep="first")
        combined = combined.sort_values(["Location", "Date", "Direction"]).reset_index(drop=True)
        combined.to_csv(master_path, index=False)
        total_new    = len(combined) - len(existing)
        dupes_dropped = before - len(combined)
        print(f"\n✓ Master → {master_path.name}: "
              f"{len(existing):,} existing + {total_new:,} new = {len(combined):,} rows"
              f"{f'  ({dupes_dropped:,} duplicates removed)' if dupes_dropped else ''}")
    else:
        print("\n✓ Master unchanged — no new rows.")

    if new_mm_rows:
        combined_mm = pd.concat([existing_mm] + new_mm_rows, ignore_index=True)
        before_mm   = len(combined_mm)
        combined_mm = combined_mm.drop_duplicates(subset=["Date", "Location"], keep="first")
        combined_mm = combined_mm.sort_values(["Location", "Date"]).reset_index(drop=True)
        combined_mm.to_csv(minmax_path, index=False)
        total_new_mm = len(combined_mm) - len(existing_mm)
        dupes_mm     = before_mm - len(combined_mm)
        print(f"✓ MinMaxAvg → {minmax_path.name}: "
              f"{len(existing_mm):,} existing + {total_new_mm:,} new = {len(combined_mm):,} rows"
              f"{f'  ({dupes_mm:,} duplicates removed)' if dupes_mm else ''}")

    print("\n── Per-file summary ──────────────────────────────────────────")
    for n, count in sorted(added_counts.items()):
        print(f"  {n:<45} +{count:>6} rows")

    if skipped:
        print("\n── Skipped ───────────────────────────────────────────────────")
        for n, reason in skipped:
            print(f"  {n:<45} {reason}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Ingest SID raw CSV files into master.")
    p.add_argument("--inputs", default=str(INPUTS_DIR))
    p.add_argument("--master", default=str(MASTER_CSV))
    p.add_argument("--minmax", default=str(MINMAX_CSV))
    args = p.parse_args()
    ingest(Path(args.inputs), Path(args.master), Path(args.minmax))
