# Soberton Parish Council — SID Analysis Dashboard

Interactive dashboard presenting speed data collected by the parish's Speed Indicator Devices (SIDs).

## Data coverage

**6 March 2025 – 7 August 2025** (Soberton SID campaign period, Sites 1–4).

Data from later periods is excluded pending confirmation of device deployment locations.

## Running locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Updating the data

New data files should be placed in `inputs/` and ingested via:

```bash
python3 ingest.py
```

File naming convention: `siteN_M.csv` (e.g. `site1_3.csv`) where N is the site number
and M is the download index. Files must include a site number to be accepted.

Before ingesting post-campaign data, confirm exact deployment dates (installation and
removal) for each device and update `deployments_template.csv` accordingly.
