# Campaign Money Graphs (FEC Schedule A)

Python tool that:
- Pulls FEC data (candidates → committees → Schedule A individual contributions)
- Cleans & exports CSVs
- Builds an interactive donor↔candidate network (PyVis)
- Renders an Alabama donor ZIP map with clustering & candidate filter (Leaflet)

## Quickstart

1) Create & activate a virtual env, then install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

Set your key:

export FEC_API_KEY="YOUR_REAL_FEC_KEY"

Run:

python campaign_money_graphs.py \
  --state AL --cycle 2024 --office H \
  --min-amount 250 --max-committees-per-candidate 6 --max-pages-per-committee 6 \
  --edge-cap 8000 --timeout 120 --retries 6

Outputs land in output/.

Live demo (GitHub Pages):

Interactive Network Graph
Alabama Donor ZIP Map
