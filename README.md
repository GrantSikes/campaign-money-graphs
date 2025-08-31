**Campaign Money Graphs** (FEC Schedule A) 

**Project site** https://grantsikes.github.io/campaign-money-graphs/ \
**Network graph** https://grantsikes.github.io/campaign-money-graphs/campaign_network.html \
**Alabama donor ZIP map** https://grantsikes.github.io/campaign-money-graphs/donor_map_AL.html 

_**Features**_  

End-to-end runner: single campaign_money_graphs.py script (self-bootstraps dependencies if needed)  \
Robust HTTP: retry/backoff, caching (requests-cache), gentle rate limiting (<1000 calls/hour)  \
Exports: tidy CSV files for candidates, committees, raw contributions, graph nodes/edges  

_**Visualization**_  

PyVis interactive network (output/campaign_network.html)  \
Leaflet ZIP-dot map for a state (defaults to Alabama) with candidate picker (output/donor_map_AL.html)  \
Configurable: filter by office, cycle, donor state, minimum amount; trim data for faster visualization  

_**Requirements**_  

Python 3.10+ (3.11/3.12)  \
FEC API key: https://api.open.fec.gov/developers/#/getting-started  \
Save it as an environment variable FEC_API_KEY.  \
The script will install missing Python packages automatically. A requirements.txt is also provided for pinned installs and CI.   

**Quickstart**  

**First**: _Clone and open the project_    

git clone https://github.com/grantsikes/campaign-money-graphs.git  \
cd campaign-money-graphs 

**Second**: _Create a virtual environment_  

macOS / Linux:  \
python -m venv .venv  \
source .venv/bin/activate  

**Third**: _Install dependencies_  

**Option A** — use the script’s self-bootstrap.  \
    When you run the script, it will install anything missing.  \
**Option B** — install upfront from requirements.txt.  \
    pip install -r requirements.txt  

**Fourth**: _Provide your FEC API key_  

macOS / Linux:  \
export FEC_API_KEY="YOUR_REAL_FEC_KEY"  

**Fifth** _Run in Terminal_  

python campaign_money_graphs.py 
  --state AL --cycle 2024 --office H \
  --min-amount 250 \
  --max-committees-per-candidate 6 \
  --max-pages-per-committee 6 \
  --edge-cap 8000 \
  --timeout 120 \
  --retries 6

_**Outputs will appear under**_  

**data/**  \
  candidates.csv  \
  committees.csv  \
  contributions_raw.csv  \
  nodes.csv  \
  edges.csv  \
**output/**  \
  top_clusters_brief.md  \
  campaign_network.html  \
  donor_map_AL.html   
\
_**Command referenc**e_  

usage: campaign_money_graphs.py  \
--state___________________________State for candidates (e.g., AL). Default: AL  \
--cycle___________________________Two-year FEC cycle (e.g., 2024, 2022). Default: 2024  \
--office___________________________H (House), S (Senate), P (President). Default: H  \
--donor-state_____________________Optional Schedule A contributor state filter  \
--min-amount_____________________Drop contributions below this amount. Default: 0  \
--top-n-clusters___________________How many clusters to summarize. Default: 10  \
--max-committees-per-candidate____Trim committees per candidate (P/A prioritized)  \
--max-pages-per-committee_________Limit paging per committee (100 rows/page)  \
--edge-cap________________________Keep only N strongest edges in visualization. Default: 5000  \
--api-key__________________________FEC API key (else read from env FEC_API_KEY)  \
--timeout__________________________HTTP timeout in seconds. Default: 90  \
--retries___________________________HTTP retries for timeouts/5xx/429. Default: 4  \
--skip-map________________________Do not build the ZIP-dot map  \
--map-state_______________________State to map ZIPs for (default AL)  \
\
_**Troubleshooting**_ 

1. No FEC_API_KEY found  \
    Set the environment variable (see Quickstart step 4) or pass --api-key "...".  
2. HTTP 429 (rate limit)  \
    The client will cool down using the server’s X-RateLimit-Reset. If it persists, reduce scope (e.g., --max-pages-per-committee 3, --max-committees-per-candidate 4) or run again later.  
3. ReadTimeout / network errors  \
    Increase --timeout (e.g., 180) and --retries (e.g., 6).  
4. Map is empty  
    The ZIP map filters to --map-state (default AL). If your donors are from other states, adjust --map-state or ensure contributions exist for that state/office/cycle.  
5. Windows can’t open files automatically  
    The script attempts open (macOS). Just open the generated HTML files from the output/ folder manually.  
6. VS Code Pylance warnings about missing stubs  
    These come from third-party libraries without type stubs (e.g., community / python-louvain). They don’t affect runtime. You can lower Pylance strictness or install types-... packages where available.  

_**Repository layout**_  

campaign-money-graphs/  
|__ campaign_money_graphs.py____ # Main script  \
|__ requirements.txt______________ # Runtime dependencies  \
|__ requirements-dev.txt__________ # Dev tooling  \
|__ pyproject.toml________________ # Lint/format config  \
|__ .pre-commit-config.yaml_______ # Pre-commit hooks  \
|__ .github/workflows_____________ # CI \
|__ data/________________________ # CSV Exports \
|__ output/______________________ # HTML & summaries  \
└── docs/____________________ # GitHub Pages site

_**Data & ethics**_  \
FEC data might include, but not limited to; errors, duplicates, or late amendments. This application applies light cleaning; Do Your Own Checks Before Drawing Conclusions.  \
Respect privacy and avoid doxxing. Donor data is public but should be handled responsibly.  \
\
_**License**_ \
Data source: Federal Election Commission (FEC) API  \
This project is not affiliated or partnered with the Federal Election Commission, AND The State of Alabama, OR The United States Government.  \
FEC Data Rremains Subject To Its Own Terms.  \
\
_**Acknowledgments**_ \
Federal Election Commission (FEC) API Team.  \
https://www.fec.gov/  \
https://api.open.fec.gov/developers/  \
https://github.com/fecgov/fec/blob/master/terms-of-service.md  \
Author: Grant Sikes
Email: grantelisikes@outlook.com \
_Version 1.13_ 
