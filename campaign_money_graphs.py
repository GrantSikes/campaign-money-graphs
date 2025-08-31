#!/usr/bin/env python3
# ruff: noqa: E402
"""
Campaign Money Graphs — single-file runner (self-bootstrapping, with AL donor ZIP map)

What it does end-to-end:
- Bootstraps dependencies (pip installs if missing)
- Robust HTTP with retries/backoff; gentle rate limiting under <1000 calls/hr
- Fetches FEC data: candidates -> committees -> Schedule A (individuals)
- Cleans and quotes CSVs (prevents CSV line break / quote issues)
- Builds a donor <-> candidate weighted network and detects communities
- Exports:
    data/candidates.csv
    data/committees.csv
    data/contributions_raw.csv
    data/nodes.csv
    data/edges.csv
    output/top_clusters_brief.md
    output/campaign_network.html  (interactive network)
    output/donor_map_AL.html      (AL ZIP-dot map with clickable candidate list)

Provide your FEC key (choose ONE):
  A) CLI flag: --api-key "YOUR_REAL_FEC_KEY"   (recommended)
  B) Env var : export FEC_API_KEY="YOUR_REAL_FEC_KEY"
  C) (not recommended) paste below into HARDCODED_FEC_API_KEY
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass

# ============  (C) OPTIONAL: paste your key here  ============
HARDCODED_FEC_API_KEY = ""  # <-- PASTE KEY HERE only if you can't use A/B
# =============================================================

# -------------------- dependency bootstrap -------------------- #
REQUIRED = [
    ("requests", "requests"),
    ("requests_cache", "requests-cache"),
    ("pandas", "pandas>=2.0.0"),
    ("networkx", "networkx"),
    ("tqdm", "tqdm"),
    ("pyvis", "pyvis"),
    ("community", "python-louvain"),  # community_louvain
    ("urllib3", "urllib3"),
    ("pgeocode", "pgeocode"),  # ZIP -> lat/lon
]


def _pip_install(pkg: str) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", pkg]
    # Prefer capture_output over PIPEs (Ruff UP022) and don't crash the run
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            print(f"[warn] pip install failed for {pkg}\n{completed.stderr}")
    except Exception as e:
        print(f"[warn] pip install crashed for {pkg}: {e}")


def ensure_deps() -> None:
    to_install = []
    for mod, pipname in REQUIRED:
        try:
            __import__(mod)
        except Exception:
            to_install.append(pipname)
    if to_install:
        print(f"[setup] Installing missing packages: {', '.join(to_install)}")
        for pipname in to_install:
            _pip_install(pipname)


ensure_deps()

# now safe to import
import networkx as nx
import pandas as pd
import requests
import requests_cache
from tqdm import tqdm

try:
    import community as community_louvain  # python-louvain

    _HAS_LOUVAIN = True
except Exception:
    _HAS_LOUVAIN = False

import pgeocode
from pyvis.network import Network
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

FEC_BASE = "https://api.open.fec.gov/v1"
DEFAULT_TWO_YEAR = 2024


# -------------------- path hygiene -------------------- #
def chdir_to_script():
    """Run from this file's folder to avoid path weirdness."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
    except Exception:
        pass


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)


chdir_to_script()
ensure_dirs()


# -------------------- utils -------------------- #
def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.upper()
    s = re.sub(r"[^A-Z0-9\s\-']", " ", s)
    s = re.sub(r"\b(MR|MRS|MS|DR|JR|SR|II|III|IV)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip newlines/oddballs so CSV exports are bulletproof."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == "object":
            out[col] = (
                out[col]
                .astype(str)
                .str.replace(r"[\r\n]+", " ", regex=True)  # kill line breaks
                .str.replace("\u0000", " ")  # any stray nulls
            )
    return out


def donor_key(row: dict) -> str:
    cid = row.get("contributor_id")
    if cid:
        return f"ID:{cid}"
    nm = normalize_name(row.get("contributor_name", "") or "")
    z = str(row.get("contributor_zip", "") or "").strip()
    return f"NMZ:{nm}|{z}"


# -------------------- rate limiter -------------------- #
class HourlyRateLimiter:
    """
    Caps calls to stay under 1000/hr. Default ~900/hr (15 rpm).
    Also backs off if FEC replies 429 with reset hints.
    """

    def __init__(self, max_per_hour: int = 950, rpm: int = 15):
        self.max_per_hour = max_per_hour
        self.rpm = rpm
        self.calls = deque()

    def wait(self, remaining_header: str | None = None, reset_in_sec: int | None = None):
        now = time.time()
        # slide window
        while self.calls and now - self.calls[0] > 3600:
            self.calls.popleft()

        # hourly cap
        if len(self.calls) >= self.max_per_hour:
            sleep_for = 3600 - (now - self.calls[0]) + 1
            print(f"[rate] Hourly cap hit. Sleeping {int(sleep_for)}s…")
            time.sleep(max(1, sleep_for))

        # rpm pacing
        if self.calls:
            delta = now - self.calls[-1]
            need = 60.0 / float(self.rpm)
            if delta < need:
                time.sleep(need - delta)

        # server-driven cooldown
        if remaining_header is not None:
            try:
                rem = int(remaining_header)
                if rem <= 0 and reset_in_sec:
                    print(f"[rate] Server reset in {reset_in_sec}s. Cooling down…")
                    time.sleep(max(1, int(reset_in_sec)))
            except Exception:
                pass

        self.calls.append(time.time())


# -------------------- HTTP client -------------------- #
def make_session(
    cache_name="fec_cache", expire_after=24 * 3600, retries: int = 4, backoff: float = 1.5
):
    sess = requests_cache.CachedSession(
        cache_name=cache_name,
        backend="sqlite",
        expire_after=expire_after,
        allowable_methods=("GET",),
        allowable_codes=(200,),
    )
    # Robust retry for timeouts & 5xx & 429 (requests-cache still respects adapters)
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=backoff,  # exponential (1.5, 3.0, 4.5, …)
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=40)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


@dataclass
class FECClient:
    api_key: str
    session: requests.Session
    limiter: HourlyRateLimiter
    timeout: int = 90  # seconds

    def _get(self, endpoint: str, params: dict) -> dict:
        url = f"{FEC_BASE}/{endpoint.lstrip('/')}"
        p = dict(params or {})
        p["api_key"] = self.api_key

        self.limiter.wait()  # pre-wait

        # main request
        resp = self.session.get(url, params=p, timeout=self.timeout)
        if not getattr(resp, "from_cache", False):
            reset = resp.headers.get("X-RateLimit-Reset")
            reset_in = None
            if reset:
                try:
                    reset_in = int(reset) - int(time.time())
                except Exception:
                    reset_in = None
            if resp.status_code == 429:
                # honor server reset if provided
                self.limiter.wait(remaining_header="0", reset_in_sec=reset_in or 30)
                time.sleep(5)
                resp = self.session.get(url, params=p, timeout=self.timeout)

        if resp.status_code != 200:
            raise RuntimeError(f"FEC error {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    def _paged(
        self,
        endpoint: str,
        params: dict,
        page_key: str = "page",
        per_page: int = 100,
        max_pages: int | None = None,
    ):
        page = 1
        out = []
        while True:
            q = dict(params)
            q["per_page"] = per_page
            q[page_key] = page
            data = self._get(endpoint, q)
            results = data.get("results", [])
            out.extend(results)
            pagination = data.get("pagination") or {}
            total_pages = pagination.get("pages") or 1
            if max_pages and page >= max_pages:
                break
            if page >= total_pages:
                break
            page += 1
        return out

    # endpoints
    def candidates(self, state: str, office: str, cycle: int) -> list[dict]:
        params = dict(state=state, office=office, election_year=cycle, cycle=cycle, sort="name")
        return self._paged("candidates", params)

    def committees_for_candidate(self, candidate_id: str, cycle: int) -> list[dict]:
        # Try candidate-specific endpoint first; fall back to general search
        try:
            return self._paged(f"candidate/{candidate_id}/committees", {"cycle": cycle})
        except Exception:
            return self._paged("committees", {"candidate_id": candidate_id, "cycle": cycle})

    def schedule_a_for_committee(
        self,
        committee_id: str,
        two_year_transaction_period: int,
        contributor_state: str | None = None,
        per_page: int = 100,
        max_pages: int | None = None,
        is_individual: bool = True,
    ) -> list[dict]:
        params = {
            "committee_id": committee_id,
            "two_year_transaction_period": two_year_transaction_period,
            "sort_hide_null": "false",
            "sort": "-contribution_receipt_date",
            "per_page": per_page,
        }
        if is_individual:
            params["is_individual"] = "true"
        if contributor_state:
            params["contributor_state"] = contributor_state
        return self._paged("schedules/schedule_a/", params, max_pages=max_pages)


# -------------------- donor ZIP -> lat/lon -------------------- #
def geocode_zip_us(zip_codes: list[str]) -> dict[str, tuple[float, float, str]]:
    """
    Returns {zip: (lat, lon, state_code)} for 5-digit ZIPs that resolve in Nominatim.
    """
    nomi = pgeocode.Nominatim("us")
    out: dict[str, tuple[float, float, str]] = {}
    # pgeocode handles vectorized lookups too, but we’ll be explicit & robust
    unique = sorted({z.strip() for z in zip_codes if z and isinstance(z, str)})
    if not unique:
        return out
    df_geo = nomi.query_postal_code(unique)
    if df_geo is None or len(df_geo) == 0:
        return out
    for _, r in df_geo.iterrows():
        z = str(r.get("postal_code") or "").strip()
        lat = r.get("latitude")
        lon = r.get("longitude")
        st = str(r.get("state_code") or "").strip()
        if z and pd.notna(lat) and pd.notna(lon):
            out[z] = (float(lat), float(lon), st)
    return out


# -------------------- render AL donor map -------------------- #
def write_alabama_map_html(dataset: dict, html_path: str = "output/donor_map_AL.html") -> None:
    """
    dataset = {
      "center": [lat, lon, zoom],
      "byCandidate": {
         "Candidate Name": [
            {"zip": "YYYYY", "lat": 32.0, "lon": -86.0, "amount": 12345.0, "count": 12},
            ...
         ],
         ...
      },
      "candidates": ["A", "B", ...]
    }
    """
    # Keep it raw to avoid backslash-escape warnings
    PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Alabama Donor ZIP Map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <!-- Leaflet -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
          integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
  <!-- MarkerCluster -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.css" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet.markercluster@1.5.3/dist/MarkerCluster.Default.css" />
  <script src="https://unpkg.com/leaflet.markercluster@1.5.3/dist/leaflet.markercluster.js"></script>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, sans-serif; background: #0b0f17; color: #eef2f6;}
    .layout { display: grid; grid-template-columns: 340px 1fr; height: 100%; }
    .side { border-right: 1px solid #1e2a38; padding: 14px; overflow: auto; }
    .main { position: relative; }
    #map { position: absolute; inset: 0; }
    h1 { font-size: 18px; margin: 0 0 8px; color:#9bd1ff; }
    .hint { color: #9aaec2; font-size: 12px; margin-bottom: 10px; }
    .controls { display:flex; gap:8px; align-items: center; margin: 6px 0 12px; }
    .controls input[type="text"] { flex:1; padding:8px 10px; border-radius:10px; border:1px solid #2a3b52; background:#0e1420; color:#e8f0fa; }
    .btn { border:1px solid #2a3b52; background:#122033; padding:8px 12px; border-radius:10px; color:#cfe9ff; cursor:pointer; }
    .btn:hover { background:#183049; }
    .list { display:flex; flex-direction: column; gap:6px; }
    .item { padding:8px 10px; border-radius:10px; border:1px solid #213044; background:#0e1726; cursor:pointer; }
    .item:hover { background:#122033; }
    .item strong { color:#e9f4ff; }
    .small { font-size:12px; color:#9aaec2; }
    .legend { position: absolute; bottom: 14px; left: 14px; background: rgba(10,14,22,0.85); border: 1px solid #213044; padding: 8px 10px; border-radius: 8px; color:#e9f4ff; }
    a.link { color:#9bd1ff; text-decoration: none; }
    a.link:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="layout">
    <div class="side">
      <h1>Alabama Donor ZIP Map</h1>
      <div class="hint">Click a name to see where their Alabama donations come from. Use search to filter the list.</div>
      <div class="controls">
        <input id="q" type="text" placeholder="Search candidate…" />
        <button id="reset" class="btn">All</button>
      </div>
      <div id="list" class="list"></div>
      <div class="hint" style="margin-top:10px;">Open the network graph: <a class="link" href="./campaign_network.html" target="_blank">campaign_network.html</a></div>
    </div>
    <div class="main">
      <div id="map"></div>
      <div class="legend" id="legend"></div>
    </div>
  </div>

<script>
  // Data payload injected from Python:
  const DATA = __DATA__;

  // Map init (center on Alabama)
  const map = L.map('map', { zoomControl: true, scrollWheelZoom: true })
    .setView([DATA.center[0], DATA.center[1]], DATA.center[2]);

  // Basemap
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 12,
    attribution: '&copy; OpenStreetMap'
  }).addTo(map);

  // Marker cluster group
  const cluster = L.markerClusterGroup({ showCoverageOnHover: false, spiderfyOnMaxZoom: true });
  map.addLayer(cluster);

  // Simple radius scaler (sqrt scale)
  function radiusForAmount(a, minA, maxA) {
    if (!isFinite(a)) return 4;
    const minR = 4, maxR = 22;
    if (maxA <= minA) return minR;
    const t = Math.sqrt((a - minA) / (maxA - minA));
    return minR + t * (maxR - minR);
  }

  // Build sidebar list
  const list = document.getElementById('list');
  let names = [...DATA.candidates];
  function renderList(filter="") {
    list.innerHTML = "";
    const f = filter.trim().toLowerCase();
    const show = names.filter(n => n.toLowerCase().includes(f));
    if (show.length === 0) {
      const div = document.createElement('div');
      div.className = 'item';
      div.textContent = 'No matches';
      list.appendChild(div);
      return;
    }
    for (const n of show) {
      const div = document.createElement('div');
      div.className = 'item';
      const pts = (DATA.byCandidate[n] || []);
      const total = pts.reduce((s,p)=>s+(p.amount||0),0);
      const count = pts.reduce((s,p)=>s+(p.count||0),0);
      div.innerHTML = `<strong>${n}</strong><div class="small">AL ZIPs: ${pts.length.toLocaleString()} • $${Math.round(total).toLocaleString()} • ${count.toLocaleString()} gifts</div>`;
      div.onclick = () => selectCandidate(n);
      list.appendChild(div);
    }
  }

  // Legend
  function renderLegend(minA, maxA) {
    const el = document.getElementById('legend');
    el.innerHTML = `<div style="font-weight:600; margin-bottom:6px;">Dot size = total $ at ZIP</div>
      <div class="small">Min: $${Math.round(minA).toLocaleString()} &nbsp;→&nbsp; Max: $${Math.round(maxA).toLocaleString()}</div>`;
  }

  // Current view
  let currentName = null;

  function selectCandidate(name=null) {
    currentName = name;
    cluster.clearLayers();
    let ptsAll = [];
    if (name === null) {
      // All candidates
      for (const nm of DATA.candidates) {
        ptsAll = ptsAll.concat(DATA.byCandidate[nm] || []);
      }
    } else {
      ptsAll = DATA.byCandidate[name] || [];
    }
    if (ptsAll.length === 0) return;

    const minA = Math.min(...ptsAll.map(p=>p.amount||0));
    const maxA = Math.max(...ptsAll.map(p=>p.amount||0));
    renderLegend(minA, maxA);

    const group = L.featureGroup();
    for (const p of ptsAll) {
      const r = radiusForAmount(p.amount||0, minA, maxA);
      const m = L.circleMarker([p.lat, p.lon], {
        radius: r,
        color: "#70e1ff",
        weight: 1,
        fillColor: "#f7b267",
        fillOpacity: 0.75
      }).bindPopup(
        `<div style="min-width:200px"><div><b>ZIP:</b> ${p.zip}</div>
         <div><b>Total:</b> $${Math.round(p.amount||0).toLocaleString()}</div>
         <div><b>Gifts:</b> ${p.count.toLocaleString()}</div>
         ${name ? `<div><b>Candidate:</b> ${name}</div>` : ""}</div>`
      );
      cluster.addLayer(m);
      group.addLayer(m);
    }
    try { map.fitBounds(group.getBounds().pad(0.12)); } catch(e) {}
  }

  // Search
  const q = document.getElementById('q');
  q.addEventListener('input', () => renderList(q.value));

  // Reset to All
  document.getElementById('reset').onclick = () => { q.value=""; renderList(""); selectCandidate(null); };

  // Initial render
  renderList("");
  selectCandidate(null);
</script>
</body>
</html>
"""
    # Inject JSON safely (no forward-slash escaping by default)
    payload = json.dumps(dataset, ensure_ascii=False)
    html = PAGE.replace("__DATA__", payload)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# -------------------- pipeline -------------------- #
def build_pipeline(
    state="AL",
    cycle=DEFAULT_TWO_YEAR,
    office="H",
    donor_state: str | None = None,
    min_amount: float = 0.0,
    top_n_clusters: int = 10,
    max_committees_per_candidate: int | None = None,
    max_pages_per_committee: int | None = None,
    edge_cap: int = 5000,
    timeout: int = 90,
    retries: int = 4,
    make_map: bool = True,
    map_state: str = "AL",
):
    ensure_dirs()

    # key resolution (CLI/env/hardcoded)
    api_key = os.getenv("FEC_API_KEY")
    if not api_key and HARDCODED_FEC_API_KEY:
        api_key = HARDCODED_FEC_API_KEY
    if not api_key:
        raise RuntimeError(
            "No FEC_API_KEY found. Use --api-key on CLI, export FEC_API_KEY, "
            "or paste into HARDCODED_FEC_API_KEY in the script."
        )

    client = FECClient(
        api_key=api_key,
        session=make_session(cache_name="fec_cache", expire_after=24 * 3600, retries=retries),
        limiter=HourlyRateLimiter(max_per_hour=950, rpm=15),  # ≈900/hr, under 1000/hr
        timeout=timeout,
    )

    # 1) Candidates
    print(f"[step] candidates: state={state} office={office} cycle={cycle}")
    cands = client.candidates(state=state, office=office, cycle=cycle)
    if not cands:
        raise RuntimeError("No candidates returned. Try different params.")
    df_cands = pd.DataFrame(cands)
    keep_c = [
        "candidate_id",
        "name",
        "party_full",
        "office_full",
        "state",
        "incumbent_challenge_full",
    ]
    for col in keep_c:
        if col not in df_cands.columns:
            df_cands[col] = None
    sanitize_df(df_cands[keep_c]).to_csv(
        "data/candidates.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
        doublequote=True,
    )

    # 2) Committees
    print("[step] committees for candidates…")
    rows_comm: list[dict] = []
    for _, row in tqdm(df_cands.iterrows(), total=len(df_cands)):
        cid = row["candidate_id"]
        coms = client.committees_for_candidate(cid, cycle=cycle)
        for com in coms:
            com["_candidate_id"] = cid
            com["_candidate_name"] = row.get("name")
        if max_committees_per_candidate:
            # prioritize principal/authorized first
            coms = sorted(
                coms, key=lambda r: (r.get("designation") not in ("P", "A"), r.get("committee_id"))
            )[:max_committees_per_candidate]
        rows_comm.extend(coms)

    if not rows_comm:
        raise RuntimeError("No committees found.")
    df_comm = pd.DataFrame(rows_comm)
    keep_m = [
        "committee_id",
        "name",
        "designation",
        "designation_full",
        "committee_type",
        "treasurer_name",
        "_candidate_id",
        "_candidate_name",
    ]
    for col in keep_m:
        if col not in df_comm.columns:
            df_comm[col] = None
    df_comm["is_principal_or_auth"] = df_comm["designation"].isin(["P", "A"])
    df_comm.sort_values(
        ["_candidate_id", "is_principal_or_auth"], ascending=[True, False], inplace=True
    )
    sanitize_df(df_comm[keep_m + ["is_principal_or_auth"]].drop_duplicates("committee_id")).to_csv(
        "data/committees.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
        doublequote=True,
    )

    # 3) Schedule A contributions
    print("[step] schedule_a individuals… (this can take a bit)")
    contrib_rows: list[dict] = []
    uniq_committees = df_comm["committee_id"].dropna().drop_duplicates().tolist()
    iterator = df_comm.drop_duplicates("committee_id").iterrows()
    for _, row in tqdm(iterator, total=len(uniq_committees)):
        committee_id = row["committee_id"]
        try:
            res = client.schedule_a_for_committee(
                committee_id=committee_id,
                two_year_transaction_period=cycle,
                contributor_state=donor_state,  # None = all states
                per_page=100,
                max_pages=max_pages_per_committee,
                is_individual=True,
            )
        except Exception as e:
            print(f"[warn] schedule_a failed for {committee_id}: {e}. Skipping this committee.")
            continue

        for r in res:
            r["_committee_id"] = committee_id
            r["_candidate_id"] = row["_candidate_id"]
            r["_candidate_name"] = row["_candidate_name"]
            contrib_rows.append(r)

    if not contrib_rows:
        raise RuntimeError(
            "No contributions pulled; loosen filters or increase --timeout/--retries."
        )
    df_a = pd.DataFrame(contrib_rows)

    # clean/filter
    amt = "contribution_receipt_amount"
    if amt not in df_a.columns:
        df_a[amt] = 0.0
    df_a = df_a[df_a[amt].fillna(0) > 0]
    if min_amount > 0:
        df_a = df_a[df_a[amt] >= float(min_amount)]
    essentials = [
        "contributor_name",
        "contributor_city",
        "contributor_state",
        "contributor_zip",
        "contribution_receipt_date",
        amt,
        "_committee_id",
        "_candidate_id",
        "_candidate_name",
        "contributor_id",
    ]
    for c in essentials:
        if c not in df_a.columns:
            df_a[c] = None
    sanitize_df(df_a[essentials]).to_csv(
        "data/contributions_raw.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
        doublequote=True,
    )

    # 4) edges (donor-candidate weights)
    df_a["donor_key"] = df_a.apply(donor_key, axis=1)
    edges = (
        df_a.groupby(["donor_key", "_candidate_name"], dropna=False)[amt]
        .sum()
        .reset_index()
        .rename(columns={amt: "weight", "_candidate_name": "candidate"})
    )
    donor_name_map = (
        df_a.groupby("donor_key")["contributor_name"]
        .agg(lambda s: normalize_name(s.dropna().iloc[0]) if len(s.dropna()) else "")
        .to_dict()
    )

    # 5) graph
    G = nx.Graph()
    for dkey, label in donor_name_map.items():
        G.add_node(dkey, label=label or dkey, kind="donor")
    for c in edges["candidate"].dropna().unique():
        G.add_node(c, label=c, kind="candidate")

    for _, r in edges.iterrows():
        u, v, w = r["donor_key"], r["candidate"], float(r["weight"])
        if u and v and w > 0:
            G.add_edge(u, v, weight=w)

    # metrics
    deg_cent = nx.degree_centrality(G)
    strength = {
        n: sum(ed.get("weight", 1.0) for _, _, ed in G.edges(n, data=True)) for n in G.nodes()
    }
    nx.set_node_attributes(G, deg_cent, "degree_centrality")
    nx.set_node_attributes(G, strength, "strength")

    # communities
    clusters: dict[str, int] = {}
    if _HAS_LOUVAIN and G.number_of_edges() > 0:
        clusters = community_louvain.best_partition(G, weight="weight")
    else:
        if G.number_of_edges() > 0:
            comps = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
            for i, com in enumerate(comps):
                for n in com:
                    clusters[n] = i
        else:
            clusters = {n: 0 for n in G.nodes()}
    nx.set_node_attributes(G, clusters, "cluster")

    # exports (sanitized & quoted)
    node_rows = [
        {
            "node_id": n,
            "label": d.get("label", n),
            "kind": d.get("kind"),
            "degree_centrality": d.get("degree_centrality", 0.0),
            "strength": d.get("strength", 0.0),
            "cluster": d.get("cluster", -1),
        }
        for n, d in G.nodes(data=True)
    ]
    df_nodes = pd.DataFrame(node_rows).sort_values(["kind", "strength"], ascending=[True, False])
    sanitize_df(df_nodes).to_csv(
        "data/nodes.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
        doublequote=True,
    )

    edge_rows = [
        {"source": u, "target": v, "weight": d.get("weight", 1.0)} for u, v, d in G.edges(data=True)
    ]
    df_edges = pd.DataFrame(edge_rows).sort_values("weight", ascending=False)
    sanitize_df(df_edges).to_csv(
        "data/edges.csv",
        index=False,
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
        doublequote=True,
    )

    # brief clusters
    cluster_groups: dict[int, list[str]] = defaultdict(list)
    for n, cl in clusters.items():
        cluster_groups[cl].append(n)
    ranked = sorted(cluster_groups.items(), key=lambda kv: len(kv[1]), reverse=True)[
        :top_n_clusters
    ]

    def top_members(nodes_list: list[str], k=8):
        cands = [n for n in nodes_list if G.nodes[n]["kind"] == "candidate"]
        dons = [n for n in nodes_list if G.nodes[n]["kind"] == "donor"]
        cands_sorted = sorted(cands, key=lambda n: G.nodes[n].get("strength", 0.0), reverse=True)[
            :k
        ]
        dons_sorted = sorted(dons, key=lambda n: G.nodes[n].get("strength", 0.0), reverse=True)[:k]

        def fmt(nodes):
            out = []
            for x in nodes:
                lab = G.nodes[x].get("label", x)
                out.append(f"{lab} (Σ=${G.nodes[x].get('strength',0):,.0f})")
            return out

        return fmt(cands_sorted), fmt(dons_sorted)

    with open("output/top_clusters_brief.md", "w", encoding="utf-8") as f:
        f.write("# Top 10 Influence Clusters\n\n")
        f.write(
            f"*State:* {state}  \n*Cycle:* {cycle}  \n*Office:* {office}  \n*Donor filter:* {donor_state or 'Any'}  \n*Min amount:* ${min_amount:,.0f}\n\n"
        )
        for i, (cid, members) in enumerate(ranked, 1):
            cand_list, donor_list = top_members(members, k=8)
            f.write(f"## Cluster {i} — ID {cid} (size {len(members)})\n")
            f.write(f"**Top candidates:** {', '.join(cand_list) if cand_list else '—'}\n\n")
            f.write(f"**Top donors:** {', '.join(donor_list) if donor_list else '—'}\n\n")

    # visualization trim for speed, keep detail
    G_draw = G
    if df_edges.shape[0] > edge_cap:
        thresh = df_edges.nlargest(edge_cap, "weight")["weight"].min()
        keep = {
            (r["source"], r["target"]) for _, r in df_edges[df_edges["weight"] >= thresh].iterrows()
        }
        H = nx.Graph()
        for n, d in G.nodes(data=True):
            H.add_node(n, **d)
        for u, v, d in G.edges(data=True):
            if (u, v) in keep or (v, u) in keep:
                H.add_edge(u, v, **d)
        G_draw = H

    # Build PyVis graph
    net = Network(
        height="900px",
        width="100%",
        bgcolor="#0b0f17",
        font_color="white",
        notebook=False,
        directed=False,
    )
    net.set_options(
        """
{
  "nodes": {
    "borderWidth": 1,
    "shape": "dot",
    "scaling": { "min": 3, "max": 40 },
    "font": { "size": 16, "face": "Inter" }
  },
  "edges": {
    "arrows": { "to": { "enabled": false } },
    "smooth": { "type": "continuous" },
    "selectionWidth": 2,
    "scaling": { "min": 1, "max": 12 }
  },
  "interaction": {
    "hover": true,
    "tooltipDelay": 120,
    "hideEdgesOnDrag": false,
    "multiselect": true,
    "navigationButtons": true,
    "keyboard": { "enabled": true, "speed": { "x": 10, "y": 10, "zoom": 0.2 } }
  },
  "physics": {
    "enabled": true,
    "solver": "barnesHut",
    "barnesHut": {
      "gravitationalConstant": -24000,
      "springLength": 210,
      "springConstant": 0.025,
      "damping": 0.6,
      "avoidOverlap": 0.1
    },
    "stabilization": { "enabled": true, "iterations": 1200, "updateInterval": 50 }
  }
}
"""
    )

    # add nodes/edges
    for n, d in G_draw.nodes(data=True):
        kind = d.get("kind")
        label = d.get("label", n)
        size = 8 + math.log1p(d.get("strength", 0.0)) * 2.6  # size by strength
        color = "#70e1ff" if kind == "candidate" else "#f7b267"
        title = (
            f"<b>{label}</b><br>"
            f"type: {kind}<br>"
            f"cluster: {d.get('cluster', 0)}<br>"
            f"degree_centrality: {d.get('degree_centrality',0):.4f}<br>"
            f"strength (sum $): {d.get('strength',0):,.0f}"
        )
        net.add_node(n, label=label, title=title, color=color, size=size, group=d.get("cluster", 0))

    for u, v, d in G_draw.edges(data=True):
        w = float(d.get("weight", 1.0))
        net.add_edge(u, v, value=w, title=f"${w:,.0f}")

    html_path_graph = "output/campaign_network.html"
    net.write_html(html_path_graph, notebook=False)
    print(f"[viz] Wrote {os.path.abspath(html_path_graph)}")
    try:
        subprocess.run(["open", html_path_graph], check=False)  # macOS auto-open
    except Exception:
        pass

    # ---- Alabama ZIP Map (always computed from AL donors) ----
    if make_map:
        print("[step] building Alabama ZIP donor map…")
        # We only want Alabama donors for the map
        df_al = df_a[(df_a["contributor_state"].fillna("").str.upper() == map_state.upper())].copy()
        df_al["zip5"] = df_al["contributor_zip"].astype(str).str.extract(r"(\d{5})", expand=False)
        df_al = df_al.dropna(subset=["zip5"])

        # Aggregate per candidate per ZIP
        amtcol = "contribution_receipt_amount"
        grouped = (
            df_al.groupby(["_candidate_name", "zip5"], dropna=False)[amtcol]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={"sum": "amount", "count": "count"})
        )

        # Geocode unique zips
        zip_list = grouped["zip5"].dropna().unique().tolist()
        zmap = geocode_zip_us(zip_list)

        byCandidate: dict[str, list[dict]] = defaultdict(list)
        for _, r in grouped.iterrows():
            nm = str(r["_candidate_name"])
            z = str(r["zip5"])
            geo = zmap.get(z)
            if not geo:
                continue
            lat, lon, st = geo
            if st.upper() != map_state.upper():
                continue
            byCandidate[nm].append(
                {
                    "zip": z,
                    "lat": lat,
                    "lon": lon,
                    "amount": float(r["amount"] or 0.0),
                    "count": int(r["count"] or 0),
                }
            )

        candidates_sorted = sorted(byCandidate.keys())
        dataset = {
            "center": [32.9, -86.8, 6],  # AL center-ish
            "byCandidate": byCandidate,
            "candidates": candidates_sorted,
        }
        html_path_map = "output/donor_map_AL.html"
        write_alabama_map_html(dataset, html_path=html_path_map)
        print(f"[map] Wrote {os.path.abspath(html_path_map)}")
        try:
            subprocess.run(["open", html_path_map], check=False)  # macOS auto-open
        except Exception:
            pass

    print("\nDone ✅")
    print("Files written:")
    print("  data/candidates.csv")
    print("  data/committees.csv")
    print("  data/contributions_raw.csv")
    print("  data/nodes.csv")
    print("  data/edges.csv")
    print("  output/top_clusters_brief.md")
    print("  output/campaign_network.html")
    if make_map:
        print("  output/donor_map_AL.html")


# -------------------- CLI -------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build Campaign Money Graphs (FEC Schedule A) + AL donor map"
    )
    parser.add_argument("--state", default="AL", help="State for candidates (e.g., AL)")
    parser.add_argument(
        "--cycle", type=int, default=DEFAULT_TWO_YEAR, help="Two-year period (e.g., 2024, 2022)"
    )
    parser.add_argument(
        "--office",
        default="H",
        choices=["H", "S", "P"],
        help="H (House), S (Senate), P (President)",
    )
    parser.add_argument(
        "--donor-state", default=None, help="Optional contributor_state filter (e.g., AL)"
    )
    parser.add_argument(
        "--min-amount", type=float, default=0.0, help="Filter contributions below this amount"
    )
    parser.add_argument(
        "--top-n-clusters", type=int, default=10, help="How many clusters to summarize"
    )
    parser.add_argument(
        "--max-committees-per-candidate",
        type=int,
        default=None,
        help="Trim committees per candidate (prioritize P/A)",
    )
    parser.add_argument(
        "--max-pages-per-committee",
        type=int,
        default=None,
        help="Trim pages per committee (each page=100 rows)",
    )
    parser.add_argument(
        "--edge-cap", type=int, default=5000, help="Trim visualization to strongest N edges"
    )
    parser.add_argument(
        "--api-key", default=None, help="FEC API key (optional; else use env FEC_API_KEY)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP timeout seconds (increase if you see ReadTimeout)",
    )
    parser.add_argument(
        "--retries", type=int, default=4, help="HTTP retry count for timeouts/5xx/429"
    )
    parser.add_argument(
        "--skip-map", action="store_true", help="Skip building the Alabama donor ZIP map"
    )
    parser.add_argument("--map-state", default="AL", help="State to map ZIPs for (default AL)")

    args = parser.parse_args()

    # Resolve key order: CLI > env > hardcoded
    if args.api_key:
        os.environ["FEC_API_KEY"] = args.api_key
    elif not os.getenv("FEC_API_KEY") and HARDCODED_FEC_API_KEY:
        os.environ["FEC_API_KEY"] = HARDCODED_FEC_API_KEY

    build_pipeline(
        state=args.state,
        cycle=args.cycle,
        office=args.office,
        donor_state=args.donor_state,
        min_amount=args.min_amount,
        top_n_clusters=args.top_n_clusters,
        max_committees_per_candidate=args.max_committees_per_candidate,
        max_pages_per_committee=args.max_pages_per_committee,
        edge_cap=args.edge_cap,
        timeout=args.timeout,
        retries=args.retries,
        make_map=(not args.skip_map),
        map_state=args.map_state,
    )
