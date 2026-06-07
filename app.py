#!/usr/bin/env python3
"""
Streamlit app — FIFA World Cup 2026 Monte-Carlo tournament simulator.

Visualizes:
  * Group-by-group placing probabilities (win group / runner-up / qualify).
  * Each team's route to the final (probability of reaching each knockout
    round and of winning the tournament).

The win/draw/loss probability for each match comes from the teams' Elo ratings
(a fast, robust proxy for the trained XGBoost model). The bracket is parsed
directly from the WC 2026 schedule feature group, so the knockout structure
matches the real 48-team format.

Run:
  uv pip install -r requirements.txt
  streamlit run app.py
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

# Schedule team-name -> Elo-ratings country-name aliases.
ALIASES = {
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "USA": "United States",
    "Congo DR": "DR Congo",
    "Cabo Verde": "Cape Verde",
    "Türkiye": "Turkey",
    "Czechia": "Czech Republic",
}

HOSTS = {"United States", "Canada", "Mexico"}  # small home advantage
HOME_ELO_BONUS = 60.0
DEFAULT_ELO = 1500.0


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, dict[str, float], str]:
    """Load the schedule and Elo ratings.

    Tries the Hopsworks feature store first, then falls back to the local
    ``data/*.csv`` files so the app is runnable without a connection.
    Returns (schedule_df, elo_by_country, source_label).
    """
    try:
        import hopsworks

        project = hopsworks.login()
        fs = project.get_feature_store()
        schedule = fs.get_feature_group("fifa2026_schedule_fixtures", version=1).read()
        elo = fs.get_feature_group("elo_ratings", version=1).read()
        source = "Hopsworks feature store"
    except Exception:
        schedule = pd.read_csv(DATA_DIR / "FIFA2026_schedule_Fixtures.csv")
        elo = pd.read_csv(DATA_DIR / "elo_ratings.csv")
        source = "local data/ CSVs"

    schedule = schedule.sort_values("match_number", key=_match_num_key).reset_index(drop=True)

    # Latest Elo per country.
    elo = elo.sort_values("date")
    elo_by_country = elo.groupby("country")["elo_rating"].last().to_dict()
    return schedule, elo_by_country, source


def _match_num_key(s: pd.Series) -> pd.Series:
    return s.map(lambda x: int(re.search(r"\d+", str(x)).group()))


def match_no(label: str) -> int:
    return int(re.search(r"\d+", str(label)).group())


# --------------------------------------------------------------------------- #
# Schedule parsing
# --------------------------------------------------------------------------- #
def canon(team: str) -> str:
    team = re.sub(r"\s+", " ", team).strip()
    return ALIASES.get(team, team)


def first_team(slot: str) -> str:
    """Pick a representative team from a 'A/B/C' play-off placeholder."""
    return canon(slot.split("/")[0].strip())


def parse_schedule(schedule: pd.DataFrame):
    """Split the schedule into group matches, groups, and knockout matches."""
    groups: dict[str, list[str]] = defaultdict(list)
    group_matches: list[tuple[str, str, str]] = []   # (group, teamA, teamB)
    ko_matches: dict[int, tuple] = {}                # mno -> (slotA, slotB)

    for _, row in schedule.iterrows():
        mno = match_no(row["match_number"])
        teams = str(row["teams"])
        if " v " not in teams:
            continue
        left, right = (s.strip() for s in teams.split(" v ", 1))
        group = str(row.get("group", "")).strip()

        if group.startswith("Group ") and len(group.split()) == 2:
            letter = group.split()[1]
            a, b = first_team(left), first_team(right)
            group_matches.append((letter, a, b))
            for t in (a, b):
                if t not in groups[letter]:
                    groups[letter].append(t)
        else:
            ko_matches[mno] = (parse_slot(left), parse_slot(right))

    return dict(groups), group_matches, ko_matches


def parse_slot(slot: str):
    """Parse a knockout-slot reference into a resolvable token."""
    s = slot.strip()
    m = re.match(r"Group ([A-L]) winners?$", s, re.I)
    if m:
        return ("winner", m.group(1).upper())
    m = re.match(r"Group ([A-L]) runners?[\s-]?up$", s, re.I)
    if m:
        return ("runner", m.group(1).upper())
    m = re.match(r"Group ([A-L/]+) third place$", s, re.I)
    if m:
        letters = [x for x in re.split(r"/", m.group(1)) if x]
        return ("third", letters)
    m = re.match(r"Winner match (\d+)$", s, re.I)
    if m:
        return ("wmatch", int(m.group(1)))
    m = re.match(r"Runner-?up match (\d+)$", s, re.I)
    if m:
        return ("lmatch", int(m.group(1)))   # loser of that match (3rd-place game)
    return ("team", first_team(s))


# --------------------------------------------------------------------------- #
# Match model (Elo -> W/D/L)
# --------------------------------------------------------------------------- #
def elo_of(team: str, elo: dict[str, float]) -> float:
    r = elo.get(team, DEFAULT_ELO)
    if team in HOSTS:
        r += HOME_ELO_BONUS
    return r


def wdl_probs(team_a: str, team_b: str, elo: dict[str, float]):
    """Return (p_a_win, p_draw, p_b_win) from Elo ratings."""
    ea = 1.0 / (1.0 + 10 ** (-(elo_of(team_a, elo) - elo_of(team_b, elo)) / 400.0))
    # Draw most likely for even matchups; tapers as the gap grows.
    d_max = 0.30
    draw = d_max * (1.0 - abs(2 * ea - 1.0))
    draw = min(draw, 2 * min(ea, 1 - ea) * 0.95)
    p_a = ea - draw / 2.0
    p_b = (1.0 - ea) - draw / 2.0
    return max(p_a, 0.0), max(draw, 0.0), max(p_b, 0.0)


def sample_match(team_a, team_b, elo, rng) -> int:
    """Return 1 (A wins), 0 (draw), -1 (B wins)."""
    p_a, p_d, p_b = wdl_probs(team_a, team_b, elo)
    total = p_a + p_d + p_b
    r = rng.random() * total
    if r < p_a:
        return 1
    if r < p_a + p_d:
        return 0
    return -1


def sample_knockout(team_a, team_b, elo, rng) -> str:
    """Knockout: no draws — split the draw mass by relative strength."""
    p_a, p_d, p_b = wdl_probs(team_a, team_b, elo)
    base = p_a + p_b
    p_a_final = (p_a + p_d * (p_a / base)) if base > 0 else 0.5
    return team_a if rng.random() < p_a_final else team_b


# --------------------------------------------------------------------------- #
# Monte-Carlo simulation
# --------------------------------------------------------------------------- #
ROUNDS = ["qualify_ko", "round_16", "quarter", "semi", "final", "champion"]


def simulate_once(groups, group_matches, ko_matches, elo, rng):
    """Run one full tournament. Returns (group_placings, round_reached)."""
    # ---- Group stage ----
    pts = defaultdict(float)
    gd = defaultdict(float)
    for letter, a, b in group_matches:
        res = sample_match(a, b, elo, rng)
        if res == 1:
            pts[a] += 3; gd[a] += 1; gd[b] -= 1
        elif res == -1:
            pts[b] += 3; gd[b] += 1; gd[a] -= 1
        else:
            pts[a] += 1; pts[b] += 1

    winners, runners, thirds = {}, {}, []
    placing = {}   # team -> "winner"/"runner"/"third"/"out"
    for letter, teams in groups.items():
        ranked = sorted(
            teams,
            key=lambda t: (pts[t], gd[t], elo_of(t, elo), rng.random()),
            reverse=True,
        )
        winners[letter] = ranked[0]
        runners[letter] = ranked[1]
        placing[ranked[0]] = "winner"
        placing[ranked[1]] = "runner"
        if len(ranked) > 2:
            thirds.append((letter, ranked[2]))
            placing[ranked[2]] = "third"
        for t in ranked[3:]:
            placing[t] = "out"

    # Best 8 third-placed teams qualify.
    thirds_ranked = sorted(
        thirds,
        key=lambda lt: (pts[lt[1]], gd[lt[1]], elo_of(lt[1], elo), rng.random()),
        reverse=True,
    )
    qualified_thirds = thirds_ranked[:8]
    third_by_group = {letter: team for letter, team in qualified_thirds}
    available_thirds = dict(third_by_group)   # consumed greedily by KO slots

    round_reached = {t: "group" for t in placing}

    def resolve(slot, results):
        kind = slot[0]
        if kind == "winner":
            return winners.get(slot[1])
        if kind == "runner":
            return runners.get(slot[1])
        if kind == "third":
            for letter in slot[1]:
                if letter in available_thirds:
                    return available_thirds.pop(letter)
            # Fallback: any remaining qualified third.
            if available_thirds:
                letter = next(iter(available_thirds))
                return available_thirds.pop(letter)
            return None
        if kind == "wmatch":
            return results.get(slot[1], (None, None))[0]
        if kind == "lmatch":
            return results.get(slot[1], (None, None))[1]
        if kind == "team":
            return slot[1]
        return None

    # ---- Knockout stage ----
    # Round buckets by match number (per the WC2026 schedule).
    R32 = range(73, 89)
    R16 = range(89, 97)
    QF = range(97, 101)
    SF = range(101, 103)
    FINAL = 104

    def mark(team, label):
        if team:
            round_reached[team] = label

    results = {}   # mno -> (winner, loser)
    for mno in sorted(ko_matches):
        if mno == 103:   # third-place play-off: ignore for "route to final"
            continue
        a = resolve(ko_matches[mno][0], results)
        b = resolve(ko_matches[mno][1], results)
        if mno in R32:
            mark(a, "qualify_ko"); mark(b, "qualify_ko")
        if a is None or b is None:
            results[mno] = (a or b, None)
            continue
        winner = sample_knockout(a, b, elo, rng)
        loser = b if winner == a else a
        results[mno] = (winner, loser)

        if mno in R16:
            mark(winner, "round_16")
        elif mno in QF:
            mark(winner, "quarter")
        elif mno in SF:
            mark(winner, "semi")
        elif mno == FINAL:
            mark(a, "final"); mark(b, "final")
            mark(winner, "champion")

    # Teams reaching R16 are the winners of R32 etc. — promote markers so that
    # a deeper round implies all shallower ones (handled at aggregation).
    return placing, round_reached


def run_simulation(groups, group_matches, ko_matches, elo, n_sims, seed):
    rng = np.random.default_rng(seed)
    teams = sorted({t for ts in groups.values() for t in ts})

    place_counts = {t: defaultdict(int) for t in teams}     # winner/runner/third/out
    round_counts = {t: defaultdict(int) for t in teams}     # cumulative round reach

    rank_order = {"group": 0, "qualify_ko": 1, "round_16": 2,
                  "quarter": 3, "semi": 4, "final": 5, "champion": 6}

    for _ in range(n_sims):
        placing, reached = simulate_once(groups, group_matches, ko_matches, elo, rng)
        for t, p in placing.items():
            place_counts[t][p] += 1
        for t, r in reached.items():
            lvl = rank_order[r]
            # cumulative: count every round at or below the furthest reached
            for label, idx in rank_order.items():
                if 1 <= idx <= lvl:
                    round_counts[t][label] += 1

    return teams, place_counts, round_counts


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="WC 2026 Monte-Carlo Simulator", layout="wide")
    st.title("⚽ FIFA World Cup 2026 — Monte-Carlo Simulator")

    schedule, elo, source = load_data()
    groups, group_matches, ko_matches = parse_schedule(schedule)

    with st.sidebar:
        st.header("Simulation settings")
        n_sims = st.select_slider(
            "Number of rollouts",
            options=[500, 1000, 2000, 5000, 10000],
            value=2000,
        )
        st.caption(f"Data source: **{source}**")
        st.caption("Match outcomes modelled from Elo ratings "
                   "(host bonus for USA/Canada/Mexico).")
        rerun = st.button("🎲 Re-run simulation", type="primary", use_container_width=True)

    # A counter in session state changes the RNG seed on each re-run.
    if "run_id" not in st.session_state:
        st.session_state.run_id = 0
    if rerun:
        st.session_state.run_id += 1

    seed = 1234 + st.session_state.run_id

    with st.spinner(f"Running {n_sims:,} tournament rollouts…"):
        teams, place_counts, round_counts = run_simulation(
            groups, group_matches, ko_matches, elo, n_sims, seed
        )

    st.caption(f"Run #{st.session_state.run_id + 1} • {n_sims:,} rollouts • seed {seed}")

    # ---- Championship odds ----
    st.header("🏆 Championship & final odds")
    champ_rows = []
    for t in teams:
        rc = round_counts[t]
        champ_rows.append({
            "Team": t,
            "Win 🏆": rc["champion"] / n_sims,
            "Reach final": rc["final"] / n_sims,
            "Reach semi": rc["semi"] / n_sims,
            "Reach QF": rc["quarter"] / n_sims,
            "Reach R16": rc["round_16"] / n_sims,
            "Reach KO": rc["qualify_ko"] / n_sims,
        })
    champ_df = pd.DataFrame(champ_rows).sort_values("Win 🏆", ascending=False).reset_index(drop=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Top 15 — title probability")
        st.bar_chart(champ_df.head(15).set_index("Team")["Win 🏆"])
    with col2:
        st.subheader("Route to the final")
        st.dataframe(
            champ_df.style.format({c: "{:.1%}" for c in champ_df.columns if c != "Team"}),
            use_container_width=True, height=560, hide_index=True,
        )

    # ---- Group placing probabilities ----
    st.header("📊 Group stage — placing probabilities")
    cols = st.columns(3)
    for i, letter in enumerate(sorted(groups)):
        with cols[i % 3]:
            st.subheader(f"Group {letter}")
            rows = []
            for t in groups[letter]:
                pc = place_counts[t]
                rows.append({
                    "Team": t,
                    "Win grp": pc["winner"] / n_sims,
                    "Runner": pc["runner"] / n_sims,
                    "Qualify": (pc["winner"] + pc["runner"] + pc["third"]) / n_sims,
                })
            gdf = pd.DataFrame(rows).sort_values("Qualify", ascending=False)
            st.dataframe(
                gdf.style.format({c: "{:.0%}" for c in ["Win grp", "Runner", "Qualify"]}),
                use_container_width=True, hide_index=True,
            )

    st.caption(
        "Qualify = top-2 in group, or one of the 8 best third-placed teams. "
        "Third-place knockout slots are assigned greedily from qualified thirds, "
        "an approximation of FIFA's official placement table."
    )


if __name__ == "__main__":
    main()
