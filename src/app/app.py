#!/usr/bin/env python3
"""WC 2026 tournament simulator (Streamlit).

Embedded model: downloads ``wc2026_match_predictor`` and predicts locally, but
builds every feature row **through the shared feature definition** (``features``,
which is also what the feature pipeline used) using ratings read **live** from
the feature store — no bundled snapshot, no training/serving skew.

Design (per project notes):
  * Pre-compute the full ordered-pair probability table for all 48 teams (both
    home=0/1) in ONE predict_proba call; every Monte-Carlo rollout is then pure
    Python sampling from that table, so 1000+ rollouts run in seconds.
  * All Streamlit calls live in ``main()`` under ``if __name__ == "__main__"``;
    the simulation / bracket functions are pure (take ``table`` / ``structure`` /
    ``rng``), so ``_selftest.py`` can import and exercise them offline.
"""
from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd

import features as F

TEAMS_PER_GROUP = 4
N_GROUPS = 12
ROUND_OF = {  # match-number ranges -> round name
    "R32": range(73, 89),
    "R16": range(89, 97),
    "QF": range(97, 101),
    "SF": range(101, 103),
}
ROUND_ORDER = ["R32", "R16", "QF", "SF", "Final", "Won"]
ROUND_COL = {"R32": 0, "R16": 1, "QF": 2, "SF": 3, "Final": 4}


# ---------------------------------------------------------------------------
# Hopsworks-backed loaders (only these touch the cluster)
# ---------------------------------------------------------------------------
def connect():
    import hopsworks

    project = hopsworks.login()
    return project


def get_ratings(project):
    """Latest elo + fifa ratings per country -> {ratings_name: {elo,rank,points}}."""
    fs = project.get_feature_store()
    elo = fs.get_feature_group("elo_ratings", version=1).read()
    fifa = fs.get_feature_group("fifa_ratings", version=1).read()
    elo["date"] = pd.to_datetime(elo["date"])
    fifa["date"] = pd.to_datetime(fifa["date"])
    elo_latest = elo.sort_values("date").groupby("country").last()
    fifa_latest = fifa.sort_values("date").groupby("country").last()
    ratings = {}
    for c in set(elo_latest.index) | set(fifa_latest.index):
        rec = {}
        if c in elo_latest.index:
            rec["elo"] = float(elo_latest.loc[c, "elo_rating"])
        if c in fifa_latest.index:
            rec["rank"] = float(fifa_latest.loc[c, "ranking"])
            rec["points"] = float(fifa_latest.loc[c, "points"])
        ratings[c] = rec
    return ratings


def get_schedule(project):
    fs = project.get_feature_store()
    return fs.get_feature_group("fifa2026_schedule_fixtures", version=1).read()


def load_model(project):
    """Download the best (by accuracy) registered model and load it locally."""
    import joblib

    mr = project.get_model_registry()
    model = mr.get_best_model("wc2026_match_predictor", "accuracy", "max")
    mdir = model.download()
    clf = joblib.load(f"{mdir}/model.pkl")
    return clf


# ---------------------------------------------------------------------------
# Strength table — one predict_proba for all ordered pairs (pure given a model)
# ---------------------------------------------------------------------------
def build_strength_table(clf, ratings, teams):
    """Return {(team, opp, home): (p_loss, p_draw, p_win)} for all ordered pairs.

    Feature rows are built via F.build_feature_row (the shared definition) and
    scored in a single predict_proba. Class order is model.classes_ = [0,1,2] =
    [loss, draw, win].
    """
    rows, keys = [], []
    for a in teams:
        for b in teams:
            if a == b:
                continue
            for home in (0, 1):
                rows.append(F.build_feature_row(a, b, home, 0, ratings))
                keys.append((a, b, home))
    X = pd.DataFrame(rows)[F.FEATURE_COLUMNS].astype(float)
    proba = clf.predict_proba(X)
    # Map model.classes_ -> column position so we are robust to class ordering.
    classes = list(getattr(clf, "classes_", [0, 1, 2]))
    li, di, wi = classes.index(0), classes.index(1), classes.index(2)
    table = {}
    for k, p in zip(keys, proba):
        table[k] = (float(p[li]), float(p[di]), float(p[wi]))
    return table


def match_probs(table, a, b):
    """(p_a_win, p_draw, p_a_loss) for a vs b, applying host home advantage."""
    a_host = a in F.HOSTS
    b_host = b in F.HOSTS
    if a_host and not b_host:
        pl, pd_, pw = table[(a, b, 1)]
        return pw, pd_, pl
    if b_host and not a_host:
        pl, pd_, pw = table[(b, a, 1)]  # b's perspective
        return pl, pd_, pw            # invert: a_win = b_loss
    pl, pd_, pw = table[(a, b, 0)]
    return pw, pd_, pl


def p_advance(table, a, b):
    """P(a beats b) in a knockout tie (neutral venue): win + 0.5*draw."""
    pl, pd_, pw = table[(a, b, 0)]
    return pw + 0.5 * pd_


# ---------------------------------------------------------------------------
# Tournament structure (pure, from the schedule)
# ---------------------------------------------------------------------------
def resolve_slot_token(token, ratings):
    """Resolve a playoff slot like 'Kosovo/Romania/Slovakia/Türkiye' to its
    best-rated (highest elo) candidate. Plain team names pass through."""
    token = token.strip()
    if "/" not in token:
        return token
    cands = [c.strip() for c in token.split("/")]
    best, best_elo = cands[0], -1.0
    for c in cands:
        elo, _, _ = F.lookup_team(c, ratings)
        if elo > best_elo:
            best, best_elo = c, elo
    return best


def parse_structure(schedule, ratings):
    """Parse groups, group fixtures and the knockout feeder tree from the schedule."""
    sched = schedule.copy()
    sched["n"] = sched["match_number"].str.extract(r"(\d+)").astype(int)
    sched = sched.sort_values("n")

    # ---- groups + group fixtures ----
    groups = {}
    group_fixtures = []  # (group_letter, teamA, teamB)
    grp_rows = sched[sched["group"].notna() & (sched["group"].astype(str).str.strip() != "")]
    for _, r in grp_rows.iterrows():
        letter = str(r["group"]).replace("Group", "").strip()
        a, b = [resolve_slot_token(t, ratings) for t in str(r["teams"]).split(" v ")]
        group_fixtures.append((letter, a, b))
        groups.setdefault(letter, set()).update([a, b])
    groups = {g: sorted(t) for g, t in groups.items()}

    # ---- knockout matches ----
    ko = {}  # match_num -> {'a': slot, 'b': slot}
    feeders = {}  # match_num -> [feeder match nums]
    for _, r in sched[sched["n"] >= 73].iterrows():
        n = int(r["n"])
        a_s, b_s = str(r["teams"]).split(" v ")
        ko[n] = {"a": a_s.strip(), "b": b_s.strip()}
        fa = re.findall(r"match (\d+)", a_s.lower())
        fb = re.findall(r"match (\d+)", b_s.lower())
        feeders[n] = [int(x) for x in fa + fb]

    teams = sorted({t for ts in groups.values() for t in ts})
    return {
        "groups": groups,
        "group_fixtures": group_fixtures,
        "ko": ko,
        "feeders": feeders,
        "teams": teams,
    }


def round_of(n):
    for name, rng in ROUND_OF.items():
        if n in rng:
            return name
    return "Final"  # 104 (and 103 = 3rd-place play-off handled separately)


# ---------------------------------------------------------------------------
# Simulation (pure: deterministic in chalk mode, rng-driven in random mode)
# ---------------------------------------------------------------------------
def simulate_group(letter, teams, group_fixtures, table, rng, chalk):
    """Return standings (ordered best->worst) and points dict for one group."""
    pts = {t: 0.0 for t in teams}
    exp = {t: 0.0 for t in teams}
    for g, a, b in group_fixtures:
        if g != letter:
            continue
        pw, pd_, pl = match_probs(table, a, b)
        exp[a] += 3 * pw + pd_
        exp[b] += 3 * pl + pd_
        if chalk:
            pts[a] += 3 * pw + pd_  # expected points = chalk seeding
            pts[b] += 3 * pl + pd_
        else:
            u = rng.random()
            if u < pw:
                pts[a] += 3
            elif u < pw + pd_:
                pts[a] += 1
                pts[b] += 1
            else:
                pts[b] += 3
    noise = {t: (0 if chalk else rng.random()) for t in teams}
    order = sorted(teams, key=lambda t: (pts[t], F.lookup_team(t, RATINGS_REF)[0],
                                         noise[t]), reverse=True)
    return order, pts, exp


# RATINGS_REF is set by the caller (tie-break by elo). Kept module-level so the
# pure functions stay simple; set once per session.
RATINGS_REF = {}


def assign_thirds(thirds_by_group, slots):
    """Bipartite-assign the 8 best third-placed teams to the 8 third-place slots.

    thirds_by_group: list of (group_letter, team, score) for ALL groups' thirds.
    slots: list of (match_num, allowed_groups_set). Returns {match_num: team}.
    """
    ranked = sorted(thirds_by_group, key=lambda x: x[2], reverse=True)[:8]
    qualified = {g: t for g, t, _ in ranked}  # group_letter -> team
    avail_groups = set(qualified)
    # order slots by fewest feasible groups first (constraint propagation)
    slots_sorted = sorted(slots, key=lambda s: len(s[1] & avail_groups))
    assignment = {}
    used = set()

    def backtrack(i):
        if i == len(slots_sorted):
            return True
        mnum, allowed = slots_sorted[i]
        for g in allowed & avail_groups:
            if g in used:
                continue
            used.add(g)
            assignment[mnum] = qualified[g]
            if backtrack(i + 1):
                return True
            used.discard(g)
            del assignment[mnum]
        return False

    if not backtrack(0):
        # fallback: assign remaining arbitrarily
        leftovers = [g for g in avail_groups if g not in used]
        for mnum, _ in slots_sorted:
            if mnum not in assignment and leftovers:
                assignment[mnum] = qualified[leftovers.pop()]
    return assignment


def simulate_tournament(structure, table, rng, chalk=False):
    """Run one full tournament. Returns a dict of results for bracket + stats."""
    groups = structure["groups"]
    ko = structure["ko"]

    # --- group stage ---
    standings = {}   # group -> ordered teams
    thirds = []      # (group, team, score)
    for g, teams in groups.items():
        order, pts, exp = simulate_group(g, teams, structure["group_fixtures"],
                                         table, rng, chalk)
        standings[g] = order
        thirds.append((g, order[2], pts[order[2]]))

    # --- assign best-8 thirds to the 8 third-place R32 slots ---
    slots = []
    for n, s in ko.items():
        for side in ("a", "b"):
            m = re.match(r"group ([a-l/]+) third place", s[side].lower())
            if m:
                allowed = {x.upper() for x in m.group(1).split("/")}
                slots.append((n, side, allowed))
    # group identical-slot constraints by match (each appears once)
    third_assign = assign_thirds(
        thirds, [(n, allowed) for n, side, allowed in slots]
    )

    # --- resolve a single slot string to a concrete team ---
    winner, loser = {}, {}

    def resolve(slot_str, mnum=None, side=None):
        s = slot_str.lower()
        m = re.match(r"group ([a-l]) winners?$", s)
        if m:
            return standings[m.group(1).upper()][0]
        m = re.match(r"group ([a-l]) runners?[- ]?up$", s)
        if m:
            return standings[m.group(1).upper()][1]
        if "third place" in s:
            return third_assign.get(mnum)
        m = re.match(r"winner match (\d+)", s)
        if m:
            return winner[int(m.group(1))]
        m = re.match(r"runner-?up match (\d+)", s)
        if m:
            return loser[int(m.group(1))]
        return slot_str  # already a concrete team

    # --- knockout matches in order ---
    ko_results = {}  # match_num -> (teamA, teamB, winner, loser)
    for n in sorted(ko):
        sa, sb = ko[n]["a"], ko[n]["b"]
        a = resolve(sa, n, "a")
        b = resolve(sb, n, "b")
        if a is None or b is None:
            continue
        pa = p_advance(table, a, b)
        if chalk:
            a_wins = pa >= 0.5
        else:
            a_wins = rng.random() < pa
        w, l = (a, b) if a_wins else (b, a)
        winner[n] = w
        loser[n] = l
        ko_results[n] = (a, b, w, l)

    champion = winner.get(104)
    third_place = winner.get(103)
    runner_up = loser.get(104)
    return {
        "standings": standings,
        "ko_results": ko_results,
        "champion": champion,
        "runner_up": runner_up,
        "third_place": third_place,
        "winner": winner,
        "loser": loser,
    }


# ---------------------------------------------------------------------------
# Monte Carlo aggregation
# ---------------------------------------------------------------------------
def monte_carlo(structure, table, n_sims, seed=0):
    teams = structure["teams"]
    reach = {t: {r: 0 for r in ROUND_ORDER} for t in teams}
    group_points = {t: 0.0 for t in teams}
    finish = {t: {"1st": 0, "2nd": 0} for t in teams}
    rng = random.Random(seed)

    # match -> round it FEEDS INTO (i.e. winning it reaches that round)
    for s in range(n_sims):
        res = simulate_tournament(structure, table, rng, chalk=False)
        # group points + finishes
        for g, order in res["standings"].items():
            # recompute points for averaging via a fresh deterministic pass would
            # double-sample; instead use the realized standings rank as proxy is
            # wrong — so we accumulate expected points separately below.
            finish[order[0]]["1st"] += 1
            finish[order[1]]["2nd"] += 1
        # Rounds reached: appearing in a match for round R means the team
        # advanced into R (resolve() fills each match with the feeders' winners).
        # Match 103 is the 3rd-place play-off, not a progression round.
        reached = {t: set() for t in teams}
        for m, (a, b, w, l) in res["ko_results"].items():
            if m == 103:
                continue
            rnd = round_of(m)  # R32 / R16 / QF / SF / Final
            for t in (a, b):
                reached[t].add(rnd)
        for t in teams:
            for rnd in reached[t]:
                reach[t][rnd] += 1
        if res["champion"]:
            reach[res["champion"]]["Won"] += 1

    # expected group points via a separate analytic pass (chalk expectation)
    exp_pts = compute_expected_group_points(structure, table)

    stats = {}
    for t in teams:
        row = {r: reach[t][r] / n_sims for r in ROUND_ORDER}
        row["exp_group_points"] = exp_pts.get(t, 0.0)
        row["p_first"] = finish[t]["1st"] / n_sims
        row["p_second"] = finish[t]["2nd"] / n_sims
        row["p_advance_group"] = row["p_first"] + row["p_second"]
        stats[t] = row
    return stats


def compute_expected_group_points(structure, table):
    exp = {t: 0.0 for ts in structure["groups"].values() for t in ts}
    for g, a, b in structure["group_fixtures"]:
        pw, pd_, pl = match_probs(table, a, b)
        exp[a] += 3 * pw + pd_
        exp[b] += 3 * pl + pd_
    return exp


# ---------------------------------------------------------------------------
# Bracket diagram (classic left/right, FINAL centered)
# ---------------------------------------------------------------------------
def _short(name, n=14):
    return name if len(name) <= n else name[: n - 1] + "…"


def draw_bracket(structure, result):
    import matplotlib.pyplot as plt

    feeders = structure["feeders"]
    ko_results = result["ko_results"]

    # Partition R32..SF matches into left (feeds 101) and right (feeds 102).
    def subtree(root):
        out = []
        stack = [root]
        while stack:
            m = stack.pop()
            out.append(m)
            stack.extend(feeders.get(m, []))
        return out

    left = set(subtree(101))
    right = set(subtree(102))

    # Assign y by in-order DFS of leaves; internal nodes = mean of children.
    ypos = {}
    counter = [0]

    def layout(m):
        ch = feeders.get(m, [])
        if not ch:  # R32 leaf
            ypos[m] = counter[0]
            counter[0] += 1
            return ypos[m]
        ys = [layout(c) for c in ch]
        ypos[m] = sum(ys) / len(ys)
        return ypos[m]

    layout(101)
    counter[0] += 1  # gap between halves
    layout(102)

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.axis("off")
    maxcol = 4

    final_y = (ypos[101] + ypos[102]) / 2
    ypos[104] = final_y

    def xy(m):
        col = ROUND_COL[round_of(m)]
        if m == 104:
            return maxcol, final_y
        x = col if m in left else (2 * maxcol - col)
        return x, ypos[m]

    box_w, box_h = 0.92, 0.78

    def draw_box(m):
        if m not in ko_results:
            return
        a, b, w, l = ko_results[m]
        x, y = xy(m)
        for i, team in enumerate((a, b)):
            yy = y + (0.22 if i == 0 else -0.22)
            is_w = team == w
            ax.add_patch(plt.Rectangle((x - box_w / 2, yy - box_h / 4),
                                       box_w, box_h / 2,
                                       facecolor="#1EB182" if is_w else "#1A1F2B",
                                       edgecolor="#444", lw=0.6, zorder=2))
            ax.text(x, yy, _short(team), ha="center", va="center",
                    fontsize=7.5, color="white",
                    fontweight="bold" if is_w else "normal", zorder=3)
        # connector to the match this one feeds into (skip the 3rd-place box)
        for parent, ch in feeders.items():
            if parent == 103:
                continue
            if m in ch:
                px, py = xy(parent)
                ax.plot([x + (box_w / 2 if x < px else -box_w / 2), px],
                        [y, py], color="#555", lw=0.6, zorder=1)
                break

    for m in sorted(ko_results):
        if m in (103,):
            continue
        draw_box(m)

    # Round headers
    headers = [("Round of 32", 0), ("Round of 16", 1), ("Quarter-finals", 2),
               ("Semi-finals", 3), ("FINAL", 4)]
    ytop = max(ypos.values()) + 1.0
    for label, col in headers:
        if col == maxcol:
            ax.text(maxcol, ytop, label, ha="center", fontsize=10,
                    fontweight="bold", color="#1EB182")
        else:
            ax.text(col, ytop, label, ha="center", fontsize=9, color="#aaa")
            ax.text(2 * maxcol - col, ytop, label, ha="center", fontsize=9, color="#aaa")

    # Champion banner + 3rd-place box
    champ = result.get("champion")
    if champ:
        ax.text(maxcol, ytop + 0.8, f"CHAMPION: {champ}", ha="center",
                fontsize=14, fontweight="bold", color="#FFD700")
    if 103 in ko_results:
        a, b, w, l = ko_results[103]
        ymin = min(ypos.values()) - 1.4
        ax.text(maxcol, ymin + 0.4, "Third-place play-off", ha="center",
                fontsize=8, color="#aaa")
        ax.add_patch(plt.Rectangle((maxcol - box_w / 2, ymin - 0.4),
                                   box_w, 0.7, facecolor="#1A1F2B",
                                   edgecolor="#888", lw=0.7))
        ax.text(maxcol, ymin, f"{_short(w)} (W)\n{_short(l)}", ha="center",
                va="center", fontsize=7, color="#CD7F32")

    ax.set_xlim(-1, 2 * maxcol + 1)
    ax.set_ylim(min(ypos.values()) - 2.5, ytop + 1.5)
    fig.patch.set_facecolor("#0E1117")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def main():
    import streamlit as st

    st.set_page_config(page_title="WC 2026 Simulator", layout="wide")
    st.markdown(
        """
        <style>
          .hw-band {background:linear-gradient(90deg,#0E1117,#1A1F2B);
                    border-left:6px solid #1EB182; padding:0.75rem 1rem;
                    border-radius:6px; margin-bottom:1rem;}
          .hw-band h1 {color:#FAFAFA; margin:0; font-size:1.5rem;}
          div[data-testid="stMetricValue"] {color:#1EB182;}
          .stButton>button {background:#1EB182; color:#0E1117; border:none; font-weight:600;}
        </style>
        <div class="hw-band"><h1>⬡ FIFA World Cup 2026 — Match-Prediction Simulator</h1></div>
        """,
        unsafe_allow_html=True,
    )

    @st.cache_resource
    def _bootstrap():
        project = connect()
        clf = load_model(project)
        ratings = get_ratings(project)
        schedule = get_schedule(project)
        structure = parse_structure(schedule, ratings)
        global RATINGS_REF
        RATINGS_REF = ratings
        table = build_strength_table(clf, ratings, structure["teams"])
        return ratings, structure, table

    with st.spinner("Loading model, ratings and schedule from Hopsworks…"):
        ratings, structure, table = _bootstrap()
    global RATINGS_REF
    RATINGS_REF = ratings

    st.sidebar.header("Simulation settings")
    n_sims = st.sidebar.select_slider("Monte-Carlo rollouts",
                                      options=[200, 500, 1000, 2000, 5000], value=1000)
    if st.sidebar.button("🔄 Re-run simulation"):
        st.session_state["seed"] = st.session_state.get("seed", 0) + 1
    seed = st.session_state.get("seed", 0)

    @st.cache_data(show_spinner=False)
    def _mc(n, seed):
        return monte_carlo(structure, table, n, seed)

    with st.spinner(f"Running {n_sims} Monte-Carlo rollouts…"):
        stats = _mc(n_sims, seed)

    tab1, tab2, tab3 = st.tabs(["🏆 Tournament odds", "🅰️ Group stage", "🗺️ Knockout bracket"])

    # ---- tournament odds ----
    with tab1:
        st.subheader("Each team's route to the final")
        df = pd.DataFrame(stats).T
        show = df[["R16", "QF", "SF", "Final", "Won"]].copy()
        show.columns = ["Reach R16", "Reach QF", "Reach SF", "Reach Final", "Win Cup"]
        show = (show * 100).round(1).sort_values("Win Cup", ascending=False)
        st.dataframe(show.style.format("{:.1f}%").background_gradient(
            cmap="Greens", subset=["Win Cup"]), height=520, use_container_width=True)
        top = show.head(8)
        st.bar_chart(top["Win Cup"])

    # ---- group stage ----
    with tab2:
        st.subheader("Group-stage outlook (expected points & advance probability)")
        cols = st.columns(3)
        for i, g in enumerate(sorted(structure["groups"])):
            with cols[i % 3]:
                st.markdown(f"**Group {g}**")
                rows = []
                for t in structure["groups"][g]:
                    s = stats[t]
                    rows.append({"Team": t,
                                 "xPts": round(s["exp_group_points"], 2),
                                 "P(adv)": f"{100*s['p_advance_group']:.0f}%"})
                gdf = pd.DataFrame(rows).sort_values("xPts", ascending=False)
                st.dataframe(gdf, hide_index=True, use_container_width=True)

    # ---- bracket ----
    with tab3:
        mode = st.radio("Bracket mode", ["Most-likely (chalk)", "One random simulation"],
                        horizontal=True)
        if mode == "Most-likely (chalk)":
            result = simulate_tournament(structure, table, random.Random(0), chalk=True)
        else:
            result = simulate_tournament(structure, table, random.Random(1000 + seed),
                                         chalk=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("Champion", result.get("champion") or "—")
        c2.metric("Runner-up", result.get("runner_up") or "—")
        c3.metric("Third place", result.get("third_place") or "—")
        fig = draw_bracket(structure, result)
        st.pyplot(fig, use_container_width=True)

        st.divider()
        st.caption("Run your own single rollout (with upsets):")
        if st.button("🎲 Roll one random tournament"):
            st.session_state["my_seed"] = st.session_state.get("my_seed", 0) + 1
        my = simulate_tournament(structure, table,
                                 random.Random(7000 + st.session_state.get("my_seed", 0)),
                                 chalk=False)
        st.success(f"Your rollout champion: **{my.get('champion')}**  "
                   f"(runner-up {my.get('runner_up')}, 3rd {my.get('third_place')})")


if __name__ == "__main__":
    main()
