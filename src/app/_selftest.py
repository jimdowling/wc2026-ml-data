#!/usr/bin/env python3
"""Offline self-test for app.py — runs against the live cluster but WITHOUT
Streamlit, so render/logic bugs are caught before a (slow) deploy cycle.

Run:  python src/app/_selftest.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))
import app  # noqa: E402


def main():
    project = app.connect()
    clf = app.load_model(project)
    ratings = app.get_ratings(project)
    schedule = app.get_schedule(project)
    structure = app.parse_structure(schedule, ratings)
    app.RATINGS_REF = ratings

    teams = structure["teams"]
    assert len(teams) == 48, f"expected 48 teams, got {len(teams)}"
    assert len(structure["groups"]) == 12, "expected 12 groups"
    for g, ts in structure["groups"].items():
        assert len(ts) == 4, f"group {g} has {len(ts)} teams"
    # no unresolved playoff slots remain
    assert not any("/" in t for t in teams), "unresolved '/' slot in teams"
    print(f"[ok] 48 teams in 12 groups of 4; slots resolved")

    table = app.build_strength_table(clf, ratings, teams)
    assert len(table) == 48 * 47 * 2, f"table size {len(table)}"
    # probabilities sum to ~1
    for k, (pl, pd_, pw) in list(table.items())[:50]:
        assert abs(pl + pd_ + pw - 1.0) < 1e-5, f"probs don't sum to 1: {k}"
    print(f"[ok] strength table: {len(table)} ordered pairs, probs normalised")

    # chalk tournament
    import random
    chalk = app.simulate_tournament(structure, table, random.Random(0), chalk=True)
    assert chalk["champion"] in teams, "no champion in chalk run"
    assert chalk["runner_up"] in teams
    assert chalk["third_place"] in teams
    # all R32..Final matches resolved (16+8+4+2+1 + 3rd = 32 matches 73..104)
    assert len(chalk["ko_results"]) == 32, f"{len(chalk['ko_results'])} KO matches"
    print(f"[ok] chalk champion = {chalk['champion']}, "
          f"runner-up = {chalk['runner_up']}, 3rd = {chalk['third_place']}")

    # random tournament differs across seeds (sanity, not guaranteed but typical)
    champs = {app.simulate_tournament(structure, table, random.Random(s))["champion"]
              for s in range(20)}
    print(f"[ok] {len(champs)} distinct champions across 20 random rollouts")

    # monte carlo invariants
    stats = app.monte_carlo(structure, table, 300, seed=0)
    p_champ = sum(s["Won"] for s in stats.values())
    assert abs(p_champ - 1.0) < 1e-6, f"P(champion) sums to {p_champ}, not 1.0"
    # round-reach monotonicity per team: R16 >= QF >= SF >= Final >= Won >= 0
    for t, s in stats.items():
        seq = [s["R16"], s["QF"], s["SF"], s["Final"], s["Won"]]
        assert all(seq[i] + 1e-9 >= seq[i + 1] for i in range(len(seq) - 1)), \
            f"{t} non-monotone reach: {seq}"
        assert all(v >= 0 for v in seq), f"{t} negative prob"
    # advance prob in [0,1]
    for t, s in stats.items():
        assert 0 <= s["p_advance_group"] <= 1.0001
    top = sorted(stats.items(), key=lambda kv: kv[1]["Won"], reverse=True)[:5]
    print("[ok] MC invariants hold. Top-5 by P(win):")
    for t, s in top:
        print(f"     {t:<20} win={100*s['Won']:.1f}%  SF={100*s['SF']:.1f}%  "
              f"xPts={s['exp_group_points']:.2f}")

    # bracket renders to PNG (chalk + random)
    for label, res in [("chalk", chalk),
                       ("random", app.simulate_tournament(structure, table,
                                                          random.Random(5)))]:
        fig = app.draw_bracket(structure, res)
        out = f"/tmp/bracket_{label}.png"
        fig.savefig(out, dpi=90, facecolor="#0E1117")
        assert os.path.getsize(out) > 5000, f"bracket {label} png too small"
        print(f"[ok] bracket ({label}) rendered -> {out}")

    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
