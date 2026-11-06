#!/usr/bin/env python3
"""Regenerate the README figures from run summaries in artifacts/ and docs/results/.

Usage:  python docs/make_figures.py
Writes light and dark PNGs to docs/figures/.
"""
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "models"
RESULTS = ROOT / "docs" / "results"
OUT = ROOT / "docs" / "figures"

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", grid="#e4e3df",
                  s1="#2a78d6", s2="#eb6834"),
    "dark": dict(surface="#1a1a19", text="#f0efec", text2="#c3c2b7", grid="#34332f",
                 s1="#3987e5", s2="#d95926"),
}


def val_acc(run: str) -> float:
    txt = (MODELS / run / "summary.txt").read_text()
    return float(re.search(r"Best epoch:.*?Val acc:\s+([\d.]+)", txt, re.S).group(1))


def val_n(run: str) -> int:
    txt = (MODELS / run / "summary.txt").read_text()
    return int(re.search(r"Val samples:\s+(\d+)", txt).group(1))


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], length=0, labelsize=10)
    ax.xaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def title(fig, t, main, sub):
    fig.text(0.02, 0.96, main, fontsize=13, weight="bold", color=t["text"], va="top")
    fig.text(0.02, 0.895, sub, fontsize=10, color=t["text2"], va="top")


def fig_generalization(t, name):
    rows = [  # (label, run, unseen-guitar?)
        ("Note model, 4 datasets, random split", "note_pilot_v2", False),
        ("Note model, 4 datasets, random split (rerun)", "note_pilot_v2_2", False),
        ("Note model, 6 datasets, random split", "note_pilot_v3", False),
        ("Note model, unseen guitar (train Strats, test RG550)", "note_pilot_v2_1", True),
        ("String model, 4 datasets, random split", "string_pilot_v1", False),
        ("String model, 6 datasets, random split", "string_20251218_160144", False),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor=t["surface"])
    fig.subplots_adjust(left=0.39, right=0.84, top=0.76, bottom=0.12)
    style(ax, t)
    ys = list(range(len(rows)))[::-1]
    for y, (label, run, held) in zip(ys, rows):
        a = val_acc(run) * 100
        ax.barh(y, a, height=0.55, color=t["s2"] if held else t["s1"])
        ax.text(a + 0.15, y, f"{a:.1f}%  (n={val_n(run):,})", va="center", fontsize=9.5, color=t["text"])
    ax.set_yticks(ys, [r[0] for r in rows], color=t["text"])
    ax.set_xlim(85, 100)
    ax.set_xlabel("Validation accuracy (%)  — axis starts at 85%", color=t["text2"], fontsize=9.5)
    h1 = plt.Rectangle((0, 0), 1, 1, color=t["s1"]); h2 = plt.Rectangle((0, 0), 1, 1, color=t["s2"])
    leg = ax.legend([h1, h2], ["Same guitars in train and val", "Val guitar unseen in training"],
                    loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=9.5)
    for tx in leg.get_texts():
        tx.set_color(t["text2"])
    title(fig, t, "Accuracy drops ~5 points on a guitar the model has never heard",
          "Best-epoch validation accuracy per training run")
    fig.savefig(OUT / name, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def fig_disambig(t, name):
    pair_runs = sorted(glob.glob(str(MODELS / "disambig_*" / "batch_summary.json")))
    pairs = {}
    for f in pair_runs:
        for k, v in json.load(open(f))["results"].items():
            pairs.setdefault(k, []).append(v["best_val_acc"] * 100)
    trip = json.load(open(RESULTS / "triplet_runs.json"))["runs"]
    trips, tn = {}, {}
    for run in trip.values():
        if len(run) < 4:  # skip incomplete batches
            continue
        for k, v in run.items():
            trips.setdefault(k, []).append(v["best_val_acc"] * 100)
            tn[k] = v.get("n_val")
    order = [("E_A", pairs), ("A_D", pairs), ("D_G", pairs), ("G_B", pairs), ("B_e", pairs),
             ("E_A_D", trips), ("A_D_G", trips), ("D_G_B", trips), ("G_B_e", trips)]
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=t["surface"])
    fig.subplots_adjust(left=0.12, right=0.95, top=0.76, bottom=0.13)
    style(ax, t)
    ys = list(range(len(order)))[::-1]
    for y, (k, src) in zip(ys, order):
        vals = src[k]
        c = t["s1"] if src is pairs else t["s2"]
        ax.plot([min(vals), max(vals)], [y, y], color=c, linewidth=2, solid_capstyle="round", alpha=0.6)
        ax.scatter(vals, [y] * len(vals), s=55, color=c, edgecolor=t["surface"], linewidth=2, zorder=3)
        spread = max(vals) - min(vals)
        label = f"range {spread:.0f} pts" + (f"  (n={tn[k]} val)" if src is trips else "")
        ax.text(101.2, y, label, va="center", fontsize=9, color=t["text2"])
    ax.set_yticks(ys, [k.replace("_", "/") for k, _ in order], color=t["text"])
    ax.set_xlim(68, 101)
    ax.set_xlabel("Best validation accuracy per rerun (%)", color=t["text2"], fontsize=9.5)
    sc1 = ax.scatter([], [], s=55, color=t["s1"]); sc2 = ax.scatter([], [], s=55, color=t["s2"])
    leg = ax.legend([sc1, sc2], ["2-string models (n=96–266 val)", "3-string models"], loc="lower left",
                    bbox_to_anchor=(-0.02, 1.0), ncol=2, frameon=False, fontsize=9.5)
    for tx in leg.get_texts():
        tx.set_color(t["text2"])
    fig.subplots_adjust(right=0.80)
    worst = max(max(trips[k]) - min(trips[k]) for k in trips)
    title(fig, t, f"3-string disambiguators swing up to {worst:.0f} points between reruns",
          "Each dot is one training run; small validation sets make single runs unreliable")
    fig.savefig(OUT / name, dpi=200, facecolor=t["surface"])
    plt.close(fig)


def fig_dataset(t, name):
    ds = json.load(open(RESULTS / "datasets.json"))["datasets"]
    fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=t["surface"])
    fig.subplots_adjust(left=0.27, right=0.9, top=0.76, bottom=0.14)
    style(ax, t)
    ys = list(range(len(ds)))[::-1]
    for y, d in zip(ys, ds):
        ax.barh(y, d["clips"], height=0.55, color=t["s1"])
        ax.text(d["clips"] + 60, y, f"{d['clips']:,} clips · {d['minutes']:.0f} min · frets {d['fret_min']}–{d['fret_max']}",
                va="center", fontsize=9, color=t["text"])
    ax.set_yticks(ys, [d["dataset"] for d in ds], color=t["text"], fontsize=9.5)
    ax.set_xlim(0, 8200)
    ax.set_xlabel("Labeled single-note clips", color=t["text2"], fontsize=9.5)
    tot = sum(d["clips"] for d in ds); mins = sum(d["minutes"] for d in ds)
    title(fig, t, f"{tot:,} hand-recorded clips, ~{mins/60:.0f} hours of audio",
          "Self-collected with tools/data_collection.py · all 6 strings · 48 kHz")
    fig.savefig(OUT / name, dpi=200, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, t in THEMES.items():
        fig_generalization(t, f"generalization_{mode}.png")
        fig_disambig(t, f"disambiguators_{mode}.png")
        fig_dataset(t, f"dataset_{mode}.png")
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
