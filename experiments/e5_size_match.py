"""E5 against size matching with earlier E2 frames (ICCE 2027 review: Reviewer 1, question 3; Reviewer 2,
generalisability of the attack evaluation).

    python experiments/e5_size_match.py                                        # synthetic scene, quick
    python experiments/e5_size_match.py --video clip.mp4 --starts 4 --gaps 1 2 5 10 25 --stride 2 --name clip
    python experiments/e5_size_match.py --summarise sim_out/e5_size_match      # one table over all sources

THREAT. The adaptive policy runs E2 and switches to E5 when it sees an attack. The eavesdropper recorded the
last E2 frame t0 (E2 shows the outline: the positions of non-zero coefficients are public) and then receives
E5 frames t0+g. In E5 (hardware form, docs/INTERFACES.md section 8.6) every MCU is one restart segment and the
segments arrive in a keyed order that is new in every frame. What stays public about each segment: its length
in bytes and its Huffman structure (run/size of every symbol, hence the non-zero AC count of each block).
The attacker uses the old E2 frame as a "box picture" and puts every received segment at the position whose
E2 MCU looks most alike, then runs the SVE-Bench attacks on the placed frame:
  A2 replacement  encrypted values replaced by fixed guesses, re-coded, decoded; edge correlation
  A3 NZCA         non-zero AC count per 8x8 luma block, correlated with the block edge energy
(docs/BENCHMARK_METHOD.md; the helpers are imported from experiments/benchmark_suite.py, so the scoring is the
same as in the benchmark.)

ATTACKERS (placement)
  rank_bytes   segments sorted by byte length, E2 MCUs sorted by coded length; i-th longest to i-th busiest
  assign_nz    6-vector of non-zero AC counts (4 Y blocks, Cb, Cr) per segment and per E2 MCU, plus the length;
               minimum-cost assignment (Hungarian algorithm, scipy) on the squared distance of the standardised
               features
REFERENCE POINTS, scored against the same frame t0+g
  e2_old       the attacker's best view of the OLD E2 frame alone (A1, A2, A3 of frame t0): what it already had,
               which E5 cannot take back
  random       random placement (chance)
  oracle       the true placement (what E4 would leak; the upper bound for any placement attack)
LEAK = the larger of A2 and A3 (as SVE-Bench's max leak). GAIN = attack leak - e2_old leak: the outline of the
new frame that the attacker gets beyond what it already had. MOTION = correlation of (placed non-zero map - old
E2 non-zero map) with |block edge energy now - at t0|: does the attacker see WHAT CHANGED since the switch
(a person who walked in), or only the old outline again? The oracle's motion shows how much there was to see. Also reported: the share of segments placed at
their true MCU, and within one MCU of it. Means come with 95 % bootstrap intervals over (start, gap) pairs.
Writes sim_out/e5_size_match/<name>.json and <name>.md.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "software")]
sys.path.insert(0, str(ROOT / "experiments"))
from benchmark_suite import (KEY, SID, block_energy, bootstrap_ci, corr, decode, edge_corr,   # noqa: E402
                             luma, native_size, nz_map, rebuild, received_segments, replace_fields,
                             split_blocks)
from link.sources import open_source                                                  # noqa: E402
from model import hw                                                                  # noqa: E402
from model.container import parse_jpeg                                                # noqa: E402
from model.entropy import huffman_encode, split_restart                               # noqa: E402
from model.jpeg import analyse_frame, package_frame                                   # noqa: E402

E2F = dict(ac_sign=True, dc_sign=True)
E4F = dict(ac_sign=True, ac_all=True, dc_sign=True, dc_all=True)
ATTACKERS = ("rank_bytes", "assign_nz")
REFS = ("e2_old", "random", "oracle")


# ------------------------------------------------------------------ public features
def nz_counts(mcu_symbols):
    """Non-zero AC count of each of the 6 blocks of one MCU (what the Huffman structure shows)."""
    return [sum(1 for s in b if not s.is_dc and s.size and not s.eob) for b in split_blocks(mcu_symbols)]


def e2_features(jpeg):
    """Per MCU of the old E2 frame (raster order): coded length in bytes and the 6 non-zero counts."""
    w, h, qy, qc, ri, segs = received_segments(jpeg)
    blocks = split_blocks([s for seg in segs for s in seg])
    mcus = [sum(blocks[6 * m:6 * m + 6], []) for m in range(len(blocks) // 6)]
    return (np.array([len(huffman_encode(m)[0]) for m in mcus], float),
            np.array([nz_counts(m) for m in mcus], float))


def e5_features(jpeg):
    """Per received E5 segment (arrival order): length in bytes and the 6 non-zero counts."""
    p = parse_jpeg(jpeg)
    w, h, qy, qc, ri, segs = received_segments(jpeg)
    lengths = np.array([len(s) for s in split_restart(p.scan)], float)
    return lengths, np.array([nz_counts(s) for s in segs], float), (w, h, qy, qc, ri, segs)


# ------------------------------------------------------------------ placement attacks
def place_rank(e2_len, e5_len, rng):
    """received segment j -> MCU position: i-th longest segment to the i-th longest E2 MCU (random ties)."""
    n = len(e5_len)
    seg_order = np.lexsort((rng.random(n), -e5_len))
    pos_order = np.lexsort((rng.random(n), -e2_len))
    a = np.empty(n, int)
    a[seg_order] = pos_order
    return a


def place_assign(e2_len, e2_nz, e5_len, e5_nz):
    """Minimum-cost assignment of segments to positions on standardised (length, 6 counts) features."""
    f2 = np.column_stack([e2_len, e2_nz])
    f5 = np.column_stack([e5_len, e5_nz])
    mu, sd = np.vstack([f2, f5]).mean(0), np.vstack([f2, f5]).std(0) + 1e-9
    f2, f5 = (f2 - mu) / sd, (f5 - mu) / sd
    cost = ((f5[:, None, :] - f2[None, :, :]) ** 2).sum(-1)
    rows, cols = linear_sum_assignment(cost)
    a = np.empty(len(f5), int)
    a[rows] = cols
    return a


# ------------------------------------------------------------------ scoring
def placed_views(parts, assign, y, energy, old_nz, change):
    """A2 and A3 on the frame with received segment j put at MCU assign[j]; motion = how well the placed
    frame's non-zero map MINUS the old E2 frame's map follows where the scene really changed."""
    w, h, qy, qc, ri, segs = parts
    nat = [None] * len(segs)
    for j, pos in enumerate(assign):
        nat[pos] = segs[j]
    a2 = edge_corr(y, decode(rebuild(w, h, qy, qc, ri, [replace_fields(s, **E4F) for s in nat])))
    nz = nz_map(w, nat, ri)
    a3 = corr(nz, energy)
    return dict(replacement=a2, nzca=a3, leak=max(a2, a3), motion=corr(nz - old_nz, change))


def old_e2_views(jpeg, y, energy):
    """The attacker's best picture from the old E2 frame alone, scored against the new frame."""
    w, h, qy, qc, ri, segs = received_segments(jpeg)
    a1 = edge_corr(y, decode(jpeg))
    a2 = edge_corr(y, decode(rebuild(w, h, qy, qc, ri, [replace_fields(s, **E2F) for s in segs])))
    a3 = corr(nz_map(w, segs, ri), energy)
    return dict(decode=a1, replacement=a2, nzca=a3, leak=max(a1, a2, a3))


def placement_accuracy(assign, truth, cols):
    exact = float(np.mean(assign == truth))
    (ra, ca), (rt, ct) = divmod(assign, cols), divmod(truth, cols)
    near = float(np.mean(np.maximum(abs(ra - rt), abs(ca - ct)) <= 1))
    return exact, near


def read_frames(spec, w, h, n):
    src = open_source(spec, w, h, loop=False)
    out = []
    for _ in range(n):
        f = src.read()
        if f is None:
            break
        out.append(f)
    return out


def run(a):
    w, h = native_size(a.video) if a.size == "native" else map(int, a.size.split("x"))
    span = max(a.gaps) * a.stride
    need = span + 1 + (a.starts - 1) * a.start_step
    frames = read_frames(a.video, w, h, need)
    starts = [s * a.start_step for s in range(a.starts) if s * a.start_step + span < len(frames)]
    if not starts:
        raise SystemExit(f"{a.video}: {len(frames)} frames, need at least {span + 1}")
    rng = np.random.default_rng(a.seed)
    cols = (w + 15) // 16
    per = {g: [] for g in a.gaps}
    ctr = 1
    for t0 in starts:
        an0 = analyse_frame(frames[t0], a.quality)
        e2 = package_frame(an0, KEY, SID, ctr, "E2").jpeg
        ctr += 1
        e2_len, e2_nz = e2_features(e2)
        w2, _, _, _, ri2, segs2 = received_segments(e2)
        old_nz = nz_map(w2, segs2, ri2)
        energy0 = block_energy(luma(frames[t0]), (an0.height // 8, an0.width // 8))
        for g in a.gaps:
            rgb = frames[t0 + g * a.stride]
            y = luma(rgb)
            an = analyse_frame(rgb, a.quality)
            energy = block_energy(y, (an.height // 8, an.width // 8))
            e5 = package_frame(an, KEY, SID, ctr, "E5", segmented=True).jpeg
            truth = np.array(hw.e5_order(KEY, SID, ctr, an.n_mcu))      # received position p carries MCU truth[p]
            ctr += 1
            e5_len, e5_nz, parts = e5_features(e5)
            change = np.abs(energy - energy0)               # where the picture changed since t0
            row = dict(start=t0, gap=g, e2_old=old_e2_views(e2, y, energy))
            for name, assign in (("rank_bytes", place_rank(e2_len, e5_len, rng)),
                                 ("assign_nz", place_assign(e2_len, e2_nz, e5_len, e5_nz)),
                                 ("random", rng.permutation(an.n_mcu)),
                                 ("oracle", truth)):
                row[name] = placed_views(parts, assign, y, energy, old_nz, change)
                row[name]["exact"], row[name]["near"] = placement_accuracy(np.asarray(assign), truth, cols)
            best = max(ATTACKERS, key=lambda k: row[k]["leak"])
            row["best_attack"] = best
            row["gain"] = row[best]["leak"] - row["e2_old"]["leak"]
            per[g].append(row)
            print(f"t0={t0:4d} gap={g:3d}  e2_old {row['e2_old']['leak']:.3f}  rank {row['rank_bytes']['leak']:.3f}"
                  f"  assign {row['assign_nz']['leak']:.3f} (exact {row['assign_nz']['exact']:.1%})"
                  f"  random {row['random']['leak']:.3f}  oracle {row['oracle']['leak']:.3f}", flush=True)
    return dict(name=a.name, source=str(a.video), size=f"{w}x{h}", quality=a.quality, stride=a.stride,
                starts=starts, gaps=a.gaps, summary=summarise_rows([r for g in a.gaps for r in per[g]]),
                by_gap={str(g): summarise_rows(per[g]) for g in a.gaps},
                rows=[r for g in a.gaps for r in per[g]])


def summarise_rows(rows):
    s = dict(n=len(rows))
    for k in ATTACKERS + REFS:
        for m in ("leak", "replacement", "nzca") + (("exact", "near", "motion") if k != "e2_old" else ()):
            vals = [r[k][m] for r in rows]
            s[f"{k}.{m}"] = [float(np.mean(vals))] + bootstrap_ci(vals)
    s["best_attack.leak"] = [float(np.mean([r[r["best_attack"]]["leak"] for r in rows]))] + \
        bootstrap_ci([r[r["best_attack"]]["leak"] for r in rows])
    s["gain"] = [float(np.mean([r["gain"] for r in rows]))] + bootstrap_ci([r["gain"] for r in rows])
    s["gain.max"] = float(max(r["gain"] for r in rows))
    mo = [max(r[k]["motion"] for k in ATTACKERS) for r in rows]
    s["best_attack.motion"] = [float(np.mean(mo))] + bootstrap_ci(mo)
    return s


def fmt(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def markdown(res):
    out = [f"# E5 against size matching with an earlier E2 frame ({res['name']}, {res['size']}, quality "
           f"{res['quality']}, starts {res['starts']}, gaps {res['gaps']} x {res['stride']} source frames)", "",
           "Leak = max of A2 replacement and A3 NZCA (edge correlation with frame t0+g, 0 = nothing); "
           "e2_old = the old E2 frame alone. Mean [95 % CI].", "",
           "| Gap | n | e2_old (had already) | rank_bytes | assign_nz | best attack | gain over e2_old | motion: attack | motion: oracle | random | oracle (true order) | placed exactly (assign_nz) | within one MCU |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, s in list(res["by_gap"].items()) + [("all", res["summary"])]:
        out.append(f"| {g} | {s['n']} | {fmt(s['e2_old.leak'])} | {fmt(s['rank_bytes.leak'])} | {fmt(s['assign_nz.leak'])} "
                   f"| {fmt(s['best_attack.leak'])} | {fmt(s['gain'])} | {fmt(s['best_attack.motion'])} | {fmt(s['oracle.motion'])} "
                   f"| {fmt(s['random.leak'])} | {fmt(s['oracle.leak'])} "
                   f"| {s['assign_nz.exact'][0]:.1%} | {s['assign_nz.near'][0]:.1%} |")
    return "\n".join(out) + "\n"


def summarise_dir(d):
    """One table over every <name>.json in d (all rows pooled, plus one line per source)."""
    files = sorted(p for p in Path(d).glob("*.json") if p.stem != "summary")
    if not files:
        raise SystemExit(f"no results in {d}")
    res = [json.loads(p.read_text()) for p in files]
    rows = [r for x in res for r in x["rows"]]
    pooled = summarise_rows(rows)
    lines = ["# E5 against size matching with earlier E2 frames: all sources", "",
             "Leak = max of A2 and A3 (edge correlation with the new frame, 0 = nothing). Mean [95 % CI] over "
             "(start, gap) pairs.", "",
             "| Source | n | e2_old (had already) | best size-matching attack | gain over e2_old | max gain | motion: attack | motion: oracle | random | oracle | placed exactly | within one MCU |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in res + [dict(name="**all sources**", summary=pooled)]:
        s = x["summary"]
        lines.append(f"| {x['name']} | {s['n']} | {fmt(s['e2_old.leak'])} | {fmt(s['best_attack.leak'])} | {fmt(s['gain'])} "
                     f"| {s['gain.max']:.3f} | {fmt(s['best_attack.motion'])} | {fmt(s['oracle.motion'])} "
                     f"| {fmt(s['random.leak'])} | {fmt(s['oracle.leak'])} "
                     f"| {s['assign_nz.exact'][0]:.1%} | {s['assign_nz.near'][0]:.1%} |")
    md = "\n".join(lines) + "\n"
    (Path(d) / "summary.md").write_text(md)
    (Path(d) / "summary.json").write_text(json.dumps(dict(sources=[x["name"] for x in res], pooled=pooled), indent=1))
    print(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default="synthetic")
    ap.add_argument("--size", default="320x240", help="WxH or native")
    ap.add_argument("--quality", type=int, default=75)
    ap.add_argument("--gaps", type=int, nargs="+", default=[1, 2, 5, 10],
                    help="E5 frames scored after the last E2 frame, in units of --stride source frames")
    ap.add_argument("--stride", type=int, default=3, help="source frames per transmitted frame")
    ap.add_argument("--starts", type=int, default=2, help="how many switch points t0 in the clip")
    ap.add_argument("--start-step", type=int, default=60, help="source frames between switch points")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="synthetic")
    ap.add_argument("--out", default="sim_out/e5_size_match")
    ap.add_argument("--summarise", metavar="DIR", help="only pool the results already in DIR")
    a = ap.parse_args()
    if a.summarise:
        summarise_dir(a.summarise)
        return
    res = run(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(res, indent=1))
    md = markdown(res)
    (out / f"{a.name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
