"""experiments/restart_interval.py, experiments/standard_metrics.py and the image-folder source."""
import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import restart_interval as ri      # noqa: E402
import standard_metrics as sm      # noqa: E402

from link.sources import open_source   # noqa: E402
from model import ascon                # noqa: E402
from model.entropy import huffman_encode, run_length_encode   # noqa: E402
from model.jpeg import analyse_frame, package_frame, receive_frame   # noqa: E402
from model.selective import apply_keystream   # noqa: E402
from model.testimage import make_image   # noqa: E402

KEY = bytes(range(16))


def dec(jpeg):
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"))


def test_restart_interval_encoder_decodes_like_the_model():
    img = make_image(80, 48, "scene")
    an = analyse_frame(img)
    plain = dec(package_frame(an).jpeg)
    for n in (1, 2, 4, 32):
        jpeg, perm, _, order = ri.encode_ri(an, n, key=None, shuffle=False)
        got = dec(jpeg)
        assert np.array_equal(got, plain), f"N={n}: first differing pixel {np.argwhere(got != plain)[0]}"
        assert perm == list(range(len(an.quant))) and order == list(range(len(order)))


def test_key_holder_can_invert_the_shuffled_restart_interval_stream():
    an = analyse_frame(make_image(64, 48, "scene"))
    n = 2
    _, perm, shown, order = ri.encode_ri(an, n, ctr=5)
    natural = [None] * len(order)
    for pos, g in enumerate(order):
        natural[g] = shown[pos]
    flat = [s for g in natural for s in g]
    flat, _ = apply_keystream(flat, ascon.Keystream(KEY, ascon.make_nonce(7, 5, 0)), "E4")   # XOR again
    groups = ri.groups_of(len(an.quant), n)
    expected = [s for g in groups for s in run_length_encode(an.quant[g.start:g.stop])]
    assert flat == expected
    assert sorted(perm) == list(range(len(an.quant)))


def test_nzca_reveals_outline_unless_shuffled():
    img = make_image(160, 112, "scene")
    an = analyse_frame(img)
    identity = list(range(len(an.quant)))
    _, perm, _, _ = ri.encode_ri(an, 1, ctr=3)
    assert ri.nzca_corr(an, identity, img) > 0.35          # small test image: 0.45; real frames 0.7-0.85
    assert abs(ri.nzca_corr(an, perm, img)) < 0.3


def test_decrypt_ignoring_tag_matches_the_receiver_and_wrong_key_scrambles():
    img = make_image(64, 48, "scene")
    an = analyse_frame(img)
    for m, seg in (("E2", False), ("E5", False), ("E2", True), ("E5", True)):   # contract v2 and v3 (E5 hw form)
        c = package_frame(an, KEY, 7, 9, m, segmented=seg).jpeg
        assert sm.decrypt_ignoring_tag(c, KEY) == receive_frame(c, KEY).plain_jpeg
        wrong = dec(sm.decrypt_ignoring_tag(c, sm.flip_bit(KEY, 3)))
        assert np.mean(wrong != dec(package_frame(an).jpeg)) > 0.5


def test_npcr_uaci_and_entropy():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[0, 0, 0] = 255
    npcr, uaci = sm.npcr_uaci(a, b)
    assert abs(npcr - 100 / 48) < 1e-9 and abs(uaci - 100 / 48) < 1e-9
    flat = np.tile(np.arange(256, dtype=np.uint8), (16, 1))
    assert abs(sm.entropy(np.stack([flat] * 3, axis=2)) - 8.0) < 1e-9


def test_image_folder_source(tmp_path):
    cv2.imwrite(str(tmp_path / "b.png"), np.full((300, 200, 3), 255, np.uint8))
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((512, 768, 3), np.uint8))
    src = open_source(str(tmp_path), 320, 240, loop=False)
    first, second = src.read(), src.read()
    assert first.shape == (240, 320, 3) and first.max() == 0 and second.min() == 255
    assert src.read() is None


def test_restart_overhead_reference_has_no_markers_and_no_secure_segment():
    """Bug log #16: the overhead was measured against the deployed E4 file, which carries the APP11
    segment that the restart-interval variants lack, so large N showed a negative overhead."""
    an = analyse_frame(make_image(160, 112, "scene"))
    ref = ri.encode_ri(an, len(an.quant), ctr=2, shuffle=False)[0]
    assert b"\xff\xdd" not in ref.split(b"\xff\xda")[0], "reference must not declare a restart interval"
    assert b"\xff\xeb" not in ref, "reference must not carry the APP11 secure segment"
    for n in (1, 4, 32):
        assert len(ri.encode_ri(an, n, ctr=2)[0]) > len(ref), f"N={n}: restart markers must cost bytes"


def test_raster_solver_reassembles_large_plain_segments():
    """Sanity check of the any-N solver: on UNENCRYPTED segments of a textured frame it must chain
    clearly more true successors than chance, or its E5 numbers would say nothing."""
    img = make_image(320, 240, "scene")
    plain = dec(package_frame(analyse_frame(img)).jpeg)
    tiles, rows, cols = ri.tiles_of(plain)
    n = 32
    groups = ri.groups_of(rows * cols, n)
    order = ri._shuffle_order(KEY, 7, 3, len(groups))
    shown = np.concatenate([np.arange(groups[g].start, groups[g].stop) for g in order])
    seq = ri.raster_solve(tiles[shown], ri.segments_of(order, n, rows * cols), cols)
    succ = sum(order[b] == order[a] + 1 for a, b in zip(seq, seq[1:])) / (len(seq) - 1)
    assert sorted(seq) == list(range(len(groups)))
    assert succ > 2 / (len(groups) - 1), f"successors {succ:.2f}, chance {1 / (len(groups) - 1):.2f}"


def test_e5_size_match_scores_and_reference_points():
    """experiments/e5_size_match.py: the true order places every segment and leaks like E4 (A3 > 0.3 on the
    scene), random placement is near chance, and the attacks give a score for every (start, gap)."""
    import argparse
    import e5_size_match as sm
    a = argparse.Namespace(video="synthetic", size="96x64", quality=75, gaps=[1, 3], stride=2, starts=1,
                           start_step=10, seed=0, name="t", out="unused")
    res = sm.run(a)
    rows = res["rows"]
    assert len(rows) == 2, f"rows: got {len(rows)}, expected 2"
    for r in rows:
        assert r["oracle"]["exact"] == 1.0, f"gap {r['gap']}: oracle placed {r['oracle']['exact']:.2f}, expected 1.0"
        assert r["oracle"]["leak"] > 0.3, f"gap {r['gap']}: oracle leak {r['oracle']['leak']:.2f}, expected > 0.3"
        for k in sm.ATTACKERS:
            assert -1 <= r[k]["leak"] <= 1 and 0 <= r[k]["exact"] <= 1, f"gap {r['gap']} {k}: {r[k]}"
    assert "gain" in res["summary"] and "best_attack.motion" in res["summary"]


def test_review_numbers_macros(tmp_path, monkeypatch):
    """experiments/review_numbers.py turns the review run's JSON into the paper's macros, without the
    synthetic scene in the size-matching numbers."""
    import json
    import review_numbers as rn
    schemes = {n: dict(max_leak=v) for n, v in [("E5 (v2)", 0.01), ("E5 (v3, hw)", 0.03), ("VanDroogenbroeck", 0.99),
                                                 ("Kishore", 0.73), ("Auer", 0.48), ("Hirose RI=1", 0.75),
                                                 ("Hirose RI=4", 0.99)]}
    (tmp_path / "benchmark_suite_xiph").mkdir()
    (tmp_path / "benchmark_suite_xiph" / "foreman_cif.json").write_text(json.dumps(dict(meta={}, schemes=schemes)))
    row = dict(best_attack="assign_nz", gain=0.01, e2_old=dict(leak=0.7), oracle=dict(motion=0.4),
               rank_bytes=dict(leak=0.6, motion=0.0, exact=0.3), assign_nz=dict(leak=0.71, motion=0.02, exact=0.6))
    (tmp_path / "e5_size_match").mkdir()
    for name, source in (("clip", "clip.mp4"), ("synthetic", "synthetic")):
        (tmp_path / "e5_size_match" / f"{name}.json").write_text(json.dumps(dict(source=source, rows=[row, row], summary={})))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["review_numbers.py", str(tmp_path), "--out", str(out)])
    rn.main()
    tex = (out / "review_numbers.tex").read_text()
    for want in (r"\newcommand{\XiphEfiveMax}{0.03}", r"\newcommand{\XiphPubMin}{0.48}", r"\newcommand{\XiphPubMax}{0.99}",
                 r"\newcommand{\SmSources}{1}", r"\newcommand{\SmExact}{60}", r"\newcommand{\SmOld}{0.70}"):
        assert want in tex, f"missing {want!r} in:\n{tex}"
