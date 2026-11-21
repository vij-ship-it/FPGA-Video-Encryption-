"""Bug #52 (ICCE 2027 review, Reviewer 1): the tag covers the 26-byte prefix and the scan, not the
DQT segment. A receiver that copies the stream's DQT shows an attacker-chosen picture although the tag
verifies. With the session's tables pinned (receive_frame(..., qtables=...)), a changed DQT is rejected
with reason 'dqt-mismatch', for every mode, in the model and in the fast C receiver, and it does not
consume the frame counter in the replay guard."""
import numpy as np
import pytest

from link import fastrx
from model import jpeg as mj
from model.dct_quant import tables

KEY = bytes(range(16))
SID = 0x1234_5678_9ABC_0001
QUALITY = 75


def _frames():
    rng = np.random.default_rng(52)
    rgb = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    rgb[:, 20:40] //= 3                                   # some structure, not only noise
    return {mode: mj.encode_frame(rgb, KEY, SID, 7 + i, mode, QUALITY).jpeg
            for i, mode in enumerate(["E0", "E1", "E2", "E3", "E4", "E5"])}


def _double_first_divisor(jpeg):
    b = bytearray(jpeg)
    i = jpeg.find(b"\xff\xdb")
    assert i > 0, "no DQT segment"
    b[i + 5] = min(255, 2 * b[i + 5])
    return bytes(b)


RECEIVERS = [("model", mj.receive_frame)]
if fastrx.available():
    RECEIVERS.append(("fastrx", fastrx.receive_frame))


@pytest.mark.parametrize("name,rx", RECEIVERS)
def test_changed_dqt_passes_the_tag_but_fails_the_pin(name, rx):
    pinned = tables(QUALITY)
    for mode, jpeg in _frames().items():
        clean = rx(jpeg, KEY, qtables=pinned)
        assert clean.ok, f"{name} {mode}: clean frame rejected ({clean.reason})"
        bad = _double_first_divisor(jpeg)
        unpinned = rx(bad, KEY)
        assert unpinned.ok, f"{name} {mode}: expected the old behaviour (DQT not authenticated), got {unpinned.reason}"
        got = rx(bad, KEY, qtables=pinned)
        assert (got.ok, got.reason) == (False, "dqt-mismatch"), \
            f"{name} {mode}: changed DQT gave ok={got.ok} reason={got.reason}, expected dqt-mismatch"


def test_wrong_quality_pin_rejects_every_frame():
    for mode, jpeg in _frames().items():
        got = mj.receive_frame(jpeg, KEY, qtables=tables(90))
        assert got.reason == "dqt-mismatch", f"{mode}: quality 90 pin on a quality 75 frame gave {got.reason}"


def test_dqt_rejection_does_not_consume_the_counter():
    jpeg = _frames()["E2"]
    guard = mj.ReplayGuard()
    first = mj.receive_frame(_double_first_divisor(jpeg), KEY, guard, qtables=tables(QUALITY))
    assert first.reason == "dqt-mismatch", f"tampered frame: {first.reason}"
    real = mj.receive_frame(jpeg, KEY, guard, qtables=tables(QUALITY))
    assert real.ok, f"genuine frame after a rejected copy: {real.reason}, expected ok"


def test_receiver_pins_by_default():
    from link import config
    assert config.DEFAULTS["receiver"]["check_dqt"] is True
