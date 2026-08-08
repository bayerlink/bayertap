"""Everything that does not need capture hardware, which is everything but
the ioctls: struct sizes pinned to the uapi, frame_array per format, and the
whole check pipeline end to end from files -- the same code paths a device
feeds, minus the device.
"""
import ctypes

import numpy as np
import pytest
from bayerlink import encode_frame, pattern, tunnel
from bayerlink.protocol import FLAG_TEST_PATTERN

from bayertap import cli, v4l2


def test_struct_sizes_match_the_kernel_uapi():
    expected = {v4l2.Capability: 104, v4l2.PixFormat: 48, v4l2.Format: 208,
                v4l2.RequestBuffers: 20, v4l2.Buffer: 88, v4l2.FmtDesc: 64}
    for struct, size in expected.items():
        assert ctypes.sizeof(struct) == size, struct.__name__


def test_frame_array_extracts_luma_from_yuyv_and_passes_rgb_untouched():
    height, width, stride = 4, 8, 8 * 2 + 4          # padded line
    yuyv = np.arange(height * stride, dtype=np.uint8)
    luma = v4l2.frame_array(yuyv, v4l2.FORMAT_YUYV, width, height, stride)
    assert luma.shape == (height, width)
    assert np.array_equal(luma[0], yuyv[:width * 2:2])

    stride = 8 * 3
    rgb = np.arange(height * stride, dtype=np.uint8)
    frame = v4l2.frame_array(rgb, v4l2.FORMAT_RGB3, width, height, stride)
    assert frame.shape == (height, width, 3)
    assert np.array_equal(frame.reshape(-1), rgb)

    with pytest.raises(ValueError, match="refuses MJPEG"):
        v4l2.frame_array(rgb, v4l2.FORMAT_MJPG, width, height, stride)


def _direct_file(tmp_path, lane_perm=(0, 1, 2), seq=4):
    raw = pattern.generate("corners", 32, 8)
    container = encode_frame(raw, "RGGB", frame_seq=seq, display=(32, 12),
                             flags=FLAG_TEST_PATTERN)
    scrambled = np.empty_like(container)
    for k in range(3):
        scrambled[:, :, lane_perm[k]] = container[:, :, k]
    path = tmp_path / "direct.npy"
    np.save(path, scrambled)
    return path


def test_check_from_file_direct_passes_and_fails_honestly(tmp_path, capsys):
    path = _direct_file(tmp_path)
    code = cli.main(["--from-file", str(path), "check", "--pattern", "corners"])
    assert code == 0 and "1 good, 0 bad" in capsys.readouterr().out

    code = cli.main(["--from-file", str(path), "check", "--pattern", "checker"])
    out = capsys.readouterr().out
    assert code == 1 and "differ from checker" in out


def test_check_applies_the_lane_map(tmp_path, capsys):
    path = _direct_file(tmp_path, lane_perm=(2, 0, 1))
    # Without the map: not a bayerlink frame at all.
    assert cli.main(["--from-file", str(path), "check"]) == 1
    capsys.readouterr()
    # detect_lane_map's answer, applied: (container byte k is on channel...)
    assert cli.main(["--from-file", str(path), "--lane-map", "2,0,1",
                     "check", "--pattern", "corners"]) == 0


def test_check_from_file_via_tunnel(tmp_path, capsys):
    display = (192, 40)
    inner_w, inner_h = tunnel.inner_display(*display)
    raw = pattern.generate("counting", inner_w * 2, 20)
    container = encode_frame(raw, "BGGR", frame_seq=0,
                             display=(inner_w, inner_h),
                             flags=FLAG_TEST_PATTERN)
    grey = tunnel.encode(container, display)
    path = tmp_path / "tunnel.npy"
    np.save(path, grey)
    code = cli.main(["--from-file", str(path), "--via", "tunnel",
                     "check", "--pattern", "counting"])
    assert code == 0 and "1 good, 0 bad" in capsys.readouterr().out
