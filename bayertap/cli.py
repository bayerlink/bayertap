"""bayertap: a passive conformance tap for bayerlink links.

Point it at any V4L2 capture device carrying a bayerlink source and it
tells you whether the bytes survive. Three subcommands, one per question:

  check       does every frame decode, does the pattern match, does
              frame_seq advance without gaps?
  probe       which byte-lane permutation is this platform pair using?
  save        keep decoded raw frames for offline analysis.

Two capture paths, chosen by --via:

  direct      RGB3/BGR3 capture (TC358743 bridge, S-variant USB3 stick):
              the container arrives as bytes, possibly lane-permuted --
              which probe resolves and check accepts via --lane-map.
  tunnel      YUYV capture (MS2109-class USB2 dongles): only luma survives
              that hardware, so the source must run --luma-tunnel, and the
              container is recovered from grey levels via the pilot line.

Everything also runs from a saved .npy instead of a device (--from-file),
so the whole validation path is testable -- and tested -- with no capture
hardware at all.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from bayerlink import decode_frame, detect_lane_map, pattern, tunnel
from bayerlink.protocol import Header


def _validate(container: np.ndarray, expect_pattern: str | None,
              last_seq: int | None):
    """One container -> (Header, list of problem strings)."""
    problems = []
    try:
        header, raw = decode_frame(container)
    except ValueError as error:
        return None, [f"decode: {error}"]
    if last_seq is not None:
        gap = (header.frame_seq - last_seq) & 0xFFFFFFFF
        if gap == 0:
            return header, ["duplicate (same frame_seq; scanout repeat)"]
        if gap != 1:
            problems.append(f"frame_seq gap: {last_seq} -> {header.frame_seq}")
    if expect_pattern is not None:
        if not header.is_test_pattern:
            problems.append("header does not claim a test pattern")
        expected = pattern.generate(expect_pattern, header.width, header.height)
        if not np.array_equal(raw, expected):
            wrong = int((raw != expected).sum())
            first = tuple(int(v) for v in np.argwhere(raw != expected)[0])
            problems.append(
                f"{wrong} of {raw.size} samples differ from {expect_pattern}; "
                f"first at (row, col) {first}")
    return header, problems


def _containers(args):
    """Yield (tag, container) from the device or the file, per --via."""
    if args.from_file:
        physical = np.load(args.from_file)
        if args.via == "tunnel":
            luma = physical[:, :, 0] if physical.ndim == 3 else physical
            _, inner_h = tunnel.inner_display(luma.shape[1], luma.shape[0])
            yield 0, tunnel.decode(luma, inner_height=inner_h)
        else:
            frame = physical[:, :, list(args.lane_map)]
            yield 0, frame
        return

    from . import v4l2

    device = v4l2.Device(args.device)
    print(f"{device.path}: {device.card} ({device.driver}), "
          f"offers {device.formats()}")
    width, height = args.capture
    if args.via == "tunnel":
        for sequence, luma in device.frames(width, height, v4l2.FORMAT_YUYV):
            _, inner_h = tunnel.inner_display(width, height)
            try:
                yield sequence, tunnel.decode(luma, inner_height=inner_h)
            except ValueError as error:
                yield sequence, error
    else:
        pixelformat = v4l2.FORMAT_BGR3 if args.bgr else v4l2.FORMAT_RGB3
        for sequence, frame in device.frames(width, height, pixelformat):
            yield sequence, frame[:, :, list(args.lane_map)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="bayertap",
        description="Passive conformance tap for bayerlink links.")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--from-file", default=None,
                        help=".npy physical frame instead of a device")
    parser.add_argument("--via", choices=["direct", "tunnel"], default="direct",
                        help="direct: RGB capture; tunnel: YUYV luma capture "
                             "of a --luma-tunnel source")
    parser.add_argument("--capture", type=lambda s: tuple(
                            int(v) for v in s.split("x")), default=(1280, 720),
                        help="capture geometry WxH (default 1280x720)")
    parser.add_argument("--lane-map", type=lambda s: tuple(
                            int(v) for v in s.split(",")), default=(0, 1, 2),
                        help="byte permutation for direct capture; find it "
                             "with `bayertap probe`")
    parser.add_argument("--bgr", action="store_true",
                        help="capture BGR3 instead of RGB3")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="validate frames continuously")
    check.add_argument("--pattern", default=None,
                       choices=sorted(pattern.PATTERNS),
                       help="regenerate this pattern locally and compare "
                            "every sample")
    check.add_argument("--frames", type=int, default=0,
                       help="stop after N NEW frames (0 = forever)")

    sub.add_parser("probe", help="resolve the byte-lane permutation "
                                 "(direct capture of one frame)")

    save = sub.add_parser("save", help="write decoded raw frames as .npy")
    save.add_argument("--out", required=True)
    save.add_argument("--frames", type=int, default=1)

    args = parser.parse_args(argv)

    if args.command == "probe":
        if args.via == "tunnel":
            parser.error("probe needs direct RGB capture: the tunnel is "
                         "greyscale, so lanes are indistinguishable there")
        args.lane_map = (0, 1, 2)               # probe must see raw order
        for _, frame in _containers(args):
            if isinstance(frame, Exception):
                continue
            permutation, header = detect_lane_map(frame)
            print(f"lane_map = {permutation}   "
                  f"(container byte k is on captured channel lane_map[k])")
            print(f"header: {header.fourcc} {header.width}x{header.height} "
                  f"seq {header.frame_seq}")
            return 0
        return 1

    seen = last_seq = None
    good = bad = saved = 0
    for tag, container in _containers(args):
        if isinstance(container, Exception):
            bad += 1
            print(f"[{tag}] tunnel: {container}")
            continue
        header, problems = _validate(
            container, getattr(args, "pattern", None), last_seq)
        if header is not None and problems == ["duplicate (same frame_seq; "
                                               "scanout repeat)"]:
            continue                            # repeats are protocol-legal
        if problems:
            bad += 1
            print(f"[{tag}] " + "; ".join(problems))
        else:
            good += 1
            last_seq = header.frame_seq
            if args.command == "save":
                _, raw = decode_frame(container)
                path = f"{args.out.rstrip('/')}/frame_{header.frame_seq:08d}.npy"
                np.save(path, raw)
                saved += 1
                print(f"saved {path}")
            elif good % 30 == 1:
                print(f"[{tag}] ok: {header.fourcc} {header.width}x"
                      f"{header.height} seq {header.frame_seq}"
                      + (f", pattern {args.pattern} exact"
                         if getattr(args, 'pattern', None) else ""))
        limit = getattr(args, "frames", 0)
        if limit and (good if args.command == "check" else saved) >= limit:
            break

    print(f"\n{good} good, {bad} bad")
    return 0 if bad == 0 and good > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
