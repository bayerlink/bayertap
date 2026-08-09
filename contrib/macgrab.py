#!/usr/bin/env python3
"""Grab capture-stick frames on macOS into bayertap's --from-file form.

macOS has no V4L2, but a UVC capture stick shows up in AVFoundation, and
ffmpeg can read it. This helper wraps the incantation and writes the .npy
that ``bayertap check --from-file`` consumes:

    tunnel mode:  (height, width) uint8   -- the luma plane, for MS2109-class
                                             sticks feeding a --luma-tunnel
                                             source
    direct mode:  (height, width, 3) uint8 -- RGB, for sticks whose pipeline
                                             is transparent

Usage, MS2109 + luma tunnel (the common cheap-stick bench):

    python3 macgrab.py --list                 # find the stick's device index
    python3 macgrab.py --device 0 --mode tunnel --out grab.npy
    bayertap check --via tunnel --from-file grab.npy --expect counting

Needs ffmpeg (brew install ffmpeg) and numpy. Nothing else.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# AVFoundation names pixel formats its own way; these are the ones cheap
# UVC sticks actually offer, tried in order until ffmpeg accepts one.
CANDIDATE_FORMATS = ("uyvy422", "yuyv422", "nv12")


def list_devices() -> int:
    # ffmpeg prints the device table to stderr and exits nonzero; that is
    # its documented behaviour for -list_devices, not a failure.
    subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation",
                    "-list_devices", "true", "-i", ""])
    return 0


def grab(device: str, size: tuple[int, int], fps: int, mode: str,
         skip: int) -> np.ndarray:
    width, height = size
    out_fmt = "gray" if mode == "tunnel" else "rgb24"
    last_error = ""
    for pixel_format in CANDIDATE_FORMATS:
        with tempfile.NamedTemporaryFile(suffix=".raw") as raw:
            # -frames skips warm-up frames: sticks need a few frames to
            # lock; the LAST grabbed frame is the one kept.
            command = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "avfoundation",
                "-framerate", str(fps),
                "-video_size", f"{width}x{height}",
                "-pixel_format", pixel_format,
                "-i", device,
                "-frames:v", str(skip + 1),
                "-f", "rawvideo", "-pix_fmt", out_fmt,
                "-y", raw.name,
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                last_error = result.stderr.strip().splitlines()[-1] if \
                    result.stderr.strip() else f"ffmpeg exit {result.returncode}"
                continue
            data = np.fromfile(raw.name, dtype=np.uint8)
            channels = 1 if mode == "tunnel" else 3
            frame_size = width * height * channels
            if data.size < frame_size:
                last_error = (f"{pixel_format}: ffmpeg wrote {data.size} bytes,"
                              f" one frame needs {frame_size}")
                continue
            frame = data[-frame_size:]          # the last (settled) frame
            if mode == "tunnel":
                return frame.reshape(height, width)
            return frame.reshape(height, width, 3)
    raise SystemExit(
        f"no candidate pixel format worked; last error: {last_error}\n"
        "Run with --list to check the device index, and confirm the stick "
        "delivers the requested --size.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true",
                        help="list AVFoundation devices and exit")
    parser.add_argument("--device", default="0",
                        help="AVFoundation video device index (see --list)")
    parser.add_argument("--size", type=lambda s: tuple(
                            int(v) for v in s.split("x")),
                        default=(1920, 1080), help="capture WxH")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mode", choices=["tunnel", "direct"],
                        default="tunnel")
    parser.add_argument("--skip", type=int, default=5,
                        help="warm-up frames to discard before keeping one")
    parser.add_argument("--out", default="grab.npy")
    args = parser.parse_args(argv)

    if args.list:
        return list_devices()

    frame = grab(args.device, args.size, args.fps, args.mode, args.skip)
    np.save(args.out, frame)
    print(f"wrote {args.out}: shape {frame.shape}, "
          f"min {frame.min()}, max {frame.max()}")
    if frame.max() == frame.min():
        print("WARNING: the frame is flat -- no signal? Check the cable, "
              "the device index, and that the source is streaming.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
