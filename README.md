# bayertap

**A passive conformance tap for bayerlink links.**

In a streaming netlist, a *tap* is a sink that observes a stream without
stalling it. This is that concept as bench equipment: point it at any V4L2
capture device carrying a [bayerlink](https://github.com/bayerlink/bayerlink)
source and it tells you whether the bytes survive — judged with the same
published codec the source encoded with, so the two ends cannot disagree
about what conformance means.

```bash
bayertap probe                          # which byte-lane permutation is this rig?
bayertap --lane-map 2,1,0 check --pattern counting     # every sample, judged
bayertap --via tunnel check --pattern counting          # through a cheap dongle
bayertap --from-file frame.npy check    # no hardware at all
```

## Capture tiers

| Tier | Hardware | Path | Fidelity |
| --- | --- | --- | --- |
| **Trusted** | TC358743 HDMI→CSI-2 bridge (~$25, the PiKVM part) on any Pi or Jetson | `--via direct`, RGB888 | byte-exact |
| **Cheap** | MS2130S / MS2131S USB3 stick (~$15) on any Linux box | `--via direct`, RGB24 | byte-exact **if** the stick's pipeline is transparent — `check` answers that in minutes |
| **Drawer** | MS2109-class USB2 stick | `--via tunnel`, YUYV | **bit-exact at 1/6 capacity**, via the luma tunnel — run the source with `--luma-tunnel` |
| Rejected | any USB2 stick, direct | MJPEG / starved YUYV | physics: USB2 cannot carry uncompressed 1080p, and MJPEG is lossy. Refused in code, not just here |

The drawer tier deserves its sentence: a $10 dongle you already own becomes a
working conformance receiver **today**, because bayerlink's luma tunnel
carries the unchanged container as grey levels with a pilot line the decoder
learns the channel from. Slow, and completely sufficient for proving bytes.

## The two bring-up commands

**`probe`** resolves the byte-lane permutation — which captured channel holds
container byte k — by trying all six against the header's magic and CRC. Its
output is directly the `lane_map` for this tool's `check` *and* for the
FPGA-side receiver (`np2hw.bayerlink_in`). One command instead of a scope.

**`check --pattern X`** regenerates the pattern locally (patterns are pure
functions — nothing travels out of band) and compares **every sample**,
watches `frame_seq` for gaps, tolerates scanout repeats (protocol-legal),
and exits nonzero on any discrepancy. Point CI at it if you like.

For the TC358743 tier, remember the bridge needs an EDID loaded
(`v4l2-ctl --set-edid`) before it captures — and that EDID is *leverage*: it
is what the source reads, so advertise only RGB 4:4:4 at your one mode and
the source is steered into the only format the protocol accepts.

## No Linux box on the bench

The capture side is V4L2, but the stick does not care what it plugs into:
[`contrib/macgrab.py`](contrib/macgrab.py) grabs frames on macOS through
ffmpeg/AVFoundation and writes the same `.npy` that `--from-file` consumes —

```sh
python3 contrib/macgrab.py --device 0 --mode tunnel --out grab.npy
bayertap check --via tunnel --from-file grab.npy --expect counting
```

## Status

Everything except the ioctl layer is proven off-target: struct sizes pinned
against the kernel uapi, both decode paths (direct with lane maps, tunnel
with pilot learning) tested end-to-end from files. First runs against real
capture silicon are the bench session; the known-good device table starts
there.

## Funding

Developed independently; recurring support via
[github.com/sponsors/lanserge](https://github.com/sponsors/lanserge), or write
first: **s.rabykin@gmail.com**. Sponsorable capability targets carry the
`sponsorable` label on the issue tracker — currently
[DNG and bare-mosaic bridges](https://github.com/bayerlink/bayertap/issues/1)
(every DNG camera and raw dataset as a source). Scope is agreed in writing
before work starts; sponsored work lands in the open tree immediately, MIT
like everything else — sponsorship buys ordering and named credit, not
exclusivity. The person behind it:
[serge.rabyking.com](https://serge.rabyking.com).

## Licence

MIT.
