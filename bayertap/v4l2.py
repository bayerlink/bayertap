"""V4L2 capture via raw ioctls: open, negotiate, mmap, stream.

The same zero-dependency posture as picam2hdmi's KMS layer, for the same
reason: an appliance should not carry a bindings package for the nine
ioctls it uses, and the V4L2 uapi is a stable kernel ABI. Every struct is
pinned against its known uapi size by the tests -- a mislaid ctypes field
fails as one assert here instead of as corruption on a bench.

This layer knows nothing about bayerlink. It negotiates a pixel format,
maps buffers, and yields frames; what the frames MEAN belongs to the
caller. That split is what lets one tool serve every capture tier -- a
TC358743 bridge delivering RGB888, an S-variant USB3 stick delivering
RGB24, an MS2109 delivering YUY2 for the luma tunnel -- they are all just
V4L2 devices with different formats.
"""
from __future__ import annotations

import ctypes
import fcntl
import mmap as _mmap
import os
import select

import numpy as np

u8, u32, u64 = ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint64


class Capability(ctypes.Structure):
    _fields_ = [("driver", ctypes.c_char * 16), ("card", ctypes.c_char * 32),
                ("bus_info", ctypes.c_char * 32), ("version", u32),
                ("capabilities", u32), ("device_caps", u32),
                ("reserved", u32 * 3)]                       # 104 bytes


class PixFormat(ctypes.Structure):
    _fields_ = [("width", u32), ("height", u32), ("pixelformat", u32),
                ("field", u32), ("bytesperline", u32), ("sizeimage", u32),
                ("colorspace", u32), ("priv", u32), ("flags", u32),
                ("ycbcr_enc", u32), ("quantization", u32),
                ("xfer_func", u32)]                          # 48 bytes


class Format(ctypes.Structure):
    _fields_ = [("type", u32), ("pad", u32),
                ("fmt", ctypes.c_uint8 * 200)]               # 208 bytes

    @property
    def pix(self) -> PixFormat:
        return PixFormat.from_buffer(self, 8)


class RequestBuffers(ctypes.Structure):
    _fields_ = [("count", u32), ("type", u32), ("memory", u32),
                ("capabilities", u32), ("flags", u8),
                ("reserved", u8 * 3)]                        # 20 bytes


class Timeval(ctypes.Structure):
    _fields_ = [("sec", ctypes.c_long), ("usec", ctypes.c_long)]


class Timecode(ctypes.Structure):
    _fields_ = [("type", u32), ("flags", u32), ("frames", u8), ("seconds", u8),
                ("minutes", u8), ("hours", u8), ("userbits", u8 * 4)]


class Buffer(ctypes.Structure):
    _fields_ = [("index", u32), ("type", u32), ("bytesused", u32),
                ("flags", u32), ("field", u32), ("pad", u32),
                ("timestamp", Timeval), ("timecode", Timecode),
                ("sequence", u32), ("memory", u32),
                ("m_offset", u64),        # union: offset/userptr/planes/fd
                ("length", u32), ("reserved2", u32),
                ("request_fd", u32)]                          # 88 bytes (LP64)


class FmtDesc(ctypes.Structure):
    _fields_ = [("index", u32), ("type", u32), ("flags", u32),
                ("description", ctypes.c_char * 32), ("pixelformat", u32),
                ("mbus_code", u32), ("reserved", u32 * 3)]   # 64 bytes


def _ioc(direction: int, nr: int, size: int) -> int:
    return (direction << 30) | (size << 16) | (ord("V") << 8) | nr


IOCTL_QUERYCAP = _ioc(2, 0, ctypes.sizeof(Capability))
IOCTL_ENUM_FMT = _ioc(3, 2, ctypes.sizeof(FmtDesc))
IOCTL_S_FMT = _ioc(3, 5, ctypes.sizeof(Format))
IOCTL_REQBUFS = _ioc(3, 8, ctypes.sizeof(RequestBuffers))
IOCTL_QUERYBUF = _ioc(3, 9, ctypes.sizeof(Buffer))
IOCTL_QBUF = _ioc(3, 15, ctypes.sizeof(Buffer))
IOCTL_DQBUF = _ioc(3, 17, ctypes.sizeof(Buffer))
IOCTL_STREAMON = _ioc(1, 18, 4)
IOCTL_STREAMOFF = _ioc(1, 19, 4)

CAPTURE = 1                       # V4L2_BUF_TYPE_VIDEO_CAPTURE
MEMORY_MMAP = 1


def fourcc(text: str) -> int:
    return int.from_bytes(text.encode("ascii"), "little")


FORMAT_YUYV = fourcc("YUYV")
FORMAT_RGB3 = fourcc("RGB3")      # RGB24: memory R,G,B per pixel
FORMAT_BGR3 = fourcc("BGR3")      # memory B,G,R per pixel
FORMAT_MJPG = fourcc("MJPG")      # rejected for payload: lossy


def frame_array(data: np.ndarray, pixelformat: int, width: int,
                height: int, bytesperline: int) -> np.ndarray:
    """Raw captured bytes -> a channel-shaped array, per format.

    YUYV yields the LUMA plane (height, width) -- the only channel the luma
    tunnel needs, extracted here so callers never index interleaved bytes.
    RGB3/BGR3 yield (height, width, 3) in MEMORY byte order, untouched:
    which byte is which lane is detect_lane_map()'s question, not this
    layer's.
    """
    rows = data[:height * bytesperline].reshape(height, bytesperline)
    if pixelformat == FORMAT_YUYV:
        return rows[:, :width * 2][:, 0::2].copy()
    if pixelformat in (FORMAT_RGB3, FORMAT_BGR3):
        return rows[:, :width * 3].reshape(height, width, 3).copy()
    raise ValueError(
        f"unhandled pixelformat {pixelformat:#010x}; this tool captures "
        "YUYV (luma tunnel) and RGB3/BGR3 (direct), and refuses MJPEG "
        "outright -- lossy compression cannot carry a byte container.")


class Device:
    """One V4L2 capture device, negotiated and streaming."""

    def __init__(self, path: str = "/dev/video0"):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        cap = Capability()
        fcntl.ioctl(self.fd, IOCTL_QUERYCAP, cap)
        self.driver = cap.driver.decode(errors="replace")
        self.card = cap.card.decode(errors="replace")
        self._maps: list = []

    def formats(self) -> list[str]:
        """The fourccs the device offers, for diagnostics and refusals."""
        found = []
        index = 0
        while True:
            desc = FmtDesc(index=index, type=CAPTURE)
            try:
                fcntl.ioctl(self.fd, IOCTL_ENUM_FMT, desc)
            except OSError:
                return found
            found.append(int(desc.pixelformat).to_bytes(4, "little")
                         .decode(errors="replace"))
            index += 1

    def set_format(self, width: int, height: int, pixelformat: int):
        """Negotiate; RETURNS what the device actually granted.

        V4L2 is a negotiation, not a command: the driver may adjust every
        field. The caller compares granted against requested and refuses a
        substitute -- silently capturing at the wrong geometry is how a
        bench lies.
        """
        request = Format(type=CAPTURE)
        pix = request.pix
        pix.width, pix.height = width, height
        pix.pixelformat = pixelformat
        pix.field = 1                                # V4L2_FIELD_NONE
        fcntl.ioctl(self.fd, IOCTL_S_FMT, request)
        granted = request.pix
        if (granted.width, granted.height) != (width, height) or \
                granted.pixelformat != pixelformat:
            got = int(granted.pixelformat).to_bytes(4, "little").decode(
                errors="replace")
            want = int(pixelformat).to_bytes(4, "little").decode(errors="replace")
            raise RuntimeError(
                f"{self.path} granted {granted.width}x{granted.height} {got} "
                f"instead of {width}x{height} {want}; it offers formats "
                f"{self.formats()}. Refusing the substitute.")
        return granted.width, granted.height, granted.bytesperline

    def frames(self, width: int, height: int, pixelformat: int, buffers: int = 4):
        """Yield frames forever: (sequence, array) per :func:`frame_array`."""
        width, height, bytesperline = self.set_format(width, height, pixelformat)

        request = RequestBuffers(count=buffers, type=CAPTURE, memory=MEMORY_MMAP)
        fcntl.ioctl(self.fd, IOCTL_REQBUFS, request)
        for index in range(request.count):
            buffer = Buffer(index=index, type=CAPTURE, memory=MEMORY_MMAP)
            fcntl.ioctl(self.fd, IOCTL_QUERYBUF, buffer)
            self._maps.append(_mmap.mmap(self.fd, buffer.length,
                                         offset=buffer.m_offset))
            fcntl.ioctl(self.fd, IOCTL_QBUF, buffer)

        fcntl.ioctl(self.fd, IOCTL_STREAMON, ctypes.c_int(CAPTURE))
        try:
            while True:
                select.select([self.fd], [], [])
                buffer = Buffer(type=CAPTURE, memory=MEMORY_MMAP)
                fcntl.ioctl(self.fd, IOCTL_DQBUF, buffer)
                data = np.frombuffer(self._maps[buffer.index], np.uint8,
                                     count=buffer.bytesused)
                yield buffer.sequence, frame_array(
                    data, pixelformat, width, height, bytesperline)
                fcntl.ioctl(self.fd, IOCTL_QBUF, buffer)
        finally:
            fcntl.ioctl(self.fd, IOCTL_STREAMOFF, ctypes.c_int(CAPTURE))
