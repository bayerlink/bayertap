"""bayertap -- a passive conformance tap for bayerlink links.

In a revela/np2hw netlist, a tap is a sink that observes a stream without
stalling it. This is that concept as bench equipment: point it at any V4L2
capture device carrying a bayerlink source and it tells you whether the
bytes survive. It judges with the same published codec the source encoded
with, so the two ends cannot disagree about what conformance means.
"""
__version__ = "0.1.0"
