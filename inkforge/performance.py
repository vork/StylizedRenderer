"""Viewer wall-clock statistics; never force a GPU synchronization."""
from collections import deque
import math


class FrameStats:
    def __init__(self,window_seconds=2.):
        self.window_seconds=window_seconds
        self.samples=deque();self.elapsed=0.

    def add(self,frame_seconds,encode_seconds,wait_seconds):
        if not math.isfinite(frame_seconds) or frame_seconds<=0:return
        self.samples.append((frame_seconds,encode_seconds,wait_seconds))
        self.elapsed+=frame_seconds
        while len(self.samples)>1 and self.elapsed-self.samples[0][0]>=self.window_seconds:
            self.elapsed-=self.samples.popleft()[0]

    def summary(self):
        if not self.samples:return None
        frames=sorted(s[0] for s in self.samples);count=len(frames)
        return dict(fps=count/self.elapsed,frame_ms=self.elapsed/count*1000,
                    p95_ms=frames[math.ceil(count*.95)-1]*1000,
                    encode_ms=sum(s[1] for s in self.samples)/count*1000,
                    wait_ms=sum(s[2] for s in self.samples)/count*1000)
