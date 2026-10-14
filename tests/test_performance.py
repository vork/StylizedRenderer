import pytest
from inkforge.performance import FrameStats


def test_frame_stats_use_elapsed_time_and_include_stalls():
    stats=FrameStats()
    assert stats.summary() is None
    for _ in range(50):stats.add(.02,.005,.01)
    stats.add(.5,.006,.49)
    report=stats.summary()
    assert report['fps']==pytest.approx(51/1.5)
    assert report['frame_ms']==pytest.approx(1500/51)
    assert report['encode_ms']==pytest.approx(256/51)
    # After the long frame ages out, the full rolling window reports 50 FPS.
    for _ in range(110):stats.add(.02,.005,.01)
    assert stats.summary()['fps']==pytest.approx(50)
    assert stats.summary()['p95_ms']==pytest.approx(20)
