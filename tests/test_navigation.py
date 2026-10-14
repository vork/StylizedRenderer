import numpy as np
from inkforge.navigation import FlyCamera,DayClock


def test_fly_look_keeps_eye_fixed_and_movement_is_normalized():
    camera=FlyCamera([0,1,5],[0,1,0]);eye=camera.eye.copy()
    camera.look(100,-50)
    np.testing.assert_array_equal(camera.eye,eye)
    assert camera.forward[0]>0 and camera.forward[1]>0
    camera.move({'w','d','left_shift'},.25)
    np.testing.assert_allclose(np.linalg.norm(camera.eye-eye),camera.speed,rtol=1e-5)
    camera.look(0,100000)
    assert np.isfinite(camera.forward).all() and camera.pitch>-np.pi/2


def test_clock_progresses_wraps_pauses_and_eases_nudges():
    clock=DayClock(23.99,20)
    assert 0<clock.tick(1)<.02
    clock.running=False;hour=clock.hour
    assert clock.tick(4)==hour
    clock.nudge(2)
    first=clock.tick(.05)-hour
    assert 0<first<.05
    for _ in range(15):clock.tick(.05)
    assert abs(clock.hour-hour-2)<1e-6
    clock.set(13);clock.nudge(-2)
    clock.tick(1)
    assert abs(clock.hour-11)<1e-6
