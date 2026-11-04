"""Passive GUI teardown completes before process-exit GLFW cleanup."""
import threading
import time

import mujoco.viewer

from opensprite_sdk.visualization import passive_viewer


def test_async_renderer_has_exited_when_viewer_context_returns(monkeypatch):
    stop=threading.Event();exited=threading.Event();workers=[]

    class Handle:
        def close(self):
            stop.set()

    def launch(*args,**kwargs):
        def render():
            stop.wait(1)
            time.sleep(.02)
            exited.set()
        worker=threading.Thread(target=render,name='simulated-mujoco-renderer')
        workers.append(worker);worker.start()
        return Handle()

    monkeypatch.setattr(mujoco.viewer,'launch_passive',launch)
    with passive_viewer(None,None):
        assert not exited.is_set()
    assert exited.is_set() and not workers[0].is_alive()
