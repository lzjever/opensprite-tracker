"""Join passive MuJoCo render workers before GLFW's process-exit cleanup."""
from contextlib import contextmanager
import threading


@contextmanager
def passive_viewer(model, data, **kwargs):
    import mujoco.viewer
    # Used by dedicated GUI processes. Capture workers created by the public
    # launch API without relying on private MuJoCo simulator handles.
    before=set(threading.enumerate())
    viewer=mujoco.viewer.launch_passive(model,data,**kwargs)
    workers=set(threading.enumerate())-before
    try:
        yield viewer
    finally:
        viewer.close()
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=5.)
                if worker.is_alive():
                    raise RuntimeError('MuJoCo render worker did not close')
