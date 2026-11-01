"""Focused CPU checks for proposed read-only coverage hooks."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import numpy as np
import torch

from opensprite_tracker.supermimic import coverage as module
from opensprite_tracker.supermimic.buffer import ReplayBuffer

def test_observed_coverage_preserves_sampling_and_boundaries():
    torch.set_num_threads(2)
    motion = SimpleNamespace(clip_start=torch.tensor([0, 10, 20]),
                             clip_end=torch.tensor([9, 19, 29]),
                             clip_names=['dance', 'getup', 'walk'])
    with tempfile.TemporaryDirectory() as tmp:
        audit = module.CoverageAudit(motion, tmp, resumed_iteration=30000,
                                     session='cpu-test', wm_window=3, phase_frames=5,
                                     validation_envs=1)
        audit.set_iteration(30001)
        buffer = ReplayBuffer(8, 3, 'cpu', state_dim=1, action_dim=1, validation_envs=1,
                              extra_fields={'term': ((), torch.bool)})
        for t in range(6):
            out = {k: torch.zeros(v.shape[1:], dtype=v.dtype) for k,v in buffer.data.items()}
            out['f'] = torch.tensor([t, 10+t, 20+t])
            out['ep_step'] = torch.tensor([t, t, t], dtype=torch.int32)
            out['done'] = torch.tensor([t==2, False, False])
            out['term'] = out['done'].clone()
            audit.real(out)
            buffer.add(out)
        torch.manual_seed(42)
        initial_rng = torch.get_rng_state().clone()
        window = buffer.sample_windows(128, 3)
        post_sample_rng = torch.get_rng_state().clone()
        preserved = {k:v.clone() for k,v in window.items()}
        audit.wm(window)
        assert torch.equal(torch.get_rng_state(), post_sample_rng)
        assert all(torch.equal(v, preserved[k]) for k,v in window.items())
        assert all((window['mask'][:,0] == 1))
        # Explicit episode-boundary mask: terminal retained at depth 1,
        # invalid successor may be an unrelated source and must not count.
        audit.wm({'f': torch.tensor([[0, 1, 20], [10, 11, 12]]),
                  'ep_step': torch.tensor([[0, 1, 0], [499, 500, 501]]),
                  'mask': torch.tensor([[1., 1., 0.], [1., 1., 1.]])})
        torch.set_rng_state(initial_rng)
        replayed = buffer.sample_windows(128, 3)
        assert all(torch.equal(v, replayed[k]) for k,v in preserved.items())
        torch.manual_seed(123)
        start, history = buffer.sample_starts(32, 10)
        before_actor = torch.get_rng_state().clone()
        preserved_start = {k:v.clone() for k,v in start.items()}
        preserved_history = {k:v.clone() for k,v in history.items()}
        audit.actor(start)
        assert torch.equal(torch.get_rng_state(), before_actor)
        assert all(torch.equal(v, preserved_start[k]) for k,v in start.items())
        assert all(torch.equal(v, preserved_history[k]) for k,v in history.items())
        assert audit.counts['real_train'].sum() == 12
        assert audit.counts['real_holdout'].sum() == 6
        assert audit.counts['real_term_train'].sum() == 1
        assert audit.counts['real_done_train'].sum() == 1
        assert audit.counts['actor_start'].sum() == 32
        assert audit.counts['wm_train_valid'].sum() == int(window['mask'].sum())+5
        assert audit.counts['wm_train_valid'][2].sum() == int(window['mask'][:,2].sum())+1
        # No source bin alias at frame9->frame10 or frame19->frame20.
        ix = audit._indices(np.array([9,10,19,20]), np.zeros(4))
        assert ix.tolist() == [6,12,18,24]
        assert audit._indices(np.array([0]*7), np.array([0,1,9,10,49,250,500])).tolist() == [0,1,1,2,2,4,5]
        path = audit.flush('cpu-test')
        report = json.loads(path.read_text())
        assert report['first_observed_iteration'] == report['last_observed_iteration'] == 30001
        assert not list(Path(tmp).glob('*.tmp'))
        assert all(v.sum()==0 for v in audit.counts.values())
        # A resumed process starts a distinct interval, no cumulative replay.
        resumed = module.CoverageAudit(motion, tmp, resumed_iteration=30001,
                                       session='resumed-test', wm_window=3, phase_frames=5,
                                       validation_envs=1)
        resumed.set_iteration(30002)
        resumed.actor(start)
        second = json.loads(resumed.flush('resume').read_text())
        assert second['resumed_from_iteration'] == 30001
        assert second['first_observed_iteration'] == second['last_observed_iteration'] == 30002
        assert second['calls'] == {'real':0, 'wm_train':0, 'actor':1}
        assert np.asarray(second['counts']['actor_start']).sum() == 32
    print(json.dumps({'result':'PASS','checks':['RNG unchanged','sample outputs unchanged',
        'terminal included and post-reset excluded','clip boundaries distinct','age bins',
        'exact count conservation','atomic disjoint interval flush','resume does not replay old counts']}))

