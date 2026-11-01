"""Read-only integer counters over ALREADY selected real training data.

No simulator/model/RNG/normalizer calls. Call outside compiled functions.
Snapshots contain disjoint process-local intervals, never inferred old coverage.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import numpy as np


def host(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


class CoverageAudit:
    AGE_LABELS = ['0', '1-9', '10-49', '50-249', '250-499', '500+']

    def __init__(self, motion, output, *, resumed_iteration, session, wm_window=8,
                 phase_frames=50, validation_envs=24):
        self.starts = host(motion.clip_start).astype(np.int64, copy=True)
        self.ends = host(motion.clip_end).astype(np.int64, copy=True)
        self.names = list(motion.clip_names)
        self.phase_frames = int(phase_frames)
        self.validation_envs = int(validation_envs)
        self.wm_window = int(wm_window)
        if self.phase_frames < 1 or self.wm_window < 1:
            raise ValueError('positive phase width and WM window required')
        if not np.all(self.ends >= self.starts) or not np.all(self.starts[1:] > self.ends[:-1]):
            raise ValueError('clip boundaries must be distinct and ordered')
        self.phase_counts = (self.ends-self.starts)//self.phase_frames+1
        self.offsets = np.r_[0, np.cumsum(self.phase_counts)]
        self.total_phases = int(self.offsets[-1])
        self.output = Path(output)
        self.session = str(session)
        if '/' in self.session or not self.session:
            raise ValueError('session must be a filename component')
        self.resumed_iteration = int(resumed_iteration)
        self.iteration = self.resumed_iteration
        self.sequence = 0
        self.last_flush = time.monotonic()
        self.last_flush_iteration = self.iteration
        self._reset()

    def _reset(self):
        shape = (self.total_phases, len(self.AGE_LABELS))
        self.counts = {key: np.zeros(shape, dtype=np.int64) for key in
                       ['real_train', 'real_holdout', 'real_term_train', 'real_term_holdout',
                        'real_done_train', 'real_done_holdout', 'actor_start']}
        self.counts['wm_train_valid'] = np.zeros((self.wm_window,)+shape, dtype=np.int64)
        self.calls = {'real': 0, 'wm_train': 0, 'actor': 0}
        self.labels = set()
        self.first_iteration = None
        self.last_iteration = None

    def set_iteration(self, iteration, label='outer'):
        self.iteration = int(iteration)
        self.label = str(label)

    def _mark(self):
        self.labels.add(getattr(self, 'label', 'refill'))
        self.first_iteration = self.iteration if self.first_iteration is None else min(self.first_iteration, self.iteration)
        self.last_iteration = self.iteration if self.last_iteration is None else max(self.last_iteration, self.iteration)

    def _indices(self, frames, ages):
        f, a = np.broadcast_arrays(host(frames), host(ages))
        f, a = f.astype(np.int64, copy=False).ravel(), a.astype(np.int64, copy=False).ravel()
        if np.any(a < 0):
            raise ValueError('negative pre-transition episode age')
        clip = np.searchsorted(self.starts, f, side='right')-1
        if np.any(clip < 0) or np.any(f > self.ends[clip]):
            raise ValueError('valid frame outside a source clip')
        phase = self.offsets[clip]+(f-self.starts[clip])//self.phase_frames
        age = np.searchsorted(np.array([1, 10, 50, 250, 500]), a, side='right')
        return phase*len(self.AGE_LABELS)+age

    def _add(self, key, frames, ages, mask=None, depth=None):
        f, a = np.broadcast_arrays(host(frames), host(ages))
        if mask is not None:
            m = np.broadcast_to(host(mask).astype(bool), f.shape)
            f, a = f[m], a[m]
        index = self._indices(f, a)
        n = self.total_phases*len(self.AGE_LABELS)
        values = np.bincount(index, minlength=n).reshape(self.total_phases, -1)
        if depth is None:
            self.counts[key] += values
        else:
            self.counts[key][depth] += values

    def real(self, out):
        # Exactly one set of device->host copies; no per-env synchronizations.
        f, a = host(out['f']), host(out['ep_step'])
        term, done = host(out['term']).astype(bool), host(out['done']).astype(bool)
        if f.ndim != 1 or not (f.shape == a.shape == term.shape == done.shape):
            raise ValueError('real vectors must have matching env dimensions')
        split = len(f)-self.validation_envs
        if split <= 0:
            raise ValueError('holdout consumes all environments')
        for label, ids in [('train', slice(0, split)), ('holdout', slice(split, None))]:
            self._add('real_'+label, f[ids], a[ids])
            self._add('real_term_'+label, f[ids], a[ids], term[ids])
            self._add('real_done_'+label, f[ids], a[ids], done[ids])
        self.calls['real'] += 1
        self._mark()

    def wm(self, window):
        # Only called by wm_update, not sample_windows/held-out validation.
        f, a, m = host(window['f']), host(window['ep_step']), host(window['mask'])
        if f.shape != a.shape or f.shape != m.shape or f.ndim != 2 or f.shape[1] != self.wm_window:
            raise ValueError('WM shapes do not match configured window')
        if not np.isin(m, [0, 1]).all() or not (m[:, 0] == 1).all() or np.any(np.diff(m, axis=1) > 0):
            raise ValueError('WM mask must retain first transition and never recover')
        for depth in range(self.wm_window):
            self._add('wm_train_valid', f[:, depth], a[:, depth], m[:, depth], depth)
        self.calls['wm_train'] += 1
        self._mark()

    def actor(self, start):
        self._add('actor_start', start['f'], start['ep_step'])
        self.calls['actor'] += 1
        self._mark()

    def due(self, *, every_iterations=1000, every_seconds=600):
        return ((self.iteration-self.last_flush_iteration >= every_iterations)
                or time.monotonic()-self.last_flush >= every_seconds)

    def flush(self, reason):
        if self.first_iteration is None:
            return None
        self.output.mkdir(parents=True, exist_ok=True)
        destination = self.output / f'{self.session}-{self.sequence:06d}.json'
        if destination.exists():
            raise FileExistsError(destination)
        payload = {
            'schema': 1, 'kind': 'observed_training_coverage_interval',
            'session': self.session, 'sequence': self.sequence, 'reason': str(reason),
            'resumed_from_iteration': self.resumed_iteration,
            'first_observed_iteration': self.first_iteration,
            'last_observed_iteration': self.last_iteration, 'labels': sorted(self.labels),
            'counts': {k: v.tolist() for k, v in self.counts.items()}, 'calls': self.calls,
            'clip_names': self.names, 'clip_start': self.starts.tolist(), 'clip_end_inclusive': self.ends.tolist(),
            'phase_frames': self.phase_frames, 'phase_offsets': self.offsets.tolist(),
            'age_bins_pretransition': self.AGE_LABELS,
            'layout': 'phase axis is concatenated COUNT BINS with explicit offsets, never reference trajectories; WM first axis is depth 0..K-1',
            'boundaries': 'real f and ep_step are pretransition; WM valid mask includes physical terminal; actor is accepted actual replay start',
            'limitations': 'Counts cover only this instrumented process interval, with replacement for samples; no reconstruction of old replay/updates; phase and age do not certify distribution match to full-source accumulated states',
        }
        temporary = destination.with_suffix('.json.tmp')
        with temporary.open('x') as stream:
            json.dump(payload, stream, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        self.sequence += 1
        self.last_flush_iteration = self.iteration
        self.last_flush = time.monotonic()
        self._reset()
        return destination
