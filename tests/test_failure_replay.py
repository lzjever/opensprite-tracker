"""Physical-failure priority must retain real replay chronology and holdout semantics."""

from types import SimpleNamespace

import torch

from opensprite_tracker.supermimic.buffer import ReplayBuffer


def make_buffer(*, T=64, N=4, fraction=0.2, validation_envs=1,
                lookback=10, gap=2):
    return ReplayBuffer(
        T, N, "cpu", state_dim=2, action_dim=1, context_dim=1,
        validation_envs=validation_envs, failure_replay_fraction=fraction,
        failure_lookback_steps=lookback, failure_gap_steps=gap,
        extra_fields={"timeout": ((), torch.bool)},
    )


def add_rows(buffer, start, stop, *, events=None, ages=None):
    """Encode absolute row/env in states; all labels are pre-reset labels."""
    events = events or {}
    ages = list(ages if ages is not None else [0] * buffer.N)
    for t in range(start, stop):
        row = {key: torch.zeros(storage.shape[1:], dtype=storage.dtype)
               for key, storage in buffer.data.items()}
        row["s"][:, 0] = t
        row["s"][:, 1] = torch.arange(buffer.N)
        row["s1"][:, 0] = t + 1
        row["s1"][:, 1] = torch.arange(buffer.N)
        row["ep_step"][:] = torch.tensor(ages, dtype=torch.int32)
        row["f"][:] = t
        for env in range(buffer.N):
            term, timeout = events.get((t, env), (False, False))
            if "term" in row:
                row["term"][env] = term
            row["timeout"][env] = timeout
            row["done"][env] = term or timeout
            ages[env] = 0 if term or timeout else ages[env] + 1
        buffer.add(row)
    return ages


def populated(fraction=0.2):
    buffer = make_buffer(fraction=fraction)
    # A physical failure can coincide with timeout; timeout alone is not failure.
    add_rows(buffer, 0, 35, events={
        (30, 0): (True, True), (30, 1): (False, True),
        (30, 3): (True, False),  # Trailing WM holdout environment.
    })
    return buffer


def test_only_physical_train_failures_create_exact_lookback_candidates():
    buffer = populated()
    candidates = buffer.priority_candidates()
    assert set(map(tuple, candidates.tolist())) == {(t, 0) for t in range(20, 29)}
    assert buffer.failure_replay_stats["physical_events"] == 1
    # 50 Hz production bounds correspond to the requested 0.2--2 seconds.
    production = make_buffer(T=160, N=1, validation_envs=0, lookback=100, gap=10)
    add_rows(production, 0, 131, events={(120, 0): (True, False)})
    assert set(production.priority_candidates()[:, 0].tolist()) == set(range(20, 111))


def test_validation_is_identical_uniform_and_private_generators_stay_isolated(monkeypatch):
    enabled, legacy = populated(), populated(fraction=0)
    def forbidden(*args, **kwargs):
        raise AssertionError("validation must not query failure priority")
    monkeypatch.setattr(enabled, "priority_candidates", forbidden)
    g1, g2 = torch.Generator().manual_seed(2026), torch.Generator().manual_seed(2026)
    actual = enabled.sample_windows(128, 8, split="validation", generator=g1)
    expected = legacy.sample_windows(128, 8, split="validation", generator=g2)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert (actual["s"][..., 1] == 3).all()
    assert enabled.failure_replay_stats["wm_priority_requested"] == 0
    # Enabled training priority must honor the caller's private generator too.
    training = populated()
    torch.manual_seed(301)
    global_before = torch.get_rng_state().clone()
    train1 = training.sample_windows(64, 4, generator=torch.Generator().manual_seed(71))
    assert torch.equal(torch.get_rng_state(), global_before)
    torch.rand(17)  # Unrelated global draws must not affect the private stream.
    train2 = training.sample_windows(64, 4, generator=torch.Generator().manual_seed(71))
    for key in train1:
        torch.testing.assert_close(train1[key], train2[key], rtol=0, atol=0)


def test_priority_windows_are_resident_consecutive_and_stop_at_reset():
    buffer = make_buffer(T=16, N=1, validation_envs=0, lookback=12, gap=0)
    add_rows(buffer, 0, 30, events={(24, 0): (True, False)})
    candidates = buffer.priority_candidates(k=4, hist=3)
    assert len(candidates) > 0
    for absolute, env in candidates.tolist():
        assert absolute >= 14 and absolute + 3 <= 24
        rows = torch.arange(absolute, absolute + 4) % buffer.T
        assert torch.equal(buffer.data["s"][rows, env, 0], torch.arange(absolute, absolute + 4).float())
        assert torch.equal(buffer.data["ep_step"][rows, env], torch.arange(absolute, absolute + 4).int())
        assert not buffer.data["done"][rows[:-1], env].any()
    assert (21, 0) in map(tuple, candidates.tolist())  # Includes terminal transition.
    assert (22, 0) not in map(tuple, candidates.tolist())


def test_actor_history_pads_with_current_episode_and_never_prior_timeout():
    buffer = make_buffer(N=1, validation_envs=0)
    add_rows(buffer, 0, 24, events={(16, 0): (False, True), (20, 0): (True, False)})
    assert set(map(tuple, buffer.priority_candidates(hist=8).tolist())) == {(17, 0), (18, 0)}
    torch.manual_seed(73)
    start, history = buffer.sample_starts(100, hist=8)
    assert start["_failure_priority"].sum().item() == 2  # Sparse pool is not duplicated.
    start_episode = start["s"][:, 0] - start["ep_step"]
    history_episode = history["s"][..., 0] - history["ep_step"]
    assert torch.equal(history_episode, start_episode[:, None].expand_as(history_episode))
    assert (history["s"][..., 0] <= start["s"][:, :1]).all()
    early = start["_failure_priority"] & (start["ep_step"] == 0)
    assert early.any()
    assert (history["s"][early, :, 0] == 17).all()


def test_same_ring_slot_and_age_cannot_resurrect_overwritten_endpoint():
    buffer = make_buffer(T=16, N=1, validation_envs=0)
    ages = add_rows(buffer, 0, 16, events={(14, 0): (True, False), (15, 0): (False, True)})
    assert len(buffer.priority_candidates()) == 9
    # New endpoint has exactly the old slot and ep_age token. Only its own
    # absolute sequence can contribute; stale event must not duplicate weight.
    ages = add_rows(buffer, 16, 32, ages=ages, events={(30, 0): (True, False)})
    candidates = buffer.priority_candidates()
    assert candidates.shape == (9, 2)
    assert set(map(tuple, candidates.tolist())) == {(t, 0) for t in range(20, 29)}
    add_rows(buffer, 32, 48, ages=ages)
    assert buffer.priority_candidates().shape == (0, 2)


def test_windows_mask_terminal_itself_but_not_postreset_successors():
    buffer = populated()
    torch.manual_seed(191)
    windows = buffer.sample_windows(400, 8)
    assert (windows["s"][..., 1] < 3).all()
    assert (windows["s"][:, 1:, 0] - windows["s"][:, :-1, 0] == 1).all()
    saw_term = saw_timeout_only = saw_masked = False
    for done, term, timeout, mask in zip(windows["done"], windows["term"], windows["timeout"], windows["mask"]):
        expected = torch.ones(8)
        boundaries = done.nonzero().flatten()
        if len(boundaries):
            expected[int(boundaries[0]) + 1:] = 0
            saw_masked |= bool((expected == 0).any())
        assert torch.equal(mask, expected)
        saw_term |= bool(term.any())
        saw_timeout_only |= bool((timeout & ~term).any())
    assert saw_term and saw_timeout_only and saw_masked


def test_timeout_only_empty_pool_falls_back_without_excluding_actor_holdout():
    buffer = make_buffer()
    add_rows(buffer, 0, 35, events={(20, 0): (False, True)})
    assert not len(buffer.priority_candidates())
    windows = buffer.sample_windows(100, 8)
    start, history = buffer.sample_starts(200, 4)
    assert windows["s"].shape == (100, 8, 2)
    assert history["s"].shape == (200, 4, 2)
    assert not start["_failure_priority"].any()
    assert (start["s"][:, 1] == 3).any()  # Original uniform actor scope survives.
    assert buffer.failure_replay_stats["wm_priority_selected"] == 0


def test_zero_fraction_keeps_old_fields_draw_order_and_rng_exactly(monkeypatch):
    buffer = populated(fraction=0)
    assert "term" not in buffer.data
    def forbidden(*args, **kwargs):
        raise AssertionError("zero fraction must not access priority")
    monkeypatch.setattr(buffer, "priority_candidates", forbidden)
    g_actual = torch.Generator().manual_seed(99)
    g_expected = torch.Generator().manual_seed(99)
    age = torch.randint(buffer.size - 8 + 1, (32,), generator=g_expected)
    env = torch.randint(0, 3, (32,), generator=g_expected)
    windows = buffer.sample_windows(32, 8, generator=g_actual)
    assert torch.equal(windows["s"][:, 0, 0], age.float())
    assert torch.equal(windows["s"][:, 0, 1], env.float())
    assert torch.equal(g_actual.get_state(), g_expected.get_state())
    torch.manual_seed(100)
    expected_age, expected_env = torch.randint(buffer.size, (32,)), torch.randint(4, (32,))
    expected_state = torch.get_rng_state().clone()
    torch.manual_seed(100)
    start, _ = buffer.sample_starts(32, 4)
    assert torch.equal(start["s"][:, 0], expected_age.float())
    assert torch.equal(start["s"][:, 1], expected_env.float())
    assert "_failure_priority" not in start
    assert torch.equal(torch.get_rng_state(), expected_state)


def test_priority_quota_is_bounded_without_replacement_and_randomly_inserted():
    buffer = populated()
    torch.manual_seed(42)
    start, _ = buffer.sample_starts(100, 1)
    flags = start["_failure_priority"]
    assert flags.sum().item() == 9  # min(20 requested, 9 actual candidates).
    priority_states = start["s"][flags]
    assert len(priority_states.unique(dim=0)) == 9
    assert (priority_states[:, 1] == 0).all()
    assert flags[:50].any() and flags[50:].any()  # No priority prefix.
    buffer.sample_windows(100, 4)
    assert buffer.failure_replay_stats["wm_priority_requested"] == 20
    # Starting at 28 would cross the term at 30 into reset row 31.
    assert buffer.failure_replay_stats["wm_priority_selected"] == 8


def test_engine_source_h_rejection_keeps_actual_quota_history_and_empty_fallback():
    from opensprite_tracker.supermimic.engine import Learner

    for seed, empty, second_end in ((0, False, 34), (7, False, 34),
                                    (0, False, 33), (0, True, 34)):
        buffer = make_buffer()
        events = {} if empty else {(30, 0): (True, False)}
        add_rows(buffer, 0, 35, events=events)
        learner = object.__new__(Learner)
        learner.buffer = buffer
        learner.args = SimpleNamespace(pi_batch=10, hist=4, horizon=8)
        learner.motion = SimpleNamespace(
            clip_id=torch.tensor([0] * 25 + [1] * 10),
            clip_end=torch.tensor([24, second_end]),
        )
        # Of nine real failure candidates, only rows25,26 fit inside either
        # source's H-step extent. Source-end rejection must precede the quota.
        candidates = buffer.priority_candidates(hist=4)
        if not empty:
            legal = [t for t, _ in candidates.tolist()
                     if t + 8 <= (24 if t < 25 else second_end)]
            assert sorted(legal) == ([25, 26] if second_end == 34 else [25])
        torch.manual_seed(seed)
        start, history = learner.sample_valid_starts()
        assert len(start["f"]) == 10
        end = learner.motion.clip_end[learner.motion.clip_id[start["f"]]]
        assert (start["f"] + 8 <= end).all()
        flags = start["_failure_priority"]
        expected_count = 0 if empty else min(2, len(legal))
        assert flags.sum().item() == expected_count
        if not empty:
            assert set(map(tuple, start["s"][flags].tolist())) == {(float(t), 0.) for t in legal}
        assert "_failure_priority" not in history
        for key in history:
            torch.testing.assert_close(history[key][:, -1], start[key], rtol=0, atol=0)
        episode = start["s"][:, 0] - start["ep_step"]
        assert torch.equal(history["s"][..., 0] - history["ep_step"],
                           episode[:, None].expand(10, 4))
        assert buffer.failure_replay_stats["actor_total_accepted"] == 10
        assert buffer.failure_replay_stats["actor_priority_accepted"] == expected_count
