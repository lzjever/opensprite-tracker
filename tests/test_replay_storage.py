"""Replay backing must preserve real terminal transitions and reset boundaries."""
import pytest
import torch

from opensprite_tracker.supermimic.buffer import ReplayBuffer


def transition(time, num_envs=2):
    envs = torch.arange(num_envs)
    return {
        "s": torch.stack((torch.full((num_envs,), float(time)), envs.float()), -1),
        "s1": torch.stack((torch.full((num_envs,), float(time+1)), envs.float()), -1),
        "a": envs.float()[:, None],
        "a_prev": torch.full((num_envs, 1), float(time)),
        "xi": envs.float()[:, None] + .5,
        "r": torch.full((num_envs,), float(time)),
        "f": time*100 + envs,
        "ep_step": (time % (envs+3)).to(torch.int32),
        "done": ((time+1) % (envs+3)) == 0,
    }


def filled_buffer(device="cpu", storage_device="cpu"):
    replay = ReplayBuffer(5, 2, device, state_dim=2, action_dim=1,
                          context_dim=1, storage_device=storage_device)
    for time in range(7):
        replay.add(transition(time))
    return replay


def test_cpu_backing_retains_ring_order_and_terminal_transition():
    replay = filled_buffer()
    torch.manual_seed(31)
    windows = replay.sample_windows(64, 4)
    assert replay.size == 5 and replay.ptr == 2
    assert all(value.device.type == "cpu" and not value.is_pinned()
               for value in replay.data.values())
    for row in range(64):
        times = windows["s"][row, :, 0].long().tolist()
        envs = windows["s"][row, :, 1].long().tolist()
        assert times[0] in (2, 3)
        assert times == list(range(times[0], times[0]+4))
        assert len(set(envs)) == 1
        assert windows["f"][row].tolist() == [time*100+envs[0] for time in times]
        # The physical s1 of the terminal transition is retained, not a reset.
        assert windows["s1"][row, :, 0].tolist() == [time+1 for time in times]
        terminal = windows["done"][row].nonzero().flatten().tolist()
        valid_count = terminal[0]+1 if terminal else 4
        assert windows["mask"][row].tolist() == [1.]*valid_count + [0.]*(4-valid_count)


def test_cpu_backing_history_never_reads_before_episode_or_oldest_row():
    replay = filled_buffer()
    torch.manual_seed(47)
    start, history = replay.sample_starts(128, 6)
    assert history["s"].shape == (128, 6, 2)
    assert set(start["f"].tolist()) == {time*100+env for time in range(2, 7) for env in range(2)}
    for row, frame in enumerate(start["f"].tolist()):
        time, env = divmod(frame, 100)
        episode_start = time - time % (env+3)
        oldest = max(2, episode_start)
        expected = [max(oldest, time-back)*100+env for back in range(5, -1, -1)]
        assert history["f"][row].tolist() == expected
        assert int(history["f"][row, -1]) == frame
    history["s"].zero_()
    # Advanced-indexed samples, even on CPU, cannot mutate the replay.
    assert replay.data["s"].abs().sum() > 0


def test_storage_indices_and_output_device_are_independent_without_gpu():
    # Meta exercises a distinct learner destination without allocating CUDA.
    replay = filled_buffer(device="meta", storage_device="cpu")
    windows = replay.sample_windows(4, 3)
    start, history = replay.sample_starts(4, 6)
    assert all(value.device.type == "cpu" for value in replay.data.values())
    assert all(value.device.type == "meta" for value in windows.values())
    assert all(value.device.type == "meta" for value in start.values())
    assert all(value.device.type == "meta" for value in history.values())
    assert windows["mask"].shape == (4, 3)
    assert history["s"].shape == (4, 6, 2)


def test_add_validation_is_atomic_and_storage_size_matches_paper_capacity():
    replay = filled_buffer()
    before = {key:value.clone() for key,value in replay.data.items()}
    ptr, size = replay.ptr, replay.size
    invalid = transition(7)
    invalid["xi"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="xi"):
        replay.add(invalid)
    assert (replay.ptr, replay.size) == (ptr, size)
    for key,value in replay.data.items():
        torch.testing.assert_close(value, before[key])
    small = ReplayBuffer(3, 2, "cpu", storage_device="cpu", state_dim=141,
                         action_dim=31, context_dim=221,
                         extra_fields={"transition_metrics": (98, torch.float32)})
    per_transition = sum(value.numel()*value.element_size() for value in small.data.values()) // 6
    assert per_transition == 2669
    assert per_transition*4096*1000 == 10_932_224_000
