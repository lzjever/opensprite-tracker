"""CPU-only checks for honest full-reference post-failure recordings."""
import pytest
import torch

from opensprite_tracker.supermimic.evaluate import (
    evaluation_action, parser, require_finite_live, stop_after_all_failed,
    update_survival,
)


def test_default_action_and_early_stop_remain_unchanged():
    args = parser().parse_args(["--checkpoint", "unused.pt"])
    assert not args.continue_after_failure
    raw = torch.tensor([[50., -25.], [1., -2.]])
    alive = torch.tensor([True, False])
    torch.testing.assert_close(evaluation_action(raw, alive), torch.tensor([[20., -20.], [0., 0.]]))
    assert not stop_after_all_failed(alive)
    assert stop_after_all_failed(torch.zeros(2, dtype=torch.bool))


def test_continuation_keeps_policy_actions_after_all_fail_without_early_stop():
    args = parser().parse_args(["--checkpoint", "unused.pt", "--continue-after-failure"])
    assert args.continue_after_failure
    dead = torch.zeros(2, dtype=torch.bool)
    raw = torch.tensor([[50., -25.], [1., -2.]])
    torch.testing.assert_close(evaluation_action(raw, dead, True), raw.clamp(-20, 20))
    assert not stop_after_all_failed(dead, True)


def test_first_failure_and_alive_before_scores_stay_sticky_through_recovery():
    alive, first = torch.ones(2, dtype=torch.bool), torch.full((2,), -1)
    masks = []
    # Env0 fails, recovers a physical pose, and fails again; neither event
    # makes the original failure disappear or any later sample scoreable.
    for frame, failures in enumerate(([True, False], [False, True], [True, False]), 1):
        masks.append(alive.clone())
        alive, first = update_survival(alive, first, torch.tensor(failures), torch.full((2,), frame))
    assert first.tolist() == [1, 2]
    assert alive.tolist() == [False, False]
    assert torch.stack(masks).tolist() == [[True, True], [False, True], [False, False]]
    assert not stop_after_all_failed(alive, True)


def test_continued_state_or_action_nan_is_not_hidden_by_dead_scoring_mask():
    alive = torch.tensor([True, False])
    values = (torch.tensor([[1.], [float("nan")]]),)
    # Preserve standard evaluation's scored-only behavior.
    require_finite_live(values, alive, 2, "action")
    # Continuation uses every simulated environment as the finite-check mask.
    with pytest.raises(RuntimeError, match="Nonfinite action.*envs \\[1\\]"):
        require_finite_live(values, torch.ones_like(alive), 2, "action")
