"""CPU-only frame/HUD contract checks; never create a MuJoCo Renderer."""

import numpy as np
import pytest

from opensprite_tracker.supermimic.render import (
    continuation_mode, failure_annotation, policy_label, recording_frame_plan,
)


def recording(states=101, failure=40, continued=True):
    live = np.ones((states, 1), dtype=bool)
    if failure is not None:
        live[failure:] = False
    arrays = {
        "root_pos": np.arange(states*3, dtype=float).reshape(states, 1, 3),
        "live": live,
        "ref_frame": np.arange(states)[:, None],
    }
    metadata = {"expected_transitions": states-1, "requested_transitions": states-1,
        "executed_transitions": states-1, "complete_clip": True,
        "training_clip_count": 1, "success_rate": 0. if failure is not None else 1.}
    if continued:
        metadata.update(continue_after_failure=True, post_failure_policy="continue_actor",
                        no_reset=True, no_reference_wrapping=True)
    return arrays, metadata


def plan(arrays, metadata, **kwargs):
    return recording_frame_plan(arrays, metadata, env=0, source_fps=50., video_fps=50., **kwargs)


def test_full_dance_keeps_every_real_state_after_failure_and_never_holds():
    arrays, metadata = recording(11284, 9689)
    before = arrays["root_pos"].copy()
    result = plan(arrays, metadata, failure_hold_seconds=99.)
    assert result["indices"] == list(range(11284))
    assert result["interval"] == (0, 11284) and result["recorded_end"] == 11283
    assert result["first_failure"] == 9689 and result["hold_frames"] == 0
    assert result["full_recorded_span_rendered"] and result["full_source_recorded"]
    assert result["indices"][-1]/50 == 225.66
    assert len(result["indices"])/50 == 225.68  # Container includes initial state.
    np.testing.assert_array_equal(arrays["root_pos"], before)
    assert metadata["success_rate"] == 0.  # Full footage does not change success.


def test_legacy_rollout_still_truncates_at_first_failure_and_uses_old_hold():
    arrays, metadata = recording(101, 40, continued=False)
    result = plan(arrays, metadata)
    assert result["indices"] == list(range(41)) and result["recorded_end"] == 40
    assert result["hold_frames"] == 75 and not result["full_recorded_span_rendered"]
    assert result["continue_after_failure"] is False
    with pytest.raises(ValueError, match="outside"):
        plan(arrays, metadata, preview_frame=80)
    with pytest.raises(ValueError, match="assertion"):
        plan(arrays, metadata, include_post_failure=True)


def test_postfailure_preview_and_qa_limits_are_explicit_not_full_span():
    arrays, metadata = recording()
    preview = plan(arrays, metadata, preview_frame=80)
    assert preview["indices"] == [80] and preview["hold_frames"] == 0
    assert not preview["full_recorded_span_rendered"]
    short = plan(arrays, metadata, max_render_frames=10)
    assert short["indices"] == list(range(10)) and short["hold_frames"] == 0
    assert not short["full_recorded_span_rendered"]
    after = plan(arrays, metadata, start_seconds=1.5)
    assert after["indices"][0] == 75 and after["indices"][-1] == 100
    with pytest.raises(ValueError, match="assertion"):
        plan(arrays, metadata, include_post_failure=False)


@pytest.mark.parametrize("broken", ["frozen_ref", "reset_live", "wrong_count", "false_complete", "zero_actor", "reset_metadata"])
def test_inconsistent_continued_recordings_fail_before_rendering(broken):
    arrays, metadata = recording()
    if broken == "frozen_ref": arrays["ref_frame"][60:] = 59
    elif broken == "reset_live": arrays["live"][60:] = True
    elif broken == "wrong_count": metadata["executed_transitions"] = 70
    elif broken == "false_complete": metadata["expected_transitions"] = 11283
    elif broken == "zero_actor": metadata["post_failure_policy"] = "zero_action"
    elif broken == "reset_metadata": metadata["no_reset"] = False
    with pytest.raises(ValueError): plan(arrays, metadata)


def test_partial_continued_source_is_not_labelled_complete():
    arrays, metadata = recording()
    metadata.update(expected_transitions=11283, complete_clip=False)
    assert not plan(arrays, metadata)["full_source_recorded"]
    with pytest.raises(ValueError, match="boolean"):
        continuation_mode({"continue_after_failure": "true"})


def test_failure_annotation_discloses_first_time_and_continuation_without_freeze():
    status, message = failure_annotation(False, True, 193.82, held=True)
    assert status == "POST-FAILURE CONTINUING"
    assert "首次失败 193.82s" in message and "失败后持续仿真不重置" in message
    assert "停帧" not in message and "截断" not in message
    assert failure_annotation(True, True, 193.82) == ("LIVE", None)
    assert "停帧回看" in failure_annotation(False, False, 193.82, held=True)[1]
    assert "记录在此截断" in failure_annotation(False, False, 193.82)[1]
    assert policy_label({"training_clip_count": 1}) == "单动作策略"
    assert policy_label({}) == "策略" and "共享" not in policy_label({})
