# Upstream native CAN controller

Source: https://github.com/BeijingDynamics/open_sprite_runtime
Commit: `4515206c24404e18a2d5c168eccdb7468aea6ab7`
File: `native/src/sprite_can_shadow.cpp`
License: AGPL-3.0-only; full license is retained alongside the source.

Local changes add an exclusive host-wide CAN owner lock, reject existing IPC paths, and add per-motor receive freshness checks before control and state publication. State packets use SDK ABI2 (776 bytes; kind 1 for actuation, kind 3 for zero-gain-only shadow) with 31 host receive timestamps appended to the original state body; targets retain upstream ABI1 (1280 bytes). Original finite-duration acknowledgements, all-disabled precondition, motor envelopes, shutdown and disable verification are retained. SDK build supplies a direct Linux C++17 target; no upstream installation is needed.

SDK review fixes preserve the joint-space `q/dq/Kp/Kd/tau` impedance interface.
Non-ankle complete PD requests outside the configured joint effort envelope are
rejected, rather than silently switching embedded PD to a feed-forward-only
controller. The ankle host PD retains its explicit saturation. Before each CAN
slot, exact quantized due commands and the other motors' last sent commands are
mapped through `A^T` and checked against every joint's effort limit at each
send-order prefix, before any frame in the slot is sent. This includes
signed/non-unit transmission ratios and staggered differential motor updates.
The same proposed-command guard runs in zero-gain shadow without transmitting
those proposed commands. These are feedback-based software estimates, not a
continuous-time mechanical torque guarantee between feedback samples.

Shutdown now attempts all three disable rounds independently for every motor,
then polls and drains each bus independently and aggregates errors. A failed bus
or malformed feedback cannot skip disable attempts on healthy buses. Both the
protected-policy and measured-hold paths share this cleanup. Shutdown receive
loops have an explicit per-call frame budget and fail real I/O errors instead of
spinning indefinitely. A fault remains a failed report even when reachable motors
have been successfully disabled. Native IPC accepts bounded historical source
states and rejects zero/unknown and future source sequences.
