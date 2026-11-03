# Upstream native CAN controller

Source: https://github.com/BeijingDynamics/open_sprite_runtime
Commit: `4515206c24404e18a2d5c168eccdb7468aea6ab7`
File: `native/src/sprite_can_shadow.cpp`
License: AGPL-3.0-only; full license is retained alongside the source.

Local changes add an exclusive host-wide CAN owner lock, reject existing IPC paths, and add per-motor receive freshness checks before control and state publication. State packets use SDK ABI2 (776 bytes; kind 1 for actuation, kind 3 for zero-gain-only shadow) with 31 host receive timestamps appended to the original state body; targets retain upstream ABI1 (1280 bytes). Original finite-duration acknowledgements, all-disabled precondition, motor envelopes, shutdown and disable verification are retained. SDK build supplies a direct Linux C++17 target; no upstream installation is needed.
