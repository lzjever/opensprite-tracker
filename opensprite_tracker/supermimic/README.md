# OpenSprite SuperMimic

Standalone whole-body tracking of `dance2_subject1` on Sprite0825 with SuperMimic.

The published setup provides an eight-hour cold-start training configuration with explicit world-model warmup. The default reference is `data/motions/lafan_dance.npz`. See the repository's Chinese and English README files for the architecture, data preparation, commands and evaluation limits.

Train: `python -m opensprite_tracker.supermimic.train --help`
Evaluate: `python -m opensprite_tracker.supermimic.evaluate --help`
Render recorded rollout: `python -m opensprite_tracker.supermimic.render --help`

Third party MIT notices are preserved in `THIRD_PARTY_LICENSE.txt` and copied source headers.
