# UMR runtime

This directory contains the native UMR correspondence-learning and surface-guided motion optimization code used by this demo.

UMR: *Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence*. Authors: Hanyang Cao, Yuetong Fang, Taesoo Kwon, Runyi Yu, Ji Ma, Jing Tan, Yangchen Zhou, Baoze Du, Yi Gu, Yukang Gao, Ruoli Dai, Lei Han and Renjing Xu. [Paper](https://arxiv.org/abs/2609.02134) · [Project](https://hanyang9.github.io/UMR/).

The OpenSprite/Geno adapter and repository-local entry point are in `../../retarget/umr_pipeline.py` and `../../scripts/retarget_lafan.py`. Use the separate environment from `../../requirements-umr.txt`. The required FBX parser and Three.js files are bundled under `scripts/`; Node.js 18 or later is required.

The released input is the complete `dance2_subject1` BVH. Robot geometry, source surface, learned correspondence, constraints and output paths are configured by the adapter. No upstream example datasets are required. Follow the [repository reproduction guide](../../docs/reproduce.md) for build, correspondence training, optimization and conversion.

This third-party code retains its [MIT license](LICENSE). Bundled JavaScript dependencies retain their included notices.
