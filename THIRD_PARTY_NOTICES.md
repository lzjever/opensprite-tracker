# Third party code and assets

`opensprite_tracker/supermimic` directly adapts the validated SuperMimic implementation in https://github.com/lzjever/beyondmimic-umr, branch `exp/agibot-x2-supermimic-fullbody-joint-2h-20261022`, commit `32fc15e2d2a2f455cea23df2d1f6a96e3f31fab5`. Only the OpenSprite robot, motion identity, tensor sizes, collision geometry and necessary launch/resume details are adapted.

The underlying independent SuperMimic implementation originated in https://github.com/zita-ch/supermimic_vibed, Copyright (c) 2026 ETH Zurich, Chong Zhang. Its MIT notices are preserved below and in copied source headers.

Official Sprite0825 assets retain their AGPL-3.0 license at `assets/opensprite/vendor/LICENSE`. Motion and native UMR third party notices are retained with their respective data/source directories.

The robot-independent algorithm and rotation utilities are adapted from
https://github.com/zita-ch/supermimic_vibed (a1508d4).
This repository is an unofficial implementation, not code released by the paper authors.

MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
