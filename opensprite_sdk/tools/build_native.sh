#!/usr/bin/env bash
set -euo pipefail
sdk_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cmake -S "$sdk_root/native" -B "${1:-$sdk_root/build/native}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${1:-$sdk_root/build/native}" --parallel 2
