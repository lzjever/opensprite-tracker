#!/usr/bin/env bash
set -euo pipefail
sdk_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$sdk_root/tools/generate_dds_schema.py" --check
cmake -S "$sdk_root/native-dds" -B "$sdk_root/native-dds/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$sdk_root/native-dds/build" --parallel "${OPENSPRITE_BUILD_JOBS:-2}"
