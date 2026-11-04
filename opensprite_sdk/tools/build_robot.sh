#!/usr/bin/env bash
set -euo pipefail
sdk_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${OPENSPRITE_PYTHON:-python3}"
with_mujoco=ON
build_directory="$sdk_root/native-runtime/build"
if [[ "${1:-}" == "--hardware-only" ]]; then
  with_mujoco=OFF
  build_directory="$sdk_root/native-runtime/build-hardware"
elif [[ $# -gt 0 ]]; then
  echo "Usage: bash tools/build_robot.sh [--hardware-only]" >&2
  exit 2
fi
mujoco_directory="${MUJOCO_ROOT:-}"
if [[ "$with_mujoco" == ON && -z "$mujoco_directory" ]]; then
  mujoco_directory="$("$python_bin" -c 'import pathlib,mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)')"
fi
"$python_bin" "$sdk_root/tools/generate_dds_schema.py" --check
cmake -S "$sdk_root/native-runtime" -B "$build_directory" \
  -DCMAKE_BUILD_TYPE=Release -DOPENSPRITE_WITH_MUJOCO="$with_mujoco" -DMUJOCO_ROOT="$mujoco_directory"
cmake --build "$build_directory" --parallel "${OPENSPRITE_BUILD_JOBS:-2}"
