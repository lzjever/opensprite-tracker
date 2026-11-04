#!/usr/bin/env bash
set -euo pipefail
sdk_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
generator_version="$(fastddsgen -version)"
if [[ "$generator_version" != *"version 2.3.0"* ]]; then
  echo "Expected fastddsgen 2.3.0; found: $generator_version" >&2
  exit 1
fi
python3 "$sdk_root/tools/generate_dds_schema.py"
fastddsgen -replace -typeros2 -d "$sdk_root/native-dds/generated" "$sdk_root/native-dds/idl/OpenSprite.idl"
python3 "$sdk_root/tools/generate_dds_schema.py" --record-native
