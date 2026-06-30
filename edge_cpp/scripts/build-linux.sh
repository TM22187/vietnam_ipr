#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

if [[ -z "${ONNXRUNTIME_ROOT:-}" ]]; then
  echo "Set ONNXRUNTIME_ROOT to an extracted ONNX Runtime C/C++ SDK." >&2
  exit 2
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DONNXRUNTIME_ROOT="${ONNXRUNTIME_ROOT}" \
  -DVLPR_BUILD_TESTS=ON
cmake --build "${BUILD_DIR}" --parallel
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "Built ${BUILD_DIR}/vlpr_edge"
