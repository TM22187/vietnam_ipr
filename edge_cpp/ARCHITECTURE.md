# Runtime contract

## Process boundary

`vlpr_edge` owns camera capture and inference only. It never opens a GUI and never
drives a barrier relay directly. Its stable integration boundary is newline-delimited
JSON on stdout and, when configured, an append-only JSONL file.

Exit codes:

- `0`: clean stop, successful dry run, benchmark or accepted self-test.
- `1`: configuration, model, source or runtime failure.
- `3`: self-test completed but produced no accepted plate.

## Reliability decisions

- Stream OCR is cached by spatial track; invalid text is retried at a bounded rate.
- Duplicate accepted plates are suppressed for a configurable interval.
- A transient camera outage triggers reconnect instead of process termination.
- A supervisor remains mandatory. The supplied systemd unit restarts failures and
  restricts filesystem/device access.
- GPIO/relay output is a separate fail-closed process. Recognition is evidence, not
  authorization.

## Hardware backends

The first backend is ONNX Runtime CPU because it is portable and deterministic across
x86-64 and ARM64 Linux. Keep preprocessing, postprocessing and the JSON contract when
adding an accelerator:

- NVIDIA Jetson: TensorRT execution provider or serialized TensorRT engines.
- Intel x86/iGPU: OpenVINO execution provider.
- RK3588: convert models to RKNN and implement the same detector/recognizer interfaces.
- Raspberry Pi: ONNX Runtime ARM64 initially; NCNN is the smaller fallback.

Benchmark on the final camera resolution and hardware before selecting frame stride,
thread count, quantization or accelerator settings.
