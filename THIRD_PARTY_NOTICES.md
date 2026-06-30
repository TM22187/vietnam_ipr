# Third-party notices

Vietnam LPR uses or distributes the following runtime components and models:

- ONNX Runtime — Microsoft and contributors — MIT License.
- OpenCV — OpenCV contributors — Apache License 2.0.
- RapidOCR — RapidAI contributors — Apache License 2.0.
- NumPy — NumPy contributors — BSD-3-Clause.
- Pillow — Pillow contributors — HPND License.
- PP-OCRv6 recognition model — model copyright belongs to its respective authors.
- YOLO detector export — trained project model; embedded metadata identifies the
  Ultralytics AGPL-3.0 export toolchain.

The Python desktop distribution retains license files supplied by packaged libraries
where available. The C++ edge package dynamically links ONNX Runtime and OpenCV and
ships the two ONNX models listed in `edge_cpp/models/model_manifest.json`.

This notice does not assign a license to this repository. The repository owner must
select and publish the project license before third-party distribution.
