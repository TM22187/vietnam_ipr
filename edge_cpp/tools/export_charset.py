"""Export RapidOCR's embedded UTF-8 character list for the C++ runtime."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map
    characters = metadata.get("character", "").splitlines()
    if not characters:
        raise RuntimeError("ONNX model has no 'character' metadata")
    output_shape = session.get_outputs()[0].shape
    class_count = output_shape[-1]
    if not isinstance(class_count, int) or class_count != len(characters) + 2:
        raise RuntimeError(
            f"Unexpected OCR class count: model={class_count}, charset={len(characters)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(characters) + "\n", encoding="utf-8", newline="\n")
    print(f"Exported {len(characters)} characters to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
