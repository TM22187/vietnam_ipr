"""Sinh icon đa kích thước, không phụ thuộc asset bên ngoài."""

from pathlib import Path

from PIL import Image, ImageDraw


def build_icon(size: int = 256) -> Image.Image:
    image = Image.new("RGBA", (size, size), "#0f172a")
    draw = ImageDraw.Draw(image)
    margin = size // 7
    radius = size // 12
    plate = (margin, size // 3, size - margin, size * 2 // 3)
    draw.rounded_rectangle(plate, radius=radius, fill="#f8fafc", outline="#22c55e", width=size // 35)

    # Các ô ký tự cách điệu, vẫn rõ khi thu nhỏ.
    box_w = size // 12
    gap = size // 28
    start_x = size // 2 - (box_w * 4 + gap * 3) // 2
    top = size // 2 - size // 18
    for index in range(4):
        x = start_x + index * (box_w + gap)
        draw.rounded_rectangle(
            (x, top, x + box_w, top + size // 9),
            radius=max(2, size // 80), fill="#172033",
        )

    # Bốn góc quét nhận dạng.
    color = "#22c55e"
    width = max(3, size // 32)
    length = size // 6
    inset = size // 12
    for x, y, sx, sy in (
        (inset, inset, 1, 1), (size - inset, inset, -1, 1),
        (inset, size - inset, 1, -1), (size - inset, size - inset, -1, -1),
    ):
        draw.line((x, y, x + sx * length, y), fill=color, width=width)
        draw.line((x, y, x, y + sy * length), fill=color, width=width)
    return image


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    destination = root / "assets" / "app.ico"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = build_icon()
    image.save(destination, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(destination)


if __name__ == "__main__":
    main()
