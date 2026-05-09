from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def draw_target_overlay(
    image_path: str | Path,
    target_map: dict[str, Any],
    output_path: str | Path,
) -> None:
    with Image.open(image_path) as raw:
        image = ImageOps.exif_transpose(raw)
        if image.mode != "RGB":
            image = image.convert("RGB")
        canvas = image.copy()

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    targets = target_map.get("key_targets", {})
    requested = target_map.get("request", {}).get("target_text", "")
    requested_labels = set(requested.lower())
    for label, target in targets.items():
        x, y = target["center_px"]
        is_requested = label in requested_labels
        color = (0, 220, 80) if not is_requested else (255, 80, 0)
        radius = 8 if is_requested else 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=3)
        text = "space" if label == " " else label
        draw.text((x + radius + 2, y - radius), text, fill=color, font=font)

    tip = target_map.get("tip")
    if tip and tip.get("center_px"):
        x, y = tip["center_px"]
        draw.line((x - 14, y, x + 14, y), fill=(0, 180, 255), width=3)
        draw.line((x, y - 14, x, y + 14), fill=(0, 180, 255), width=3)
        draw.text((x + 16, y + 4), "tip", fill=(0, 180, 255), font=font)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=95)

