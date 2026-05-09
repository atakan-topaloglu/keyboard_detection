from __future__ import annotations

from dataclasses import dataclass

Point = tuple[float, float]


SUPPORTED_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz "


@dataclass(frozen=True)
class KeyboardLayout:
    name: str
    rows: tuple[str, ...]
    row_offsets: tuple[float, ...]
    key_pitch_x: float = 1.0
    key_pitch_y: float = 1.0
    space_center: Point = (5.2, 4.15)

    @property
    def labels(self) -> list[str]:
        return [label for row in self.rows for label in row]

    def centers(self) -> dict[str, Point]:
        points: dict[str, Point] = {}
        for row_index, row in enumerate(self.rows):
            y = float(row_index) * self.key_pitch_y
            offset = self.row_offsets[row_index]
            for col_index, label in enumerate(row):
                x = offset + float(col_index) * self.key_pitch_x
                points[label] = (x, y)
        points[" "] = self.space_center
        return points

    def anchor_centers(self) -> dict[str, Point]:
        centers = self.centers()
        return {label: centers[label] for label in self.labels}


QWERTZ_LAYOUT = KeyboardLayout(
    name="qwertz",
    rows=("1234567890", "qwertzuiop", "asdfghjkl", "yxcvbnm"),
    row_offsets=(0.0, 0.35, 0.55, 0.95),
)


def normalize_label(raw_label: str) -> str | None:
    label = str(raw_label).strip().lower()
    aliases = {
        "space": " ",
        "spacebar": " ",
        "space bar": " ",
        "keyboard": None,
        "key": None,
    }
    if label in aliases:
        return aliases[label]
    if len(label) == 1 and label in SUPPORTED_CHARS:
        return label
    return None

