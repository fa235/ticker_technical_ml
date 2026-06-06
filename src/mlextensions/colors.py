"""Color helpers for technical indicator output."""

from __future__ import annotations

from typing import Iterable


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    color = hex_color.strip().lstrip("#")
    if len(color) not in (6, 8):
        raise ValueError("hex_color must be #RRGGBB or #RRGGBBAA")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, round(v))) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def get_color_shades(hex_color: str) -> list[str]:
    """Return 25%, 50%, 75%, and 100% intensity shades of a color."""
    r, g, b = _hex_to_rgb(hex_color)
    return [_rgb_to_hex((r * i / 100, g * i / 100, b * i / 100)) for i in (25, 50, 75, 100)]


def get_prediction_color(prediction: float, neighbors_count: int, shades: Iterable[str]) -> str | None:
    """Choose a shade based on prediction percentile."""
    shades_list = list(shades)
    if neighbors_count == 0:
        return None
    percentile = prediction / neighbors_count * 100
    if percentile >= 75:
        return shades_list[3]
    if percentile >= 50:
        return shades_list[2]
    if percentile >= 25:
        return shades_list[1]
    if percentile >= 0:
        return shades_list[0]
    return None


def color_green(prediction: float) -> str:
    """Return green color strength for bullish predictions."""
    if prediction >= 9:
        return "#15FF00"
    if prediction >= 8:
        return "#15FF00E5"
    if prediction >= 7:
        return "#09FF00CC"
    if prediction >= 6:
        return "#09FF00B2"
    if prediction >= 5:
        return "#09FF0099"
    if prediction >= 4:
        return "#15FF007F"
    if prediction >= 3:
        return "#00FF0066"
    if prediction >= 2:
        return "#09FF004C"
    if prediction >= 1:
        return "#09FF0033"
    return "#15FF0019"


def color_red(prediction: float) -> str:
    """Return red color strength for bearish predictions."""
    if prediction >= 9:
        return "#CC3311"
    if prediction >= 8:
        return "#CC3311E5"
    if prediction >= 7:
        return "#B23111CC"
    if prediction >= 6:
        return "#B23111B2"
    if prediction >= 5:
        return "#B2311199"
    if prediction >= 4:
        return "#CC33117F"
    if prediction >= 3:
        return "#CC331166"
    if prediction >= 2:
        return "#CC33114C"
    if prediction >= 1:
        return "#CC331133"
    return "#CC331119"
