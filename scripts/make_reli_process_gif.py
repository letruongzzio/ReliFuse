"""Generate the ReliFuse process GIF used by the project page."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "relifuse-process.gif"
W, H = 1100, 620
MASK_SIZE = 136
FRAMES = 72


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for name in names:
        if name:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                pass
    return ImageFont.load_default()


TITLE = font(28, True)
HEAD = font(18, True)
BODY = font(14)
SMALL = font(12)


def ease(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def vessel_field(n: int = MASK_SIZE) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:n, 0:n]
    vessels = np.zeros((n, n), float)
    centerlines = [
        (0.58 * n + 9 * np.sin(y / 14), 0.23 * n, 8),
        (0.35 * n + 13 * np.sin((y + 16) / 18), 0.68 * n, 6),
        (0.82 * n + 7 * np.sin(y / 11), 0.52 * n, 4),
    ]
    for cx, cy, width in centerlines:
        dist = np.abs(x - cx) + 0.28 * np.abs(y - cy)
        vessels = np.maximum(vessels, np.exp(-(dist / width) ** 2))
    branch = np.exp(-(((x - 0.5 * n) - 0.9 * (y - 0.5 * n)) / 7) ** 2)
    branch *= np.exp(-((y - 0.48 * n) / 34) ** 2)
    vessels = np.maximum(vessels, branch)
    mask = (vessels > 0.42).astype(float)
    tissue = np.dstack(
        [
            244 + 8 * np.sin(x / 11) + 5 * np.cos(y / 17),
            202 + 9 * np.sin((x + y) / 23),
            213 + 7 * np.cos(x / 19),
        ]
    )
    tissue -= mask[..., None] * np.array([70, 60, 58])
    return np.clip(tissue, 0, 255).astype(np.uint8), mask


def blur_mask(mask: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(img, dtype=float) / 255


def expert_masks(gt: np.ndarray) -> list[np.ndarray]:
    variants = []
    shifts = [(-2, 1), (1, 0), (2, -1), (-1, -2), (3, 1), (0, 2), (-3, 0)]
    thresholds = [0.35, 0.48, 0.56, 0.43, 0.52, 0.38, 0.58]
    for i, (dy, dx) in enumerate(shifts):
        v = np.roll(gt, (dy, dx), axis=(0, 1))
        v = blur_mask(v, 1.2 + (i % 3) * 0.45)
        yy, xx = np.mgrid[0 : gt.shape[0], 0 : gt.shape[1]]
        local = 0.08 * np.sin(xx / (8 + i) + i) + 0.06 * np.cos(yy / (10 + i))
        variants.append(((v + local) > thresholds[i]).astype(float))
    return variants


def map_image(values: np.ndarray, tint: tuple[int, int, int], alpha: float = 1.0) -> Image.Image:
    values = np.clip(values, 0, 1)
    base = np.full((*values.shape, 3), 248, dtype=np.uint8)
    color = np.array(tint, dtype=float)
    out = base * (1 - values[..., None] * alpha) + color * values[..., None] * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def paste_card(
    frame: Image.Image,
    xy: tuple[int, int],
    title: str,
    image: Image.Image,
    note: str,
    alpha: float = 1.0,
    accent: tuple[int, int, int] = (20, 129, 132),
) -> None:
    x, y = xy
    w, h = 178, 212
    overlay = Image.new("RGBA", (w, h), (255, 255, 255, int(238 * alpha)))
    d = ImageDraw.Draw(overlay)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=14, outline=(218, 226, 232, int(255 * alpha)), width=1)
    d.rounded_rectangle((12, 12, 166, 30), radius=9, fill=(*accent, int(28 * alpha)))
    d.text((20, 14), title, fill=(*accent, int(255 * alpha)), font=SMALL)
    overlay.alpha_composite(image.resize((136, 136)).convert("RGBA"), (21, 44))
    d.text((18, 184), note, fill=(71, 82, 94, int(255 * alpha)), font=SMALL)
    frame.alpha_composite(overlay, xy)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], t: float) -> None:
    sx, sy = start
    ex, ey = end
    mx, my = sx + (ex - sx) * ease(t), sy + (ey - sy) * ease(t)
    color = (33, 97, 140, 220)
    draw.line((sx, sy, mx, my), fill=color, width=5)
    if t > 0.92:
        draw.polygon([(ex, ey), (ex - 14, ey - 8), (ex - 14, ey + 8)], fill=color)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, active: float) -> None:
    x, y = xy
    fill = (18, 28, 36, int(255 * active))
    draw.rounded_rectangle((x, y, x + 166, y + 44), radius=14, fill=(255, 255, 255, int(235 * active)), outline=(216, 225, 232, int(255 * active)))
    draw.text((x + 14, y + 12), text, fill=fill, font=BODY)


def frame_at(i: int) -> Image.Image:
    tissue, gt = vessel_field()
    experts = expert_masks(gt)
    stack = np.mean(experts, axis=0)
    disagreement = np.std(experts, axis=0) * 2.4
    reliability = np.clip(stack * 0.72 + (1 - disagreement) * 0.28, 0, 1)
    ambiguity = np.clip(disagreement + blur_mask(gt, 2.8) * (1 - stack) * 0.6, 0, 1)
    correction = np.clip((gt - stack) * 0.9 + 0.5, 0, 1)
    final = np.clip(stack * reliability + gt * ambiguity * 0.52, 0, 1) > 0.48

    p = i / (FRAMES - 1)
    frame = Image.new("RGBA", (W, H), (246, 248, 249, 255))
    d = ImageDraw.Draw(frame, "RGBA")
    d.rectangle((0, 0, W, 102), fill=(17, 28, 34, 255))
    d.text((38, 28), "ReliFuse: reliability-calibrated posterior fusion", fill=(255, 255, 255, 255), font=TITLE)
    d.text((40, 66), "K=7 frozen expert masks -> diagnostic state -> local reliability -> ambiguity-gated residual -> fused vessel mask", fill=(205, 223, 229, 255), font=BODY)

    tissue_img = Image.fromarray(tissue).resize((200, 200))
    paste_card(frame, (34, 136), "Histology patch", tissue_img, "image only feeds experts", 1, (160, 80, 96))

    reveal = ease((p - 0.04) / 0.22)
    for k, mask in enumerate(experts):
        x = 270 + (k % 4) * 72
        y = 130 + (k // 4) * 82
        a = ease((reveal * 7 - k) / 0.8)
        tile = map_image(mask, (22, 139, 128), 0.92).resize((58, 58)).convert("RGBA")
        tile.putalpha(int(255 * a))
        d.rounded_rectangle((x - 4, y - 4, x + 62, y + 62), radius=10, fill=(255, 255, 255, int(235 * a)), outline=(211, 222, 229, int(255 * a)))
        frame.alpha_composite(tile, (x, y))
        d.text((x + 17, y + 64), f"E{k + 1}", fill=(71, 82, 94, int(255 * a)), font=SMALL)
    d.text((272, 302), "Seven complementary posteriors", fill=(71, 82, 94, int(255 * reveal)), font=BODY)

    arrow(d, (236, 235), (270, 235), ease((p - 0.12) / 0.08))
    arrow(d, (558, 235), (634, 235), ease((p - 0.28) / 0.08))
    arrow(d, (812, 235), (870, 235), ease((p - 0.62) / 0.08))

    label(d, (620, 150), "Diagnostic state", ease((p - 0.32) / 0.08))
    label(d, (620, 216), "Reliability pool", ease((p - 0.42) / 0.08))
    label(d, (620, 282), "Ambiguity gate", ease((p - 0.52) / 0.08))

    paste_card(frame, (62, 386), "Consensus", map_image(stack, (48, 140, 92), 0.9), "mean + minority evidence", ease((p - 0.30) / 0.12), (48, 140, 92))
    paste_card(frame, (266, 386), "Disagreement", map_image(disagreement, (218, 109, 55), 0.95), "where experts diverge", ease((p - 0.38) / 0.12), (218, 109, 55))
    paste_card(frame, (470, 386), "Reliability", map_image(reliability, (35, 121, 166), 0.95), "validation-anchored trust", ease((p - 0.48) / 0.12), (35, 121, 166))
    paste_card(frame, (674, 386), "Residual", map_image(correction, (136, 96, 172), 0.7), "bounded correction", ease((p - 0.58) / 0.12), (136, 96, 172))
    paste_card(frame, (878, 386), "Fused mask", map_image(final.astype(float), (18, 121, 116), 1), "final vessel posterior", ease((p - 0.70) / 0.16), (18, 121, 116))

    pulse = 0.5 + 0.5 * np.sin(p * np.pi * 8)
    d.rounded_rectangle((860, 132, 1056, 330), radius=18, fill=(255, 255, 255, 235), outline=(205, 219, 228, 255))
    d.text((884, 154), "Fuse only where", fill=(22, 34, 42, 255), font=HEAD)
    d.text((884, 182), "evidence is ambiguous", fill=(22, 34, 42, 255), font=HEAD)
    d.ellipse((925, 226, 994, 295), fill=(18, 121, 116, int(45 + 60 * pulse)), outline=(18, 121, 116, 220), width=4)
    d.text((890, 306), "confident consensus stays stable", fill=(71, 82, 94, 255), font=SMALL)
    return frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128)


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    frames = [frame_at(i) for i in range(FRAMES)]
    frames[0].save(
        OUT,
        save_all=True,
        append_images=frames[1:],
        duration=56,
        loop=0,
        optimize=True,
    )
    assert OUT.exists() and OUT.stat().st_size > 100_000
    print(OUT)


if __name__ == "__main__":
    main()
